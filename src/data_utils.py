import torch
import os
import sys
import re
import mne
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
import mne_bids as mb

def directory_spider(input_dir, path_pattern="", file_pattern="", maxResults=500):
    file_paths = []
    if not os.path.exists(input_dir):
        raise FileNotFoundError("Could not find path: %s"%(input_dir))
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if re.search(path_pattern, dirpath):
            file_list = [item for item in filenames if re.search(file_pattern,item)]
            file_path_list = [os.path.join(dirpath, item) for item in file_list]
            file_paths += file_path_list
            if len(file_paths) > maxResults:
                break
    return file_paths[0:maxResults]

def returnFilePaths(bidsDir,subjectIds=None,sessionIds=None,taskIds=None, validation=False):
    """ A wrapper for get_entity_vals and BIDSPath, to get all files matching certain ID's """
    
    def debugMessage(input,inputName):
        if type(input) is not list:
            raise Exception( "returnFilepaths expects a list or None for " + inputName + " Id's. Consider enclosing id in '[]'" )

    #list of subjects:
    if not subjectIds:
        subjectIds=mb.get_entity_vals(bidsDir,'subject')
        if len(subjectIds)==0:
            subjectIds=[None]
    debugMessage(subjectIds,'subject')


    #list of sessions:
    if not sessionIds:
        sessionIds=mb.get_entity_vals(bidsDir,'session')
        if len(sessionIds)==0:
            sessionIds=[None]
    debugMessage(sessionIds,'session')

    #list of tasks:
    if not taskIds:
        taskIds=mb.get_entity_vals(bidsDir,'task')
        if len(taskIds)==0:
            taskIds=[None]
    debugMessage(taskIds,'task')

    # during validation we only use 4 first sessions as they contain all channels
    if validation:
        sessionIds = sessionIds[:4]

    print(subjectIds,sessionIds,taskIds)

    #and here we just check and add all possible combinations:
    filePaths=[]
    for sub in subjectIds:
        for ses in sessionIds:
            for task in taskIds:
                try:
                    temp=mb.BIDSPath(root=bidsDir,subject=sub,session=ses,task=task,datatype='eeg',extension='.set',check=False)
                    if os.path.isfile(str(temp)):
                        filePaths.append(str(temp))
                except Exception as error:
                    print(error)
                    print(sub,ses,task)

    return filePaths

def plotResult(x1, x2, y, yhat, beforePts, afterPts, targetPts, axis, idx=0):
    """
    Input can be tensor
    """
    x1 = x1[idx,:]
    x2 = x2[idx,:]
    y = y[idx,:]
    yhat = yhat[idx,:]
    
    # background plot
    axis.plot(np.arange(0,beforePts), x1)
    axis.plot(np.arange(beforePts+targetPts, beforePts+afterPts+targetPts), x2, color='#1f77b4')
    # target segment plot
    target_axis = np.arange(beforePts-1,beforePts+targetPts+1)
    target_values = np.concatenate((x1[-1:].detach().numpy(), y.detach().numpy(), x2[:1].detach().numpy()))
    axis.plot(target_axis, target_values)
    # pred plot
    pred_values = np.concatenate((x1[-1:].detach().numpy(), yhat.detach().numpy(), x2[:1].detach().numpy()))
    axis.plot(target_axis, pred_values)


class EEG_dataset_from_paths(torch.utils.data.Dataset):
    """
    Class to genererate EEG dataset from path.

    Args:
        bidsPaths: paths to sessions
        beforePts: length of pts before target
        afterPts: length of pts after target
        targetPts: length of target
        channelIdxs: channels to include
        transform: datatransform e.g. filter
        limit: max number of patches to generate
        validation: if validation dataset
    """
    def __init__(self, bidsPaths, beforePts,afterPts,targetPts, 
    channelIdxs, transform=None,preprocess=False,limit=None, RayTune=False):
        self.transform = transform
        self.beforePts = beforePts
        self.afterPts = afterPts
        self.targetPts = targetPts
        self.channelIdxs = channelIdxs
        self.nChannels = len(channelIdxs) if isinstance(channelIdxs, (list,tuple,range)) else 1
        self.file_paths = [str(fp) for fp in bidsPaths]
        self.limit = limit #if

        maxFilesLoaded = self.determineMemoryCapacity()

        #preload:
        self.raws = []
        nfilesToLoad = min(maxFilesLoaded,len(self.file_paths))
        # choose random sessions from random subject
        fileIdxToLoad = np.random.choice(len(self.file_paths),nfilesToLoad,replace=False)
        for fileIdx in fileIdxToLoad:
            tempRaw = mne.io.read_raw_eeglab(self.file_paths[fileIdx],preload=True,verbose=False)
            # if we only pick one channel just use that
            # else use all channels available
            if self.nChannels == 1:
                tempRaw.pick(self.channelIdxs)
            else:
                tempRaw.pick([idx for idx in self.channelIdxs if idx in np.arange(0,len(tempRaw.info["ch_names"]))])
            if self.transform:
                    tempRaw = self.transform(tempRaw)
            self.raws.append(tempRaw)
        
        # if we set a limit on how many patches we want
        if limit:
            self.dataDraws=np.zeros((self.__len__(),3),np.int64) #columns for: file, channel, time
            print('Preparing ready-made data draws...')


            #def myfun(arg):
            #    result=self.getAllowedDatapoint()
            #    return result

            def myfun():
                result=self.getAllowedDatapoint()
                return result

            def _get_reproducible_Datapoint(seed):
                np.random.seed(seed)
                return myfun()
            
            seeds = np.random.randint(1e8, size=self.__len__())
            
            if RayTune:
                # Use RayTune multiGPU
                for i in range(self.__len__()):
                    randFileIdx,channelIdx,randomIdx=self.getAllowedDatapoint()
                    self.dataDraws[i,0]=randFileIdx
                    self.dataDraws[i,1]=channelIdx
                    self.dataDraws[i,2]=randomIdx
            else:
                # Use Parallel multiGPU
                par = Parallel(n_jobs=np.max((1,cpu_count()-1)), verbose=1, backend="threading")
                results = par(delayed(_get_reproducible_Datapoint)(seed) for seed in seeds)
                self.dataDraws = np.asarray(results)

            

    def determineMemoryCapacity(self):
        """A function that determines how much space we can use for pre-loaded data"""
        
        import psutil
        freeMemory = psutil.virtual_memory().available
        print("Detected free memory:",freeMemory / (1024**3),"GB")

        fileSizeMax = 10*3600*250 #10 hours of data at 250Hz
        fileSizeMax = fileSizeMax*self.nChannels
        fileSizeMax *= 64/8 #size of a 10 hr night in bytes

        nFiles = int(freeMemory/fileSizeMax)
        print("This will fit approximately %s files with %s channels each"%(nFiles,self.nChannels))
        print('')

        return nFiles

    def getAllowedDatapoint(self,returnData=False):
        """
        A function that finds a random window without nan's
        Args:
            returnData: default False. When True we return data containing a window
        """

        windowSize = self.beforePts+self.afterPts+self.targetPts
        #keep looking until we find a data window without nan's
        data = np.nan

        # while data is containing nan's, select a new random window
        # when we find a valid window, exit while loop
        while np.any(np.isnan(data)):
            randFileIdx = np.random.randint(0,len(self.raws))       
            randomChannelIdx = np.random.choice(len(self.raws[randFileIdx].ch_names))
            randomIdx = np.random.randint(0,self.raws[randFileIdx].n_times-windowSize)
            #print("length of raws:", len(self.raws))
            #print("file idx:", randFileIdx)
            #print("channel idx:", randomChannelIdx)
            #print("random idx signal:", randomIdx)

            data,_ = self.raws[randFileIdx][randomChannelIdx,randomIdx:randomIdx+windowSize]
        #print(f"File idx: {randFileIdx}, channel idx: {randomChannelIdx}, sample idx: {randomIdx}")
        if returnData:
            return randFileIdx,randomChannelIdx,randomIdx,data
        else:
            return randFileIdx,randomChannelIdx,randomIdx


    def __len__(self):
        if self.limit:
            numel=self.limit
        else:
            # the dataloader needs a length, so we just put it at numel=100000
            # numel=int(np.sum([raw.n_times for raw in self.raws])*self.nChannels/2)
            numel=100000

        return numel

    def __getitem__(self, idx):
        windowSize = self.beforePts+self.afterPts+self.targetPts

        if self.limit:
            #uses a predefined list of data points
            fileIdx = self.dataDraws[idx,0]
            channelIdx = self.dataDraws[idx,1]
            randomIdx = self.dataDraws[idx,2]
            data,_ = self.raws[fileIdx][channelIdx,randomIdx:randomIdx+windowSize]
        else:
            #randomly selects a data point from all possible. As we can generate infinite
            #patches we continue until dataloader thinks we are done:
            fileIdx, channelIdx,randomIdx,data = self.getAllowedDatapoint(returnData=True)
    
        #make sure there are no nan's in the data:
        assert(not np.any(np.isnan(data)))

        data = torch.tensor(data, dtype=torch.float32) 
        # x12 is the data before and after target window
        x12 = (data[0,0:self.beforePts],data[0,-self.afterPts:])
        # target is the window we want to predict
        target = data[0,self.beforePts:(-self.afterPts)]


        return x12,target


if __name__ == "__main__":
    bidsPath = "../data/overfit"
    subjectIds = mb.get_entity_vals(bidsPath,'subject',with_key=False)
    trainIds = subjectIds.copy()
    trainPaths = returnFilePaths(bidsPath,trainIds)

    beforePts=500
    afterPts=500
    targetPts=100
    batchSize = 100
    limit = 1000
    channelIdxs=[3, 9, 17, 18, 19, 21, 22, 23]


    ds_train = EEG_dataset_from_paths(trainPaths, beforePts=beforePts,afterPts=afterPts,targetPts=targetPts,
                                    channelIdxs=channelIdxs,preprocess=False,limit=limit,transform=None)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, num_workers=24)

    instance = iter(dl_train)
    x12, target = instance.next()