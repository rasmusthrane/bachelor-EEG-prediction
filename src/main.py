#%%
# Standard libraries
import os
import sys
import random
from argparse import ArgumentParser
from unicodedata import bidirectional

# Plotting
import matplotlib.pyplot as plt

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.callbacks import ModelCheckpoint 
from pytorch_lightning.utilities.seed import seed_everything

# PyTorch
import mne
import torch
import numpy as np

# Own modules
from data_utils import *
#from lightningTransformer import * 
from lightningTransformer import * 

# Setup devices
cuda=torch.device('cpu')
if torch.cuda.is_available():
    cuda=torch.device('cuda:0')
    print(torch.version.cuda)
    print("current device:",torch.cuda.current_device())
    print("device count:",torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f/1e6)
else:
    print("no cuda available")


#%% 
def main(
    neptuneTags=["bigData"],
    batchSize=10000,
    lr=0.001,
    max_epochs=10,
    patchLength=50,      
    channelIdxs=[1,7,12],
    dropout=0,           
    model_dim=50,        
    num_heads=1,         
    num_layers=1,        
    valSub=0,            
    beforePts=500,       
    afterPts=500,        
    targetPts=100,
    add_positional_encoding=False,
    PEcat_size=0,
    PElearned=False,
    stepsize=None
    ):
    """
    The main function for training.
    Args:
        neptuneTags: tags for neptune.ai
        batchSize: batch size for data loader
        lr: learning rate
        max_epochs: max epochs for training. Early stopping is enabled
        patchLength: number of samples in one patch before and after target
        channelIdxs: selected channels
        dropout: probability of dropout
        model_dim: dimensionality of the representations used as input to the multi-head attention
        num_heads: number of heads for multi-head attention
        num_layers: number of layers of encoder blocks
        valSub: id of validations subject
        beforePts: how many samples included before target
        afterPts: how many samples included after target      
        targetPts: how many samples included in target
        add_positional_encoding: if True use additative PE, if false use concatenated PE
        PEcat_size: Chooses size of concatenated PE. Allowed: 0,1,2 -> codes for [2*, 1.75*, 1.5*]
        PElearned: If true use learnable PE. If false use standard sinusoid method       
        stepsize: When none apply non-overlapping patches. When integer apply overlapping patches with stepsize  
    """
    def transform(raw):
        """
        Transform data by first filtering and then scaling to make it microvolt
        """
        raw.filter(0.1,40, verbose=False)
        raw._data = raw._data*1e6
        return raw
    
    def create_dataloaders(path, valSub, num_workers=24):
        # get subject IDs and create train and validation set based on parameter valSub(ID of subject)
        subjectIds = mb.get_entity_vals(path,'subject',with_key=False)
        trainIds = subjectIds.copy()
        trainIds = np.delete(trainIds, valSub).tolist()
        valIds = [Id for Id in subjectIds if Id not in trainIds]
    
        trainPaths = returnFilePaths(bidsPath,trainIds)
        valPaths = returnFilePaths(bidsPath,valIds,validation=True)

        print('Loading training data')
        # construct train data loader. limit=how many patches we allow. 
        # If none we don't set limit so we can keep generating patches, 
        # but hardcoded limit at numel=100000, meaning we move to validation when the dataloader hits this.
        ds_train = EEG_dataset_from_paths(trainPaths, beforePts=beforePts,afterPts=afterPts,targetPts=targetPts,
                                        channelIdxs=channelIdxs,preprocess=False,limit=100000,transform=transform)

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, num_workers=num_workers)
        print("batchSize = ", batchSize)
        print("dimension of data loader = ", len(dl_train.dataset))
        print('Loading validation data, subject = ', valIds)

        # construct validation data loader. 
        # limit should be finite as we cannot compare progress if we keep getting patches. 
        ds_val = EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,targetPts=targetPts,
                                      channelIdxs=channelIdxs,preprocess=False,limit=100000,transform=transform)

        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batchSize, num_workers=num_workers)

        return dl_train, dl_val

    # where are we?    
    tempPath ="D:\databachelor"
    if os.path.isdir(tempPath):
        bidsPath = tempPath

    num_cpus = 24
    gpu_idx = 0

    # create dataloaders
    dl_train, dl_val = create_dataloaders(bidsPath, valSub, num_workers=num_cpus)
    max_iters = len(dl_train)*max_epochs # one iter is the number of batches needed to complete one epoch

    # construct network
    net = TransformerPredictor(contextSize=beforePts+afterPts,output_dim=targetPts,patchLength=patchLength, 
                                                   model_dim=model_dim, num_heads=num_heads, num_layers=num_layers,
                                                   lr=lr, 
                                                    warmup=500,max_iters=max_iters,
                                                    #warmup=300,max_iters=15000,
                                                                dropout=dropout, input_dropout=dropout,
                                                   add_positional_encoding=add_positional_encoding, stepsize=stepsize, 
                                                                PEcat_size=PEcat_size, PElearned=PElearned)

    # EarlyStopping ensures that we don't keep training when nothing new happens. 
    # patience parameter counts the number of validation checks with no improvement.
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=False, mode="min")
    model_checkpoint = ModelCheckpoint(
        monitor='val_loss', save_top_k=1, dirpath='checkpoints', filename="{epoch}-{step}-{val_loss:.2f}")
    
    # Setup logging with NeptuneLogger
    neptune_token = ""
    neptune_logger = pl.loggers.NeptuneLogger(
           api_token=neptune_token,project="NTLAB/eegPrejShare", 
           source_files=["main.py", "data_utils.py", "lightningTransformer.py"],tags=neptuneTags)
    
    # train using pytorchlightning trainer
    trainer = pl.Trainer(accelerator='gpu', devices=[gpu_idx],
                        #logger = neptune_logger,
                        callbacks = [
                            #early_stop_callback, 
                            #model_checkpoint
                        ],
                        max_epochs=max_epochs,deterministic=True)

    # fit model
    trainer.fit(net,dl_train,dl_val)

    print('Done')
    neptune_logger.finalize('Success')

    return trainer,net


#%% run experiment
if __name__ == "__main__": 
    seed_everything(42, workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    parser = ArgumentParser()
    parser.add_argument("--neptuneTags", nargs='+', type=str, default=["experiment"])
    parser.add_argument("--batchSize", default=10000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--max_epochs", default=1500, type=int)
    parser.add_argument("--patchLength", default=50, type=int)
    parser.add_argument('--channelIdxs', nargs='+', type=int, default=[3, 9, 17, 18, 19, 21, 22, 23])
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--model_dim", default=50, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--valSub", nargs='+', type=int, default=[0])
    parser.add_argument("--beforePts", default=500, type=int)
    parser.add_argument("--afterPts", default=500, type=int)
    parser.add_argument("--targetPts", default=100, type=int)
    parser.add_argument("--add_positional_encoding", action='store_true')
    parser.add_argument("--PEcat_size", default=0, type=int)
    parser.add_argument("--PElearned", action='store_true')  
    parser.add_argument("--stepsize", default=None, type=int)

    args = parser.parse_args()

    main(args.neptuneTags,args.batchSize,args.lr,args.max_epochs,args.patchLength,
         args.channelIdxs,args.dropout,args.model_dim,args.num_heads,args.num_layers,
         args.valSub,args.beforePts,args.afterPts,args.targetPts,args.add_positional_encoding,
         args.PEcat_size,args.PElearned,args.stepsize)

# hyperparam opt https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html

