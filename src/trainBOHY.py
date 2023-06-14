# Standard libraries
import os
import sys
import random
from argparse import ArgumentParser
from unicodedata import bidirectional
from math import floor
from math import ceil

# Plotting
import matplotlib.pyplot as plt

# Saving best hyp parameters
import json

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.callbacks import ModelCheckpoint 
from pytorch_lightning.utilities.seed import seed_everything

# PyTorch
import mne
import torch
import numpy as np

# Ray Tuner
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

# Own modules
from data_utils import *
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

# the transform we will be using
def transform(raw):
    """
    Transform data by first filtering and then scaling to make it microvolt
    """
    raw.filter(0.1,40, verbose=False)
    raw._data = raw._data*1e6
    return raw

# 1. Function that creates dataloders
def create_dataloaders(path, valSub, batchSize, beforePts, afterPts, targetPts, 
                        channelIdxs, transform, valLimit=100000, numCpus=1):
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
                                    channelIdxs=channelIdxs,preprocess=False,limit=None,transform=transform, RayTune=True)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, num_workers=numCpus)
    print("batchSize = ", batchSize)
    print("dimension of data loader = ", len(dl_train.dataset))
    print('Loading validation data, subject = ', valIds)

    # construct validation data loader. 
    # limit should be finite as we cannot compare progress if we keep getting patches. 
    ds_val = EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,targetPts=targetPts,
                                  channelIdxs=channelIdxs,preprocess=False,limit=valLimit,transform=transform)

    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batchSize, num_workers=numCpus)

    return dl_train, dl_val

# 2. Training function

def train(config, data_dir, numCpus, numGpus):
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
        stepsize: When none apply non-overlapping patches. When integer apply overlapping patches with stepsize  
    """
    # unpack config
    neptuneTags = config["neptuneTags"]
    batchSize = config["batchSize"]
    lr = config["lr"]
    max_epochs = config["max_epochs"]
    patchLength = config["patchLength"]
    channelIdxs = config["channelIdxs"]
    dropout = config["dropout"]
    model_dim = config["model_dim"]
    num_heads = config["num_heads"]
    num_layers = int(config["num_layers"])
    valSub = config["valSub"]
    targetPts = config["targetPts"]
    beforePts = config["beforePts"]
    afterPts = config["beforePts"]
    add_positional_encoding = config["add_positional_encoding"]
    stepsize = config["stepsize"]
    
    # transform stepsize to None as tune.choice() does not allow NoneType
    if isinstance(stepsize, str):
        stepsize = None
        
    # create dataloaders
    dl_train, dl_val = create_dataloaders(data_dir, valSub, batchSize, beforePts, afterPts, targetPts, 
                                            channelIdxs, transform,valLimit=100000, numCpus=numCpus)

    # construct network
    net = TransformerPredictor(contextSize=beforePts+afterPts, output_dim=targetPts, patchLength=patchLength, 
                                                   model_dim=model_dim, num_heads=num_heads, num_layers=num_layers,
                                                   lr=lr, warmup=300,max_iters=15000, dropout=dropout, input_dropout=dropout,
                                                   add_positional_encoding=add_positional_encoding,
                                                    stepsize=stepsize,linearPredictor=False)

    # Setup callbacks                                            
    # EarlyStopping ensures that we don't keep training when nothing new happens. 
    # patience parameter counts the number of validation checks with no improvement.
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=25, verbose=False, mode="min")
    model_checkpoint = ModelCheckpoint(
        monitor='val_loss', save_top_k=1, dirpath='checkpoints', filename="{epoch}-{step}-{val_loss:.2f}")
    tune_callback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")
    
    # Setup logging with NeptuneLogger
    neptune_token = ""
    neptune_logger = pl.loggers.NeptuneLogger(
           api_token=neptune_token,project="<name>", 
           source_files=["trainBOHY.py", "data_utils.py", "lightningTransformer.py"],tags=neptuneTags)

    # train using pytorchlightning trainer
    trainer = pl.Trainer(accelerator='gpu',
                        gpus = ceil(numGpus),
                        #devices=[0],
                        logger = neptune_logger,
                        callbacks = [
                            #early_stop_callback, 
                            #model_checkpoint, 
                            tune_callback
                        ],
                        max_epochs=max_epochs,deterministic=True)  

    # fit model
    trainer.fit(net,dl_train,dl_val)
    print('Done')

    neptune_logger.finalize('Success')                                 

if __name__ == "__main__": 
    seed_everything(42, workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    parser = ArgumentParser()
    parser.add_argument("--neptuneTags", nargs='+', type=str, default=["experiment"])
    parser.add_argument("--batchSize", default=10000, type=int)
    #parser.add_argument("--lr", default=tune.uniform(0.0001, 0.01))
    parser.add_argument("--lr", default=tune.uniform(0.0001, 0.001))
    parser.add_argument("--max_epochs", default=1500, type=int)
    #parser.add_argument("--patchLength", default=tune.choice(np.arange(30, 110, step=10)))
    parser.add_argument("--patchLength", default=tune.choice(np.arange(60, 110, step=10)))
    parser.add_argument('--channelIdxs', nargs='+', type=int, default=[3, 9, 17, 18, 19, 21, 22, 23])
    parser.add_argument("--dropout", default=tune.uniform(0,1), type=float)
    #parser.add_argument("--model_dim", default=tune.choice(np.arange(10, 110, step=10)))
    parser.add_argument("--model_dim", default=tune.choice([30,60,90]))
    #parser.add_argument("--num_heads", default=tune.choice([1,2,5,10]))
    parser.add_argument("--num_heads", default=tune.choice([2,3,5,6]))
    #parser.add_argument("--num_layers", default=tune.choice([1,2,3,4,5,6]))
    parser.add_argument("--num_layers", default=tune.choice([3,4,5,6,7,8]))
    parser.add_argument("--valSub", nargs='+', type=int, default=[0])
    parser.add_argument("--beforePts", default=tune.choice(np.arange(200,550, step=50)),type=int)
    parser.add_argument("--targetPts", default=tune.choice(np.arange(80,320, step=20)), type=int)
    parser.add_argument("--add_positional_encoding", action='store_true')
    #parser.add_argument("--stepsize", default=tune.choice([10, 20, 'None']))
    parser.add_argument("--stepsize", default=tune.choice([5, 10, 20, 30, 40, 50, 60]))

    config = vars(parser.parse_args())
    
    # defining a path to data
    #tempPath ="/home/roman/bachelor/data/small"
    tempPath ="<data>"
    
    # naming the tuning experiment, very primitive

    name = "tune_result"
    exp_name = name +"0"
    i = 0
    while os.path.exists("/../ray_results/%s%s" % (name, i)):
        i += 1
    
    exp_name = name + f"{i}"
    #exp_name = "tune_result_Hyp_Instance_Norm0"
    
    if os.path.isdir(tempPath):
        bidsPath = tempPath
    
    number_of_trials = 200
    cpus_per_trial = 12
    gpus_per_trial = 1
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}    
    
    bayesopt = BayesOptSearch()
    
    # used together
    algo = TuneBOHB()
    bohb = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=config["max_epochs"],
            reduction_factor=4)
    
    train_fn_with_parameters = tune.with_parameters(
                train, data_dir=bidsPath, numCpus=cpus_per_trial, numGpus=gpus_per_trial
    )

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        run_config=air.RunConfig(
            name=exp_name),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=bohb,
            #search_alg=bayesopt,
            search_alg=algo,
            num_samples=number_of_trials
        ),
        param_space=config,
    )
    #tuner = tune.Tuner.restore("/home/roman/ray_results/tune_result_Hyp_Instance_Norm_experiement2",
    #                          trainable=train_fn_with_parameters)

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    # from https://stackoverflow.com/questions/50916422/python-typeerror-object-oftype-int64-is-not-json-serializable
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    
    # write to json files containing best hyp params and metrics for best run
    json_object1 = json.dumps(results.get_best_result().config, indent=4, cls=NpEncoder)
    json_object2 = json.dumps(results.get_best_result().metrics, indent=4, cls=NpEncoder)
 
    # Writing to json
    with open("/../ray_results/%s/hyp.json" % exp_name, "w") as outfile:
        outfile.write(json_object1)
    with open("/../ray_results/%s/hypResults.json" % exp_name, "w") as outfile:
        outfile.write(json_object2)

    df = results.get_dataframe()
    df.to_csv("../ray_results/%s/all_trials.csv" % exp_name)
    df = df.nsmallest(5, 'loss')
    df.to_csv("../ray_results/%s/best_trials.csv" % exp_name)  
    

    
    