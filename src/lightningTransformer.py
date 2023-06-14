#from https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html:

#%%

# Standard libraries
import math
import os
import sys
from functools import partial

# Linear Model
from sklearn.linear_model import LinearRegression as LR

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# PyTorch Lightning
import pytorch_lightning as pl
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

# Own modules
from data_utils import *


# References
#https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

#https://github.com/qingsongedu/time-series-transformers-review

#https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial

#https://huggingface.co/docs/transformers/create_a_model


#%% transformer architecture:

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Implements the 'encoder' block from the original transformer, residual connections included

        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers

            The 'encoder' block from the original transformer, residual connections included

        """
        super().__init__()

        # Attention layer - uses same dimension for k, q and v. 
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, x, x)[0] #ignoring second output
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        """
        Implements full Transformer encoder

        Args:
            num_layers: number of encoder blokcs
        """
        
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            #_, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            _, attn_map = layer.self_attn(x, x, x)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, learned=False):
        """
        Implements the positional encoding using sinusoidals 

        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()
        
        if learned:
            # Learnable PE
            self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        else:
            # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
            # Used for tensors that need to be on the same device as the module.
            # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        """
        Implements warm-up with a cosine-shaped learning rate decay

        Args
            optimizer: What optimizer is being used
            warmup: When to start decaying
            max_iters: Number of maximum iterations the model is trained for. We need to know at what rate to decay
        """

        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class TransformerPredictor(pl.LightningModule):
    def __init__(
        self,
        output_dim,
        patchLength, 
        contextSize,
        model_dim,
        num_heads,
        num_layers,
        lr,
        warmup=100,
        max_iters=1000,
        dropout=0.0,
        input_dropout=0.0,
        linearPredictor=None,
        stepsize=None,
        add_positional_encoding=False,
        PEcat_size=0,
        PElearned=False
    ):
        """
        Implementation of the entie architecture

        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_classes: Number of classes to predict per sequence element
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            num_layers: Number of encoder blocks to use.
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
            input_dropout: Dropout to apply on the input features
            stepsize: When none apply non-overlapping patches. When integer apply overlapping patches with stepsize 
            add_positional_encoding: if True use additative PE, if false use concatenated PE 
            PEcat_size: Chooses size of concatenated PE. Allowed: 0,1,2 -> codes for [2*, 1.75*, 1.5*]
            PElearned: If true use learnable PE. If false use standard sinusoid method       
        """
        super().__init__()
        self.save_hyperparameters()
        #assert(contextSize%patchLength==0)
        assert PEcat_size in [0,1,2], f"PEcat_size shoud be 0,1 or 2, got: {PEcat_size}"
        
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.patchLength, self.hparams.model_dim)
        )
        # choose which cat size we are using
        PEcat_sizes = [2, 1.75, 1.5]
        self.PEcat_size = PEcat_sizes[self.hparams.PEcat_size]

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim, learned=self.hparams.PElearned)

        # no stepsize corresponds to a step size of patchLength
        if self.hparams.stepsize is None:
            self.hparams.stepsize = self.hparams.patchLength
        


        # to calculate nPatches use nPatches=(sequence-patchlength)/stepsize + 1 
        # multiply by 2 as we have before and after target
        nPatches = 2*(math.floor((self.hparams.contextSize/2-self.hparams.patchLength)/self.hparams.stepsize) + 1)
 
        # Transformer
        # if we add PE then input_dim stays the same as model_dim
        #self.add_positional_encoding = True
        #self.hparams.add_positional_encoding
        if  self.hparams.add_positional_encoding:
            self.transformer = TransformerEncoder(
                num_layers=self.hparams.num_layers,
                input_dim=self.hparams.model_dim,
                dim_feedforward=2 * self.hparams.model_dim,
                num_heads=self.hparams.num_heads,
                dropout=self.hparams.dropout,
            )
            # Output layer
            transFormerOutSize=int(nPatches*self.hparams.model_dim)
            
        # if we concatenate PE at the end of x then the model_dim of x will be changed according to PEcat_size
        else:
            #self.add_positional_encoding = False
            self.transformer = TransformerEncoder(
                num_layers=self.hparams.num_layers,
                input_dim=math.ceil(self.PEcat_size*self.hparams.model_dim),
                dim_feedforward=2 * self.hparams.model_dim,
                num_heads=self.hparams.num_heads,
                dropout=self.hparams.dropout,
            )
            # Output layer
            transFormerOutSize = int(nPatches*math.ceil(self.PEcat_size*self.hparams.model_dim))
            
        # Linear out
        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(transFormerOutSize,self.hparams.output_dim),
            nn.PReLU(),
            nn.Linear(self.hparams.output_dim,self.hparams.output_dim),
            nn.PReLU(),
            nn.Linear(self.hparams.output_dim,self.hparams.output_dim)
        )
        
        #self.linearPredictor=False
        #self.hparams.contextSize, self.hparams.output_dim

        if linearPredictor:
            # Only used for the linear predictor
            self.linear_pred_layer = nn.Linear(self.hparams.contextSize, self.hparams.output_dim)
        ##    #make sure the linear predictor is compatible with the task:
        ##    assert linearPredictor.coef_.shape[0]==output_dim
        ##    assert linearPredictor.coef_.shape[1]==contextSize
         #   self.linearPredMat = torch.nn.parameter.Parameter(data=torch.tensor((self.hparams.output_dim, self.hparams.contextSize)), requires_grad=False)
         #   self.linearPredIntercept = torch.nn.parameter.Parameter(data=torch.tensor(self.hparams.output_dim),requires_grad=False)
         #   self.linearPredictor=True

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
            
        else:
            pe = self.positional_encoding.pe
            pe = pe[:, : x.size(1)]
            # for experimenting with PE sizes where we make PE shorter in last dimension
            # when PEcat_size==2 nothing happens, but if 1.75 or 1.5 we shorten
            pe = pe[:,:,:math.ceil(x.shape[2]*(self.PEcat_size-1))]
            # concatenate for each batch by broadcasting 
            x = torch.cat((x,pe.expand(x.shape[0],x.shape[1],-1)),dim=-1)
            
        attention_maps = self.transformer.get_attention_maps(x)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration



    def patchingFunc(self, x, patchLength, stepsize=None, afterTarget=False):
        """
        Function used to patch a signal. Assumes input is (batch,sequence). Returns a list of patches
        when stepsize=None create disjoint patches, returns (batch,sequence/patchLength,patchLength)
        when stepsize is an integer create patch using a sliding window based on stepsize 
        when looking at seq before target we need to ensure correct patching
        returns (batch, #patches, patchlength) where #patches=floor((sequence-patchlength)/stepsize) + 1
        """ 
        # disjoint patches when stepsize=patchLength 
        if stepsize is None:
            stepsize=patchLength  
            
        # if we are looking at sequence after target 
        if afterTarget:
            # (batch, patchIdx, patchlength)
            return x.unfold(1,patchLength,stepsize)
        
        # if we are looking at sequence before target we need to flip it to ensure we start patching
        # from left edge of target
        else:
            xrev = torch.flip(x, dims = [1])
            # (batch, patchIdx, patchlength)
            return torch.flip(xrev.unfold(1,patchLength,stepsize), dims = [1,2])

    def _forward_only(self,x1,x2,mask=None):
        xInput=torch.cat((x1,x2),dim=1)
        # We try to standardize each signal instance before patching and then add them back to output prediction
        # mitigating the distribution shift effect between the training and testing data
        # (Instance normalization from Nie et al.)
        xmeans = xInput.mean(dim=1, keepdim=True)
        xstds = xInput.std(dim=1, keepdim=True)
        x_normalized = (xInput - xmeans) / xstds
        x_split = torch.split(x_normalized,int(self.hparams.contextSize/2),dim=1)
        x1_normalized = x_split[0]
        x2_normalized = x_split[1]
        #patch x1 and x2: returns as (batch,sequence/patchLength,patchLength)
        x1=self.patchingFunc(x1_normalized, self.hparams.patchLength, self.hparams.stepsize)
        x2=self.patchingFunc(x2_normalized, self.hparams.patchLength, self.hparams.stepsize, afterTarget=True)
    
        x=torch.cat((x1,x2),dim=1)

        #forward pass
        x = self.input_net(x)
        if self.hparams.add_positional_encoding:
            x = self.positional_encoding(x)
        else:
            pe = self.positional_encoding.pe
            pe = pe[:, : x.size(1)]
            # for experimenting with PE sizes where we make PE shorter in last dimension
            # when PEcat_size==2 nothing happens, but if 1.75 or 1.5 we shorten
            pe = pe[:,:,:math.ceil(x.shape[2]*(self.PEcat_size-1))]
            # concatenate for each batch by broadcasting 
            x = torch.cat((x,pe.expand(x.shape[0],x.shape[1],-1)),dim=-1)
            
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        x = x*xstds + xmeans
    
        return x

    def _forward_only_LP(self,x1,x2,mask=None):
        x = torch.cat((x1,x2),dim=1)
        x = self.linear_pred_layer(x)

        return x


    def _forward_with_loss(self, batch, mode="train",mask=None):
        torch.autograd.set_detect_anomaly(True)
        x,y=batch
        #xInput=torch.cat(x,dim=1) #in case we want to do linear Prediction later
        #print("target mean:", torch.mean(y,dim=1))
        #print("target std:", torch.std(y,dim=1))
        x1,x2=x #splitting in before and after context

        #making sure things can be patched neatly:
        #assert x1.shape[1]%self.hparams.patchLength==0 , print("x1 has wrong shape", x1.shape,self.hparams.patchLength)
        #assert x2.shape[1]%self.hparams.patchLength==0, print("x2 has wrong shape", x2.shape,self.hparams.patchLength)
        assert y.shape[1]==self.hparams.output_dim, print("y has wrong shape", y.shape,self.hparams.patchLength)

        if self.hparams.linearPredictor:
            x = self._forward_only_LP(x1,x2)

        else:
            x = self._forward_only(x1,x2)

        #loss
        loss = F.mse_loss(x, y)
        loss_l1 = F.l1_loss(x, y)
        
        return loss, loss_l1


    # These functions define the different loggers according to 
    # https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.NeptuneLogger.html#lightning.pytorch.loggers.NeptuneLogger
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        loss, _ = self._forward_with_loss(batch, mode="train")
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("lr", self.lr_scheduler.get_lr()[0], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._forward_with_loss(batch, mode="val")

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_l1 = self._forward_with_loss(batch, mode="test")

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss_l1", loss_l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'test_loss': loss, 'test_loss_l1': loss_l1}


#%%
#
if __name__=="__main__":

    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    patchLength=50
    outputDim=100
    batchSize=10
    max_epochs=10    
    channelIdxs=[1,7,12]
    dropout=0          
    model_dim=50    
    num_heads=2        
    num_layers=7        
    #valSub=0
    valSub=[0]    
    beforePts=500
    afterPts=500        
    targetPts=300
    add_positional_encoding=True
    stepsize=None
    PEcat_size=1 # 0 yields *2
    PElearned=False

    transf=TransformerPredictor(contextSize=beforePts+afterPts,output_dim=targetPts,patchLength=patchLength, 
                                                   model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, lr=0.001, warmup=300,
                                                   max_iters=15000, dropout=dropout, input_dropout=dropout,
                                add_positional_encoding=add_positional_encoding,stepsize=stepsize,
                                                    PEcat_size=PEcat_size, PElearned=PElearned)
    
    bidsPath = "D:\databachelor"
    subjectIds = mb.get_entity_vals(bidsPath,'subject',with_key=False)
    trainIds = subjectIds.copy()
    trainIds = np.delete(trainIds, valSub).tolist()
    valIds = [Id for Id in subjectIds if Id not in trainIds]
    
    trainPaths = returnFilePaths(bidsPath,trainIds)
    valPaths = returnFilePaths(bidsPath,valIds,validation=True)

    ds_train = EEG_dataset_from_paths(trainPaths, beforePts=beforePts,afterPts=afterPts,targetPts=targetPts,
                                    channelIdxs=channelIdxs,preprocess=False,limit=100,transform=None)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, num_workers=24)

    ds_val = EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,targetPts=targetPts,
                                  channelIdxs=channelIdxs,preprocess=False,limit=100,transform=None)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batchSize, num_workers=24)

    trainer = pl.Trainer(accelerator='gpu', devices=[1],max_epochs=max_epochs)
    trainer.fit(transf, dl_train,dl_val)
#
##%%
    


