import sys
import os
import torch
from joblib import dump, load
import pandas as pd

# sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression as LR

# PyTorch Lightning
import pytorch_lightning as pl

# Own modules
from data_utils import *
from lightningTransformer import * 


if __name__=="__main__":

    # Setting the seed
    pl.seed_everything(42)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    def transform(raw):
        """
        Transform data by first filtering and then scaling to make it microvolt
        """
        raw.filter(0.1,40, verbose=False)
        raw._data = raw._data*1e6
        return raw
    
    beforePts = 400
    afterPts = 400
    targetPts = 80
    channelIdxs = [3, 9, 17, 18, 19, 21, 22, 23]
    batchSize = 5000
    limit = 200000
    #batchSize = 10
    #limit = 200
    trainsize = 200000
    num_gpus = 24
    gpu_idx = 0
    train_models = False
    
    # load top 5 model and trainer
    model_path1 = "model_checkpoints/transf1.ckpt"
    model_path2 = "model_checkpoints/transf2.ckpt"
    model_path3 = "model_checkpoints/transf3.ckpt"
    model_path4 = "model_checkpoints/transf4.ckpt"
    model_path5 = "model_checkpoints/transf5.ckpt"
    
    model1 = TransformerPredictor.load_from_checkpoint(model_path1)
    model2 = TransformerPredictor.load_from_checkpoint(model_path2)
    model3 = TransformerPredictor.load_from_checkpoint(model_path3)
    model4 = TransformerPredictor.load_from_checkpoint(model_path4)
    model5 = TransformerPredictor.load_from_checkpoint(model_path5)

    trainer = pl.Trainer(accelerator='gpu', devices=[gpu_idx])
    
    # find test data
    bidsPath = "../data/testing"
    subjectIds = mb.get_entity_vals(bidsPath,'subject',with_key=False)
    testIds = subjectIds.copy()

    if train_models:
        # load train data for sklearn
        trainPath = "../data/training"
        subjectIdsTrain = mb.get_entity_vals(trainPath,'subject',with_key=False)
        trainIds = subjectIdsTrain.copy()    
        trainPaths = returnFilePaths(trainPath,trainIds)
        ds_train = EEG_dataset_from_paths(trainPaths, beforePts=beforePts,afterPts=afterPts,targetPts=targetPts,
                                            channelIdxs=channelIdxs,preprocess=False,limit=trainsize,transform=transform)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=trainsize, shuffle=True, num_workers=num_gpus)

        # train models for benchmark on same data
        Xtrain, ytrain = next(iter(dl_train))
        Xtrain = torch.concat((Xtrain[0],Xtrain[1]),dim=1)

        LR_model = LR()
        LR_model.fit(Xtrain, ytrain)
        dump(LR_model, 'model_checkpoints/linear.joblib') # save model for later use
        spline_model = make_pipeline(SplineTransformer(n_knots=20, degree=3), Ridge(alpha=1e-1))
        spline_model.fit(Xtrain, ytrain)
        dump(spline_model, 'model_checkpoints/spline.joblib') # save model for later use
    else:
        LR_model = load('model_checkpoints/linear.joblib')
        spline_model = load('model_checkpoints/spline.joblib')


    # We test on a subject basis and compare with each model
    avg_l1_loss_transf1 = []
    avg_l1_loss_transf2 = []
    avg_l1_loss_transf3 = []
    avg_l1_loss_transf4 = []
    avg_l1_loss_transf5 = []
    avg_l1_loss_spline = []
    avg_l1_loss_linear = []

    avg_l2_loss_transf1 = []
    avg_l2_loss_transf2 = []
    avg_l2_loss_transf3 = []
    avg_l2_loss_transf4 = []
    avg_l2_loss_transf5 = []
    avg_l2_loss_spline = []
    avg_l2_loss_linear = []
    for sub in testIds:
        testPaths = returnFilePaths(bidsPath,[sub])
        ds_test = EEG_dataset_from_paths(testPaths, beforePts=beforePts,afterPts=afterPts,targetPts=targetPts,
                                    channelIdxs=channelIdxs,preprocess=False,limit=limit,transform=transform)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batchSize, num_workers=num_gpus)

        instance = iter(dl_test)

        #loss_transf = []
        l1_loss_spline = []
        l1_loss_linear = []
        l2_loss_spline = []
        l2_loss_linear = []
        for (before, after), ytest in instance:
            Xtest = torch.concat((before, after),dim=1)

            # perform predictions
            #model1.eval()
            #with torch.no_grad():
            #    transf_pred = model1._forward_only(before, after)

            spline_pred = spline_model.predict(Xtest)
            LR_pred = LR_model.predict(Xtest)
        
            # calculate losses
            #loss_transf.append(np.mean((transf_pred.detach().numpy() - ytest.detach().numpy())**2))
            l1_loss_spline.append(np.abs((spline_pred - ytest.detach().numpy())))
            l1_loss_linear.append(np.abs((LR_pred - ytest.detach().numpy())))
            l2_loss_spline.append(np.mean((spline_pred - ytest.detach().numpy())**2))
            l2_loss_linear.append(np.mean((LR_pred - ytest.detach().numpy())**2))

        avg_l1_loss_transf1.append([*trainer.test(model1, dl_test)[0].values()][1])
        avg_l1_loss_transf2.append([*trainer.test(model2, dl_test)[0].values()][1])
        avg_l1_loss_transf3.append([*trainer.test(model3, dl_test)[0].values()][1])
        avg_l1_loss_transf4.append([*trainer.test(model4, dl_test)[0].values()][1])
        avg_l1_loss_transf5.append([*trainer.test(model5, dl_test)[0].values()][1])
        avg_l1_loss_spline.append(np.mean(l1_loss_spline))
        avg_l1_loss_linear.append(np.mean(l1_loss_linear))
    
        avg_l2_loss_transf1.append([*trainer.test(model1, dl_test)[0].values()][0])
        avg_l2_loss_transf2.append([*trainer.test(model2, dl_test)[0].values()][0])
        avg_l2_loss_transf3.append([*trainer.test(model3, dl_test)[0].values()][0])
        avg_l2_loss_transf4.append([*trainer.test(model4, dl_test)[0].values()][0])
        avg_l2_loss_transf5.append([*trainer.test(model5, dl_test)[0].values()][0])
        avg_l2_loss_spline.append(np.mean(l2_loss_spline))
        avg_l2_loss_linear.append(np.mean(l2_loss_linear))

    print("Average transformer1 loss:", avg_l1_loss_transf1)
    print("Average transformer2 loss:", avg_l1_loss_transf2)
    print("Average transformer3 loss:", avg_l1_loss_transf3)
    print("Average transformer4 loss:", avg_l1_loss_transf4)
    print("Average transformer5 loss:", avg_l1_loss_transf5)
    print("Average spline loss:", avg_l1_loss_spline)
    print("Average linear loss:", avg_l1_loss_linear)

    # save results
    d_l1 = {
        'Subjects': [12, 13], 
        'Average transformer1 loss': avg_l1_loss_transf1,
        'Average transformer2 loss': avg_l1_loss_transf2,
        'Average transformer3 loss': avg_l1_loss_transf3,
        'Average transformer4 loss': avg_l1_loss_transf4,
        'Average transformer5 loss': avg_l1_loss_transf5,
        'Average spline loss': avg_l1_loss_spline,
        'Average linear loss' : avg_l1_loss_linear}
    df_l1 = pd.DataFrame(data=d_l1).round(decimals=3)
    df_l1.to_csv("results/l1_test_results3.csv")

    d_l2 = {
        'Subjects': [12, 13], 
        'Average transformer1 loss': avg_l2_loss_transf1,
        'Average transformer2 loss': avg_l2_loss_transf2,
        'Average transformer3 loss': avg_l2_loss_transf3,
        'Average transformer4 loss': avg_l2_loss_transf4,
        'Average transformer5 loss': avg_l2_loss_transf5,
        'Average spline loss': avg_l2_loss_spline,
        'Average linear loss' : avg_l2_loss_linear}
    df_l2 = pd.DataFrame(data=d_l2).round(decimals=3)
    df_l2.to_csv("results/l2_test_results3.csv")
        



