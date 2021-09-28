import numpy as np
import torch
from sklearn import metrics
import pickle
import random
from scipy import stats
import os
from torch.autograd import Variable
from dataset import *
from sys import argv
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset
from early_stopping import EarlyStopping
from ACME import ACME

with open('./data_process/cvdata.pkl','rb') as f:
    data=pickle.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MAX_EPOCHS=20
bs=64
lr=1e-3
savepath='./pretrain_model'
if not os.path.exists(savepath):
    os.makedirs(savepath)
models=[]
for cv in range(5):
    train_data,test_data=data[cv]
    random.shuffle(train_data)
    valid_data=train_data[:len(train_data)//10]
    traindata=train_data[len(train_data)//10:]
    train_loader=DataLoader(allele_dataset(traindata),batch_size=bs,shuffle=True)
    valid_loader=DataLoader(allele_dataset(valid_data),batch_size=bs,shuffle=True)
    test_loader=DataLoader(allele_dataset(test_data),batch_size=bs,shuffle=False)
    net=ACME()
    net=net.cuda()
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    for epochs in range(MAX_EPOCHS):
        net.train()
        if epochs==0:
            while True:
                net.train_loop(epochs,train_loader,optimizer)
                tests,testy=net.Test(valid_loader)
                pr,_=stats.pearsonr(tests[:,0],testy[:,0])
                print(pr)
                if pr>0.75:
                    models.append(net.state_dict())
                    break
                else:
                    net=ACME()
                    net.cuda()
        else:
            break

with open('./pretrain_model/pretrain.pkl','wb') as f:
    pickle.dump(models,f)
