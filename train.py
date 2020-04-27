import numpy as np
import torch
from sklearn import metrics
import pickle
import random
from scipy import stats
import os
from torch.autograd import Variable
from clust import MHCKNN
from dataset import *
from sys import argv
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset
from BT import *
from CNNpan import CNNpan
from CNNpan_DBT import CNNpan_DBT
from ete3 import Tree
import argparse


def train(lr,bs,eps,FOREST_NUM,MAX_EPOCHS):
    # loading data and build train_loader
    '''
    datafile='../../data/iedb2013/data_pan_nonoverlap.pkl'
    with open(datafile,'rb') as f:
        data=pickle.load(f)['data']
    allele_list,allele_near=MHCKNN(K=1)
    datanum={}
    with open('../../data/iedb2013/mhcI_nonoverlap_allelenum_sort.tsv') as f:
        for l in f:
            ll=l.strip().split()
            datanum[ll[0]]=int(ll[1])
    allele_test=[]
    for a in datanum:
        if datanum[a]>200:
            allele_test.append(a)
    print(allele_test)
    total_data=[]
    for d in allele_test:
        total_data+=data[d]
    kf = KFold(n_splits=5,shuffle=True)
    train_index,test_index=zip(*kf.split(total_data))
    train_data=[total_data[i] for i in train_index[0]]
    test_data=[total_data[i] for i in test_index[0]]
    '''
    with open('./data.pkl','rb') as f:
        train_data=pickle.load(f)['data']


    train_loader=DataLoader(allele_dataset(train_data),batch_size=bs,shuffle=True)
    #test_loader=DataLoader(allele_dataset(test_data),batch_size=bs,shuffle=False)

    net=CNNpan_DBT()
    net.cuda()

    # pretrain the model using regular DNN
    train_loader=DataLoader(allele_dataset(train_data),batch_size=64,shuffle=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    for epochs in range(3):
        net.train_loop(epochs,train_loader,optimizer)

    #now train the tree	
    train_loader=DataLoader(allele_dataset(train_data),batch_size=bs,shuffle=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.6)
    for epochs in range(MAX_EPOCHS):
        Loss=[]
        Pred=[]
        Label=[]
        for batch_idx,(sx,mx,y) in enumerate(train_loader):
            #sx peptide sequence
            #mx mhc sequence
            curbs=sx.size(0)
            sx_s=sx[:curbs//2]
            mx_s=mx[:curbs//2]
            y_s=y[:curbs//2]
            sx_q=sx[curbs//2:]
            mx_q=mx[curbs//2:]
            y_q=y[curbs//2:]
            trainfeature=net.build(sx_s,mx_s,y_s)
            dist=distance(trainfeature,trainfeature)
    #         print('building tree')
            for _ in range(FOREST_NUM):
            #rebuild tree and train
                tree=BTtrain(dist,y_s.data.numpy(),eps)    
                loss,pred=net.train_loop_tree(tree,sx_q,mx_q,y_q,optimizer)
                pr,_=stats.pearsonr(pred,y_q.data.numpy()[:,0])
                print(loss,pr)
            Label.append(y_q.data.numpy())
            Loss.append(loss)
            Pred.append(pred)
        scheduler.step()
        Loss=np.mean(Loss)
        Pred=np.concatenate(Pred,0)
        Label=np.concatenate(Label,0)
        pr,_=stats.pearsonr(Pred,Label[:,0])
        print('Epoch {}, loss {}, test pr {}'.format(epochs,Loss,pr))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-lr','--learning_rate',default=0.001,type=float,help='learning rate')
    parser.add_argument('-bs','--batch_size',default=1000,type=int,help='batch size in which a half is used to build tree and the other is used to query the model')
    parser.add_argument('-nfor','--forest_num',default=1,type=int,help='the num of trees that are build in each mini-batch')
    parser.add_argument('-maxepoch','--max_epochs',default=50,type=int,help='max_epochs')
    parser.add_argument('-eps','--epsilon',default=0.1,type=float,help='the absolute distance between query sample and its nearest node when building tree')
    args=parser.parse_args()
    train(args.learning_rate,args.batch_size,args.epsilon,args.forest_num,args.max_epochs)
