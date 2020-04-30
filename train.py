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


def train(lr,bs,eps,cv,modeldir,savedir,FOREST_NUM,MAX_EPOCHS,savecheck=10):
    # loading data and build train_loader
    with open('cvdata.pkl','rb') as f:
        data=pickle.load(f)
    train_data,test_data=data[cv]
    train_loader=DataLoader(allele_dataset(train_data),batch_size=bs,shuffle=True)
    #test_loader=DataLoader(allele_dataset(test_data),batch_size=bs,shuffle=False)

    net=CNNpan_DBT()
    net.cuda()
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    if modeldir:
        checkpoint=torch.load(modeldir)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        nowep=checkpoint['epochs']
        print('Continue training from epoch {}, learning rate {}'.format(nowep,optimizer.state_dict()['param_groups'][0]['lr']))
    else:
        # loading pretrain model
        net.load_state_dict(torch.load('./pretrain_model/cv_'+str(cv)+'.pkl'))
        nowep=0
    #now train the tree	
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.6)
    for epochs in range(nowep,MAX_EPOCHS+nowep):
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
                #print(loss,pr)
            Label.append(y_q.data.numpy())
            Loss.append(loss)
            Pred.append(pred)
            if (batch_idx+1)%20==0:
                print('Batch {}, loss {}, pr {}'.format((batch_idx+1),loss,pr))
        scheduler.step()
        Loss=np.mean(Loss)
        Pred=np.concatenate(Pred,0)
        Label=np.concatenate(Label,0)
        pr,_=stats.pearsonr(Pred,Label[:,0])
        print('Epoch {}, loss {}, pr {}'.format((epochs+1),Loss,pr))

        if (epochs+1)%savecheck==0:
            torch.save({'epochs':epochs,
                        'model_state_dict':net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},
                        os.path.join(savedir,'checkpoint_cv_'+str(cv)+'_epoch_'+str(epochs)+'.pkl')
                      )


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-gpu','--gpu',default=3,type=str,help='#gpu')
    parser.add_argument('-savedir','--save_dir',default=None,type=str,help='saving dir')
    parser.add_argument('-mpath','--model_path',default=None,type=str,help='continuning training path')
    parser.add_argument('-lr','--learning_rate',default=0.001,type=float,help='learning rate')
    parser.add_argument('-cv','--cv',type=int,help='cross validation index')
    parser.add_argument('-bs','--batch_size',default=1000,type=int,help='batch size in which a half is used to build tree and the other is used to query the model')
    parser.add_argument('-nfor','--forest_num',default=1,type=int,help='the num of trees that are build in each mini-batch')
    parser.add_argument('-maxepoch','--max_epochs',default=50,type=int,help='max_epochs')
    parser.add_argument('-eps','--epsilon',default=0.1,type=float,help='the absolute distance between query sample and its nearest node when building tree')
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train(args.learning_rate,args.batch_size,args.epsilon,args.cv,args.model_path,args.save_dir,args.forest_num,args.max_epochs)

