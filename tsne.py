import ete3
from ete3 import Tree
import numpy as np
import argparse
import torch
import os
import pickle
from torch.utils.data import Dataset, DataLoader,TensorDataset
from dataset import *
from CNNpan import CNNpan
from CNNpan_DBT import CNNpan_DBT
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
import time

def tsne(cv,bs,modeldir,savedir,mode='tsne'):
    with open('cvdata.pkl','rb') as f:
        data=pickle.load(f)
    train_data,test_data=data[cv]
    train_loader=DataLoader(allele_dataset(train_data),batch_size=bs,shuffle=False)
    net=CNNpan_DBT()
    net.cuda()
    checkpoint=torch.load(modeldir)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch=checkpoint['epochs']
    print('model checkpoint {}'.format(epoch))
    trainfeature=[]
    trainlabel=[]
    for batch_idx,(sx,mx,y) in enumerate(train_loader):
        trainfeature.append(net.build(sx,mx,y))
    trainfeature=np.concatenate(trainfeature,0)
    start=time.time()
    if mode=='tsne':
        tsne = TSNE(n_components=2)
        tsnet=tsne.fit_transform(trainfeature)
    if mode=='pca':
        pca=PCA(n_components=2)
        tsnet=pca.fit_transform(trainfeature)
    if mode=='umap':
        um=umap.UMAP()
        tsnet=um.fit_transform(trainfeature)
    end=time.time()
    print('finished, spend time {}'.format(end-start))
    with open(os.path.join(savedir,modeldir.split('/')[-1].split('.')[0]+'_'+mode+'.pkl'), 'wb') as f:
        pickle.dump(tsnet,f)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-gpu','--gpu',default="0",type=str,help='#gpu')
    parser.add_argument('-savedir','--save_dir',default=None,type=str,help='saving dir')
    parser.add_argument('-mpath','--model_path',default=None,type=str,help='continuning training path')
    parser.add_argument('-mode','--mode',default='tsne',type=str,help='mode')
    parser.add_argument('-cv','--cv',type=int,help='cross validation index')
    parser.add_argument('-bs','--batch_size',default=1000,type=int,help='batch size in which a half is used to build tree and the other is used to query the model')
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    tsne(args.cv,args.batch_size,args.model_path,args.save_dir,args.mode)

