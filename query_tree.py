import ete3
from ete3 import Tree
import numpy as np
import argparse
import torch
import os
import pickle
from torch.utils.data import Dataset, DataLoader,TensorDataset
from dataset import *
from ACME import ACME
from ACME_DBT import ACME_DBT
def query_tree(cv,bs,modeldir,savedir,datadir,eps,initroot,rep,treedir,USE_CUDA):
    with open(datadir,'rb') as f:
        data=pickle.load(f)
    train_data,test_data=data[cv]
    train_loader=DataLoader(allele_dataset(train_data),batch_size=bs,shuffle=False)
    test_loader=DataLoader(allele_dataset(test_data),batch_size=10000,shuffle=False)
    net=ACME_DBT(USE_CUDA)
    if USE_CUDA:
        net.cuda()
        checkpoint=torch.load(modeldir)
    else:
        checkpoint=torch.load(modeldir,map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch=checkpoint['epochs']
    '''
    if epoch%2!=0:
        epoch-=1
    '''
    print('model checkpoint {}'.format(epoch))
    SX=[]
    MX=[]
    Y=[]
    for batch_idx,(sx,mx,y) in enumerate(train_loader):
        SX.append(sx)
        MX.append(mx)
        Y.append(y)
    SX=torch.cat(SX,0)
    MX=torch.cat(MX,0)
    Y=torch.cat(Y,0)
    a=net.build(SX,MX,Y)
    if rep is not None:
        with open(os.path.join(treedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch)+'_eps_'+str(eps)+'_rep_'+str(rep)+'.pkl'), 'rb') as f:
            treedir=pickle.load(f)
        tree=treedir['tree']
        nodelist=treedir['nodelist']
        edge=treedir['edge']
        path=treedir['path']
            
    else:
        with open(os.path.join(treedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch)+'_eps_'+str(eps)+'.pkl'), 'rb') as f:
             treedir=pickle.load(f)
        tree=treedir['tree']
        nodelist=treedir['nodelist']
        edge=treedir['edge']
        path=treedir['path']
    Pred=[]
    tPath=[]
    for batch_idx,(sx,mx,y) in enumerate(test_loader):
        tpath,pred=net.query(tree,sx,mx,edge,path)
        tPath+=tpath
        Pred.append(pred)
    Pred=np.concatenate(Pred,0)
    if rep is not None:
        with open(os.path.join(savedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch)+'_eps_'+str(eps)+'_rep_'+str(rep)+'_pred.pkl'), 'wb') as f:
            pickle.dump({'pred':Pred,'path':tPath},f)
    else:
        with open(os.path.join(savedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch)+'_eps_'+str(eps)+'_pred.pkl'), 'wb') as f:
            pickle.dump({'pred':Pred,'path':tPath},f)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-gpu','--gpu',default=None,type=str,help='#gpu')
    parser.add_argument('-savedir','--save_dir',default=None,type=str,help='saving dir')
    parser.add_argument('-datadir','--data_dir',default=None,type=str,help='data dir')
    parser.add_argument('-mpath','--model_path',default=None,type=str,help='continuning training path')
    parser.add_argument('-initroot','--init_root',default=False,type=bool,help='whether init root')
    parser.add_argument('-cv','--cv',type=int,help='cross validation index')
    parser.add_argument('-bs','--batch_size',default=1000,type=int,help='batch size in which a half is used to build tree and the other is used to query the model')
    parser.add_argument('-eps','--eps',default=0.1,type=float,help='tree label distance')
    parser.add_argument('-rep','--rep',default=None,type=int,help='cross validation index')
    parser.add_argument('-tpath','--tree_path',default=None,type=str,help='continuning training path')
    args=parser.parse_args()
    if args.gpu!=None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        USE_CUDA=True
    else:
        USE_CUDA=False
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    query_tree(args.cv,args.batch_size,args.model_path,args.save_dir,args.data_dir,args.eps,args.init_root,args.rep,args.tree_path,USE_CUDA)


