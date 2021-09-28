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
def BTQuery(q,tree,trainfeature,k=128):
    currentname=tree.get_tree_root().name
    path=[currentname]
    while True:
        children=[n.name for n in tree.search_nodes(name=currentname)[0].get_children()]
        if len(children)<k:
            children.append(tree.search_nodes(name=currentname)[0].name)
        tchildren=trainfeature[children]
        d=np.sum((np.expand_dims(trainfeature[q],0)-tchildren)**2,1)
        minnode=children[int(np.argmin(d))]
        if minnode==currentname:
            break
        currentname=minnode
        path.append(minnode)
    return currentname,path
def BTtrain(trainfeature,label,initroot,eps=0.1):
    meann=np.mean(trainfeature,0,keepdims=True)
    middle=np.argsort(np.sum((trainfeature-meann)**2,1))[0]
    Nnodes=trainfeature.shape[0]
    Nodelist_=np.random.permutation(Nnodes)
    if initroot:
        Nodelist=[middle]
        for n in Nodelist_:
            if n not in Nodelist:
                Nodelist.append(n)
    else:
        Nodelist=Nodelist_
    tree=Tree(name=Nodelist[0])
    querynum=1
    nodenum=1
    Path=[[Nodelist[0]]]
    edge=[]
    for nodeidx in Nodelist[1:]:
        vmin,p=BTQuery(nodeidx,tree,trainfeature)
        if np.abs(label[vmin]-label[nodeidx])>eps:
#             tree.search_nodes(name=vmin)[0].add_child(name=nodeidx,dist=np.abs(label[vmin]-label[nodeidx]))
            tree.search_nodes(name=vmin)[0].add_child(name=nodeidx)
            nodenum+=1
            Path.append(p+[nodeidx])
            edge.append([vmin,nodeidx])
        querynum+=1
        if querynum%2500==0:
            print('querynum {}, treenum {}'.format(querynum,nodenum))
    print('tree building finished, treenum {}'.format(nodenum))
    return tree,Nodelist,Path,edge
def build_tree(cv,bs,modeldir,savedir,datadir,eps,initroot,rep):
    with open(datadir,'rb') as f:
        data=pickle.load(f)
    train_data,test_data=data[cv]
    train_loader=DataLoader(allele_dataset(train_data),batch_size=bs,shuffle=False)
    net=ACME()
    checkpoint=torch.load(modeldir,map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch=checkpoint['epochs']
    print('model checkpoint {}'.format(epoch))
    trainfeature=[]
    trainlabel=[]
    for batch_idx,(sx,mx,y) in enumerate(train_loader):
        trainfeature.append(net.extract_feature(sx,mx,False).data.numpy())
        trainlabel.append(y.data.numpy())
    trainfeature=np.concatenate(trainfeature,0)
    trainlabel=np.concatenate(trainlabel,0)
    print(trainfeature.shape,trainlabel.shape)
    tree,nodelist,path,edge=BTtrain(trainfeature,trainlabel,initroot,eps)
    if rep is not None:
        with open(os.path.join(savedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch)+'_eps_'+str(eps)+'_rep_'+str(rep)+'.pkl'), 'wb') as f:
            pickle.dump({'tree':tree,'nodelist':nodelist,'path':path,'edge':edge},f)
    else:
        with open(os.path.join(savedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch)+'_eps_'+str(eps)+'.pkl'), 'wb') as f:
            pickle.dump({'tree':tree,'nodelist':nodelist,'path':path,'edge':edge},f)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-gpu','--gpu',default="3",type=str,help='#gpu')
    parser.add_argument('-savedir','--save_dir',default=None,type=str,help='saving dir')
    parser.add_argument('-datadir','--data_dir',default=None,type=str,help='data dir')
    parser.add_argument('-mpath','--model_path',default=None,type=str,help='continuning training path')
    parser.add_argument('-initroot','--init_root',default=False,type=bool,help='whether init root')
    parser.add_argument('-cv','--cv',type=int,help='cross validation index')
    parser.add_argument('-bs','--batch_size',default=1000,type=int,help='batch size in which a half is used to build tree and the other is used to query the model')
    parser.add_argument('-eps','--eps',default=0.1,type=float,help='tree label distance')
    parser.add_argument('-rep','--rep',default=None,type=int,help='tree index')
    args=parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    build_tree(args.cv,args.batch_size,args.model_path,args.save_dir,args.data_dir,args.eps,args.init_root,args.rep)


