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
def BTQuery(q,tree,trainfeature,k=128):
    currentname=tree.get_tree_root().name
    path=[currentname]
    while True:
        children=tree.search_nodes(name=currentname)[0].get_children()
        if len(children)<k:
            children.append(tree.search_nodes(name=currentname)[0])
        dmin=np.inf
        minnode=None
        for c in children:
            name=c.name
            d=np.sum((trainfeature[q]-trainfeature[name])**2)
            if d<dmin:
                dmin=d
                minnode=c.name
        if minnode==currentname:
            break
        currentname=minnode
        path.append(minnode)
    return currentname,path
def BTtrain(trainfeature,label,eps=0.1):
    meann=np.mean(trainfeature,0,keepdims=True)
    middle=np.argsort(np.sum((trainfeature-meann)**2,1))[0]
    Nnodes=trainfeature.shape[0]
    Nodelist_=np.random.permutation(Nnodes)
    Nodelist=[middle]
    for n in Nodelist_:
        if n not in Nodelist:
            Nodelist.append(n)
    tree=Tree(name=Nodelist[0])
    querynum=1
    nodenum=1
    Path=[[] for _ in range(Nnodes)]
    for nodeidx in Nodelist[1:]:
        vmin,p=BTQuery(nodeidx,tree,trainfeature)
        if np.abs(label[vmin]-label[nodeidx])>eps:
#             tree.search_nodes(name=vmin)[0].add_child(name=nodeidx,dist=np.abs(label[vmin]-label[nodeidx]))
            tree.search_nodes(name=vmin)[0].add_child(name=nodeidx)
            nodenum+=1
            Path[nodeidx]=p
        querynum+=1
        if querynum%2500==0:
            print('querynum {}, treenum {}'.format(querynum,nodenum))
    print('tree building finished, treenum {}'.format(nodenum))
    return tree,Nodelist,Path

def build_tree(cv,bs,modeldir,savedir):
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
        trainlabel.append(y)
    trainfeature=np.concatenate(trainfeature,0)
    trainlabel=np.concatenate(trainlabel,0)
    print(trainfeature.shape,trainlabel.shape)
    tree,nodelist,path=BTtrain(trainfeature,trainlabel,0.1)
    with open(os.path.join(savedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch+1)+'.pkl'), 'wb') as f:
        pickle.dump({'tree':tree,'nodelist':nodelist,'path':path},f)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-gpu','--gpu',default="3",type=str,help='#gpu')
    parser.add_argument('-savedir','--save_dir',default=None,type=str,help='saving dir')
    parser.add_argument('-mpath','--model_path',default=None,type=str,help='continuning training path')
    parser.add_argument('-cv','--cv',type=int,help='cross validation index')
    parser.add_argument('-bs','--batch_size',default=1000,type=int,help='batch size in which a half is used to build tree and the other is used to query the model')
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    build_tree(args.cv,args.batch_size,args.model_path,args.save_dir)


