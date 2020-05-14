import ete3
from ete3 import Tree
import numpy as np
import argparse
import torch
import os
import pickle
from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch.nn.functional as F
import torch.nn as nn
from dataset import *
from CNNpan import CNNpan
from CNNpan_DBT import CNNpan_DBT
def BTQuery(q,tree,trainfeature,k=10):
    currentname=tree.get_tree_root().name
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
    return currentname
def BTtrain(trainfeature,label,eps=0.1):
    Nnodes=trainfeature.shape[0]
    Nodelist=np.random.permutation(Nnodes)
    tree=Tree(name=Nodelist[0])
    querynum=1
    nodenum=1
    for nodeidx in Nodelist[1:]:
        vmin=BTQuery(nodeidx,tree,trainfeature)
        if np.abs(label[vmin]-label[nodeidx])>eps:
#             tree.search_nodes(name=vmin)[0].add_child(name=nodeidx,dist=np.abs(label[vmin]-label[nodeidx]))
            tree.search_nodes(name=vmin)[0].add_child(name=nodeidx)
            nodenum+=1
        querynum+=1
        if querynum%2500==0:
            print('querynum {}, treenum {}'.format(querynum,nodenum))
    print('tree building finished, treenum {}'.format(nodenum))
    return tree,Nodelist

class CNNpan_DBT_q(CNNpan_DBT):
    def __init__(self):
        super(CNNpan_DBT,self).__init__()
    def query(self,q_s,q_m,tree):
        rootname=int(tree.get_tree_root().name)
        currentname=int(tree.get_tree_root().name)
        path=[rootname]
        while True:
            #get children
            children=[currentname]
            children+=[n.name for n in tree.search_nodes(name=currentname)[0].get_children()]
            distance=self.compute_dist(children,q_s,q_m)
            minnode=torch.argmin(distance).item()
            if minnode!=0 and len(tree.search_nodes(name=children[minnode])[0].get_children())!=0:
                #not to the final node
                currentname=children[minnode]
                path.append(children[minnode])
            elif children[minnode]==rootname:
                #if is root
                sibling=[rootname]
                sibling+=[n.name for n in tree.search_nodes(name=rootname)[0].get_children()]
                distance=self.compute_dist(sibling,q_s,q_m)
                weight=F.softmax(-1*distance,0)
                indices=torch.LongTensor(sibling)
                sibling_label=torch.index_select(self.total_label, 0, indices).cuda()
                score=weight.unsqueeze(0).mm(sibling_label)
                total_prob=score
                return total_prob.squeeze(1),[rootname]                
            else:
                # to the final node
                sibling=path.copy()
                sibling+=[n.name for n in tree.search_nodes(name=children[minnode])[0].up.get_children()]
                distance=self.compute_dist(sibling,q_s,q_m)
                weight=F.softmax(-1*distance,0)
                indices=torch.LongTensor(sibling)
                sibling_label=torch.index_select(self.total_label, 0, indices).cuda()
                score=weight.unsqueeze(0).mm(sibling_label)
                score=torch.clamp(score,1e-8,1)
                total_prob=score
                if path[-1]==children[minnode]:
                    return total_prob.squeeze(1),path
                else:
                    return total_prob.squeeze(1),path+[children[minnode]]


def build_tree(cv,bs,modeldir,treedir,savedir):
    with open('cvdata.pkl','rb') as f:
        data=pickle.load(f)
    train_data,test_data=data[cv]
    train_loader=DataLoader(allele_dataset(train_data),batch_size=bs,shuffle=False)
    test_loader=DataLoader(allele_dataset(test_data),batch_size=1,shuffle=False)
    net=CNNpan_DBT_q()
    net.cuda()
    checkpoint=torch.load(modeldir)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch=checkpoint['epochs']+1
    print('model checkpoint {}'.format(epoch))
    train_sx=[]
    train_mx=[]
    train_y=[]
    for batch_idx,(sx,mx,y) in enumerate(train_loader):
        train_sx.append(sx)
        train_mx.append(mx)
        train_y.append(y)
    train_sx=torch.cat(train_sx,0)
    train_mx=torch.cat(train_mx,0)
    train_y=torch.cat(train_y,0)
    net.total_data_s=train_sx
    net.total_data_m=train_mx
    net.total_label=train_y
    with open(os.path.join(treedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch)+'.pkl'), 'rb') as f:
        tree=pickle.load(f)['tree']
    testnum=len(test_data)
    print('test num {}'.format(testnum))
    Pred=np.zeros((testnum,2))
    Path=[[]for _ in range(testnum)]
    for batch_idx,(sx,mx,y) in enumerate(test_loader):
        p,path=net.query(sx[0],mx[0],tree)
        Pred[batch_idx,0]=p.item()
        Pred[batch_idx,1]=y.item()
        Path[batch_idx]=path
    
    with open(os.path.join(savedir,'tree_cv_'+str(cv)+'_epoch_'+str(epoch)+'_pred.pkl'), 'wb') as f:
        pickle.dump({'pred':Pred,'path':Path},f)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-gpu','--gpu',default="3",type=str,help='#gpu')
    parser.add_argument('-savedir','--save_dir',default=None,type=str,help='saving dir')
    parser.add_argument('-mpath','--model_path',default=None,type=str,help='continuning training path')
    parser.add_argument('-tpath','--tree_path',default=None,type=str,help='continuning training path')
    parser.add_argument('-cv','--cv',type=int,help='cross validation index')
    parser.add_argument('-bs','--batch_size',default=1000,type=int,help='batch size in which a half is used to build tree and the other is used to query the model')
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    build_tree(args.cv,args.batch_size,args.model_path,args.tree_path,args.save_dir)


