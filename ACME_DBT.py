import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import stats
from ACME import ACME
class ACME_DBT(ACME):
    def __init__(self,USE_CUDA=True):
        super(ACME_DBT,self).__init__()
        #one hot encoded data
        self.sigmoid=nn.Sigmoid()
        self.loss_fn=nn.MSELoss()
        self.matrix=None
        self.nodearray=None
        self.nodearraylen=None
        self.cut=None
        self.USE_CUDA=USE_CUDA

    def reset(self):
        self.matrix=None
        self.nodearray=None
        self.nodearraylen=None
        self.cut=None


    def build(self,sx,mx,sy):
        #build tree based on support samples
        self.total_data_s=sx
        self.total_data_m=mx
        self.total_label=sy
        self.eval()
        if self.USE_CUDA:
            self.total_data_s.cuda()
            self.total_data_m.cuda()
            self.total_label.cuda()
            self.trainfeature=self.extract_feature(sx,mx,self.USE_CUDA)
            return self.trainfeature.cpu().data.numpy()
        else:
            self.trainfeature=self.extract_feature(sx,mx,False)
            return self.trainfeature.data.numpy()

    def compute_dist(self,children,q_s,q_m):
        q_s=q_s.unsqueeze(0)
        q_m=q_m.unsqueeze(0)
        #input: children are index, q are raw data
        indices=torch.LongTensor(children)
        children_data_s=torch.index_select(self.total_data_s, 0, indices)
        children_data_m=torch.index_select(self.total_data_m, 0, indices)
        #building represent
        children_rep=self.extract_feature(children_data_s,children_data_m,self.USE_CUDA)
        q_rep=self.extract_feature(q_s,q_m,self.USE_CUDA)
        #cal distance
        dist=torch.sum((children_rep-q_rep.repeat(children_rep.size(0),1))**2,1)
        return dist
    def get_tree_edge_path(self,tree):
        edge=[]
        path=[]
        num=0
        for node in tree.traverse("preorder"):
            num+=1
            nodename=node.name
            e=[(nodename,n.name) for n in node.get_children()]
            edge.extend(e)
            path.append(tuple([n.name for n in tree.search_nodes(name=nodename)[0].iter_ancestors()][::-1]+[nodename]))
            #if num%5000==0:
                #print(num)
        return edge,path
    def distance(self,test,train):
        if self.USE_CUDA:
            test=test.cuda()
            train=train.cuda()
        d1=-2*torch.mm(test,train.permute(1,0))
        d2=torch.sum(test**2,dim=1,keepdim=True)
        d3=torch.sum(train**2,dim=1)
        dist=d1+d2+d3
        dist=torch.clamp(dist,min=0.0)
        dist=torch.sqrt(dist)
        return dist
    def get_tree_matrix(self,tree,edge=None,path=None):
        if edge is None and path is None:
            edge,path=self.get_tree_edge_path(tree)
        else:
            edge=sorted(edge, key=lambda e:e[0])
        nodearray=[]
        pre=None
        nowindx=0
        cut=[0]
        childset={}
        for e in edge:
            if pre==None:
                nodearray.append(e[0])
                nodearray.append(e[1])
                childset[e[0]]=[(e[0],nowindx),(e[1],nowindx+1)]
                nowindx+=2
                pre=e[0]        
            elif e[0]==pre:
                nodearray.append(e[1])
                childset[pre].append((e[1],nowindx))
                nowindx+=1
            else:
                cut.append(nowindx)
                nodearray.append(e[0])
                nodearray.append(e[1])
                childset[e[0]]=[(e[0],nowindx),(e[1],nowindx+1)]
                nowindx+=2
                pre=e[0]
        cut.append(len(nodearray))
        nodearraylen=len(nodearray)
        maxdepth=np.max([len(p) for p in path])
        nodedepth={p[-1]:len(p) for p in path}
        scorelist=[0.5**i for i in range(maxdepth)]
        matrix=np.zeros((nodearraylen,len(path)))
        for i,p in enumerate(path):
            if len(p)==1:
                matrix[childset[p[0]][0][1],i]=scorelist[nodedepth[p[0]]-1]
            else:
                for j in range(len(p)-1):
                    candidate=childset[p[j]]
                    index=None
                    for c in candidate:
                        if c[0]==p[j+1]:
                            index=c[1]
                            break
                    if index!=None:
                        matrix[index,i]=scorelist[nodedepth[p[j+1]]-1]
                    else:
                        print('error')
                if p[-1] in childset:
                    matrix[childset[p[-1]][0][1],i]=(scorelist[nodedepth[p[-1]]-1]+np.sum(scorelist[nodedepth[p[-1]]:]))/2
        self.matrix=matrix
        self.nodearray=nodearray
        self.nodearraylen=nodearraylen
        self.cut=cut
        self.childset=childset
        return edge,path
    def get_path(self,q_s,q_m,tree,edge=None,path=None):
        if self.matrix is None or self.nodearraylen is None or self.nodearray is None or self.cut is None:
            edge,path=self.get_tree_matrix(tree,edge,path)
        self.testfeature=self.extract_feature(q_s,q_m,self.USE_CUDA)
        trainfeaturere=torch.zeros((self.nodearraylen,self.trainfeature.size(1)))
        for i,a in enumerate(self.nodearray):
            trainfeaturere[i]=self.trainfeature[a]
        testtrain=self.distance(self.testfeature,trainfeaturere).cpu().data.numpy()
        testtrainmask=np.zeros((testtrain.shape))
        for i in range(len(self.cut)-1):
            start=self.cut[i]
            end=self.cut[i+1]
            mask=np.where(testtrain[:,start:end]==np.min(testtrain[:,start:end],1,keepdims=True))
            mask=(mask[0],mask[1]+start)
            testtrainmask[mask]=1
        out=np.matmul(testtrainmask,self.matrix)
        pathmax=np.argmax(out,1)
        outpath=[]
        for p in pathmax:
              outpath.append(path[p])
        sibling=[]
        for p in outpath:
            if len(p)>1:
                sib=self.childset[p[-2]]
                sib,_=zip(*sib)
                a=list(p)+list(sib)
            else:
                sib=self.childset[p[0]]
                sib,_=zip(*sib)
                a=list(sib)
            aa=[]
            for aaa in a:
                if aaa not in aa:
                    aa.append(aaa)
            sibling.append(aa)
        return outpath,sibling
    def train_loop_tree(self,tree,query_s,query_m,query_label,optimizer,edge=None,path=None,update_step=1):
        #cal loss in query one by one
        batchsize=query_s.size(0)
        Loss=[]
        Pred=[]
        if self.USE_CUDA:
            query_label=query_label.cuda()
        optimizer.zero_grad()
        testpath,sibling=self.get_path(query_s,query_m,tree,edge,path)
        self.reset()
        for i in range(query_s.size(0)):
            distance=self.compute_dist(sibling[i],query_s[i],query_m[i])
            weight=F.softmax(-1*distance,0)
            indices=torch.LongTensor(sibling[i])
            if self.USE_CUDA:
                sibling_label=torch.index_select(self.total_label, 0, indices).cuda()
            else:
                sibling_label=torch.index_select(self.total_label, 0, indices)
            score=weight.unsqueeze(0).mm(sibling_label)
            score=torch.clamp(score,1e-8,1)
            p=score.squeeze(1)                        
            if self.USE_CUDA:
                Pred.append(p.cpu().data.numpy())
            else:
                Pred.append(p.data.numpy())
            loss=self.loss_fn(p,query_label[i])/batchsize
            Loss.append(loss.item())
            loss.backward()
        optimizer.step()
        Pred=np.concatenate(Pred)
        return np.sum(Loss),Pred
    def query(self,tree,query_s,query_m,edge=None,path=None):
        Pred=[]
        testpath,sibling=self.get_path(query_s,query_m,tree,edge,path)
        for i in range(query_s.size(0)):
            distance=self.compute_dist(sibling[i],query_s[i],query_m[i])
            weight=F.softmax(-1*distance,0)
            indices=torch.LongTensor(sibling[i])
            if self.USE_CUDA:
                sibling_label=torch.index_select(self.total_label, 0, indices).cuda()
            else:
                sibling_label=torch.index_select(self.total_label, 0, indices)

            score=weight.unsqueeze(0).mm(sibling_label)
            score=torch.clamp(score,1e-8,1)
            p=score.squeeze(1)                        
            if self.USE_CUDA:
                Pred.append(p.cpu().data.numpy())
            else:
                Pred.append(p.data.numpy())

        Pred=np.concatenate(Pred,0)
        return testpath,Pred

