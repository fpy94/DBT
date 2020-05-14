import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import stats
from CNNpan import CNNpan
class CNNpan_DBT(CNNpan):
    def __init__(self):
        super(CNNpan_DBT,self).__init__()
        #one hot encoded data
        self.sigmoid=nn.Sigmoid()
        self.loss_fn=nn.MSELoss()
    def build(self,sx,mx,sy):
        #build tree based on support samples
        self.total_data_s=sx
        self.total_data_m=mx
        self.total_label=sy
        self.total_data_s.cuda()
        self.total_data_m.cuda()
        self.total_label.cuda()
        self.eval()
        feature=self.extract_feature(sx,mx)
        return feature.cpu().data.numpy()
    def compute_dist(self,children,q_s,q_m):
        q_s=q_s.unsqueeze(0)
        q_m=q_m.unsqueeze(0)
        #input: children are index, q are raw data
        indices=torch.LongTensor(children)
        children_data_s=torch.index_select(self.total_data_s, 0, indices)
        children_data_m=torch.index_select(self.total_data_m, 0, indices)
        #building represent
        children_rep=self.extract_feature(children_data_s,children_data_m)
        q_rep=self.extract_feature(q_s,q_m)
        #cal distance
        dist=torch.sum((children_rep-q_rep.repeat(children_rep.size(0),1))**2,1)
        return dist
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
                return total_prob.squeeze(1)                
            else:
                # to the final node
                sibling_=path.copy()
                sibling_+=[n.name for n in tree.search_nodes(name=children[minnode])[0].up.get_children()]
                sibling=[]
                for s in sibling_:
                    if s not in sibling:
                        sibling.append(s)
                distance=self.compute_dist(sibling,q_s,q_m)
                weight=F.softmax(-1*distance,0)
                indices=torch.LongTensor(sibling)
                sibling_label=torch.index_select(self.total_label, 0, indices).cuda()
                score=weight.unsqueeze(0).mm(sibling_label)
                score=torch.clamp(score,1e-8,1)
                total_prob=score
                return total_prob.squeeze(1)
    def train_loop_tree(self,tree,query_s,query_m,query_label,optimizer,update_step=1):
        #cal loss in query one by one
        batchsize=query_s.size(0)
        Loss=[]
        Pred=[]
        query_label=query_label.cuda()
        optimizer.zero_grad()
        for ii,i in enumerate(range(batchsize)):
            p=self.query(query_s[i],query_m[i],tree)
            Pred.append(p.cpu().data.numpy())
            loss=self.loss_fn(p,query_label[i])/batchsize
            Loss.append(loss.item())
            loss.backward()
        optimizer.step()
        Pred=np.concatenate(Pred)
        return np.sum(Loss),Pred
    def inference(self,raw_test,tree):
        predict2=[]
        for i in range(raw_test.size(0)):
            p=self.query(raw_test[i],tree)
            predict2.append(p.cpu().data.numpy())
        predict2=np.concatenate(predict2)
        return predict2
