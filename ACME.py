import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import stats

class ACME(nn.Module):
    #reimplementation of ACME without attention module
    def __init__(self):
        super(ACME,self).__init__()

        self.pconv1=nn.Sequential(
            nn.Conv1d(20,128,3,padding=1),
            nn.ReLU()
        )
        self.mconv1=nn.Sequential(
            nn.Conv1d(20,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.mconv2=nn.Sequential(
            nn.Conv1d(128,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )


        self.dense0=nn.Sequential(
            nn.Linear(3752,256),
            nn.ReLU()
        )
        self.dense1=nn.Sequential(
            nn.Linear(5248,256),
            nn.ReLU()
        )
        self.dense2=nn.Sequential(
            nn.Linear(4096,256),
            nn.ReLU()
        )
        self.dense3=nn.Sequential(
            nn.Linear(768,64)
        )
        self.dense4=nn.Sequential(
            nn.Linear(64,1)
        )


        self.sigmoid=nn.Sigmoid()

        self.loss_fn=nn.MSELoss()

    def extract_feature(self,sx,mx,USE_CUDA=True):
        if USE_CUDA:
            sx=Variable(sx.cuda())
            mx=Variable(mx.cuda())
        else:
            sx=Variable(sx)
            mx=Variable(mx)
        sx=sx.permute(0,2,1)
        mx=mx.permute(0,2,1)
        pep_conv=self.pconv1(sx)#batch,128,24
        mhc_conv_1=self.mconv1(mx)#batch,128,17
        mhc_conv_2=self.mconv2(mhc_conv_1)#batch,128,8
        flat_pep_0=pep_conv.view(pep_conv.size(0),-1)#3072
        flat_pep_1=pep_conv.view(pep_conv.size(0),-1)#3072
        flat_pep_2=pep_conv.view(pep_conv.size(0),-1)#3072
        flat_mhc_0=mx.contiguous().view(mx.size(0),-1)#680
        flat_mhc_1=mhc_conv_1.view(mhc_conv_1.size(0),-1)#2176
        flat_mhc_2=mhc_conv_2.view(mhc_conv_2.size(0),-1)#1024
        cat_0=torch.cat((flat_pep_0, flat_mhc_0),1)#3752
        cat_1=torch.cat((flat_pep_1, flat_mhc_1),1)#5248
        cat_2=torch.cat((flat_pep_2, flat_mhc_2),1)#4096
        fc1_0=self.dense0(cat_0)#256
        fc1_1=self.dense1(cat_1)
        fc1_2=self.dense2(cat_2)
        merge_1=torch.cat((fc1_0, fc1_1,fc1_2),1)#768
        self.feature=self.dense3(merge_1)#64
        return self.feature

    def forward(self,sx,mx):
        self.feature=self.extract_feature(sx,mx)
        self.represent=self.dense4(self.feature)
        out=self.sigmoid(self.represent)
        return out

    def train_loop(self,epoch,train_loader,optimizer):
        avg_loss=0
        avg_l2loss=0
        SCORE=[]
        LABEL=[]

        for batch_idx,(sx,mx,y) in enumerate(train_loader):
            scores=self.forward(sx,mx)
            y=Variable(y.cuda())
            loss=self.loss_fn(scores,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss+=loss.item()
            SCORE.append(scores.cpu().data.numpy())
            LABEL.append(y.cpu().data.numpy())
        S=np.concatenate(SCORE,0)
        L=np.concatenate(LABEL,0)
        pr,_=stats.pearsonr(S[:,0],L[:,0])
        avg_loss/=batch_idx
        print('Epoch {:d} | Loss {:f}| pr {}'.format(epoch,avg_loss,pr))


    def test_loop(self,epochs,test_loader):
        avg_loss=0
        avg_l2loss=0
        SCORE=[]
        LABEL=[]

        for batch_idx,(sx,mx,y) in enumerate(test_loader):
            scores=self.forward(sx,mx)
            y=Variable(y.cuda())
            l2_loss=torch.norm(self.parms,2)
            loss=self.loss_fn(scores,y)+self.lambda_*l2_loss
            avg_loss+=loss.item()
            avg_l2loss+=l2_loss.item()
            SCORE.append(scores.cpu().data.numpy())
            LABEL.append(y.cpu().data.numpy())
        S=np.concatenate(SCORE,0)
        L=np.concatenate(LABEL,0)
        pr,_=stats.pearsonr(S[:,0],L[:,0])
        avg_loss/=batch_idx
        avg_l2loss/=batch_idx
        return avg_loss,pr
    def Test(self,test_loader):
        tY=[]
        tScore=[]
        for batch_idx,(sx,mx,y) in enumerate(test_loader):
            scores=self.forward(sx,mx)
            tY.append(y)
            tScore.append(scores.clone().detach())
        tY=torch.cat(tY,0)
        tScore=torch.cat(tScore,0)
        return tScore.cpu().data.numpy(),tY.cpu().data.numpy()

