import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import stats

class CNNpan(nn.Module):
    def __init__(self):
        super(CNNpan,self).__init__()
        self.lambda_=lambda_

        self.pconv1=nn.Sequential(
            nn.Conv1d(21,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.mconv1=nn.Sequential(
            nn.Conv1d(21,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.mconv2=nn.Sequential(
            nn.Conv1d(128,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )


        self.MLP=nn.Sequential(
            nn.Linear(15*128,256),
            nn.ReLU()
        )
        self.MLP2=nn.Sequential(
            nn.Linear(256,20),
            nn.ReLU()
        )
        self.MLP3=nn.Linear(20,1,bias=False)
        self.sigmoid=nn.Sigmoid()

        self.loss_fn=nn.MSELoss()

    def extract_feature(self,sx,mx):
        sx=Variable(sx.cuda())
        mx=Variable(mx.cuda())
        sx=sx.permute(0,2,1)
        mx=mx.permute(0,2,1)
        peptideout=self.pconv1(sx)#batch,128,7
        mhcout1=self.mconv1(mx)
        mhcout2=self.mconv2(mhcout1)#batch,128,8
        concat=torch.cat((peptideout.view(peptideout.size(0),-1),mhcout2.view(mhcout2.size(0),-1)),1)
        self.feature=self.MLP(concat)
        self.feature=self.MLP2(self.feature)
        return self.feature

    def forward(self,sx,mx):
		self.feature=self.extract_feature(sx,mx)
        self.represent=self.MLP3(self.feature)
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
