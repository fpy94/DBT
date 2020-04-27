import numpy as np
import torch
import torch.nn as nn
import pickle
import random
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import KFold

			

class allele_dataset:
	def __init__(self,data):
		seq,soh,mhcoh,score=zip(*data)
		self.sx=np.concatenate([np.expand_dims(i,0) for i in soh],0)
		self.sx=self.sx.astype(np.float32)
		self.sx/=10
		self.mx=np.concatenate([np.expand_dims(i,0) for i in mhcoh],0)
		self.mx=self.mx.astype(np.float32)
		self.mx/=10

		self.y=np.expand_dims(np.array(score),1)
		self.y=self.y.astype(np.float32)
	def __getitem__(self,i):
		return self.sx[i],self.mx[i],self.y[i]
	def __len__(self):
		return len(self.sx)

class meta_dataset:
	def __init__(self,data,allele_list,n_support,n_query):
		self.data=data
		self.allele_list=allele_list
		self.n_support=n_support
		self.n_query=n_query      
		self.batch_size=self.n_support+self.n_query
		
		self.dataset=[]
		for key in self.allele_list:
			if key in self.data:
				self.dataset.append(DataLoader(allele_dataset(self.data[key]),batch_size=self.batch_size,shuffle=True))
		
	def __getitem__(self,i):
		return next(iter(self.dataset[i]))
	def __len__(self):
		return len(self.dataset)


class EpisodicBatchSampler(object):
	def __init__(self,dataset,n_way,n_episode):
		self.n_episode=n_episode
		self.n_way=n_way
		self.allele_num=len(dataset)
	def __len__(self):
		return self.n_episode
	def __iter__(self):
		for i in range(self.n_episode):
			yield torch.randperm(self.allele_num)[:self.n_way]
class Meta_Loader():
	def __init__(self,datafile,allelelist,batch_size,n_support,n_query,n_episode=100,mode='train'):
		self.mode=mode
		self.batch_size=batch_size
		self.n_support=n_support
		self.n_query=n_query
		self.n_episode=n_episode
		self.allelelist=allelelist
		with open(datafile,'rb') as f:
			self.data=pickle.load(f)['data']
	def get_data_loader(self):
		metad=meta_dataset(self.data,self.allelelist,self.n_support,self.n_query)
		sampler=EpisodicBatchSampler(metad,self.batch_size,self.n_episode)
		if self.mode=='test':
			data_loader=DataLoader(metad,batch_size=1,shuffle=False)
		else:
			data_loader=DataLoader(metad,batch_sampler=sampler)
		return data_loader

class Meta_Loader_4rs():
	def __init__(self,data,allelelist,batch_size,n_support,n_query,n_episode=100,mode='train'):
		self.mode=mode
		self.batch_size=batch_size
		self.n_support=n_support
		self.n_query=n_query
		self.n_episode=n_episode
		self.allelelist=allelelist
		self.data=data
	def get_data_loader(self):
		metad=meta_dataset(self.data,self.allelelist,self.n_support,self.n_query)
		sampler=EpisodicBatchSampler(metad,self.batch_size,self.n_episode)
		if self.mode=='test':
			data_loader=DataLoader(metad,batch_size=1,shuffle=False)
		else:
			data_loader=DataLoader(metad,batch_sampler=sampler)
		return data_loader

class allele_dataset_single:
	def __init__(self,data):
		seq,soh,score=zip(*data)
		self.sx=np.concatenate([np.expand_dims(i,0) for i in soh],0)
		self.sx=self.sx.astype(np.float32)
		self.sx/=10

		self.y=np.expand_dims(np.array(score),1)
		self.y=self.y.astype(np.float32)
	def __getitem__(self,i):
		return self.sx[i],self.y[i]
	def __len__(self):
		return len(self.sx)
class meta_dataset_single:
	def __init__(self,data,allele_list,n_support,n_query):
		self.data=data
		self.allele_list=allele_list
		self.n_support=n_support
		self.n_query=n_query      
		self.batch_size=self.n_support+self.n_query
		
		self.dataset=[]
		for key in self.allele_list:
			if key in self.data:
				self.dataset.append(DataLoader(allele_dataset_single(self.data[key]),batch_size=self.batch_size,shuffle=True))
		
	def __getitem__(self,i):
		return next(iter(self.dataset[i]))
	def __len__(self):
		return len(self.dataset)
class Meta_Loader_4rs_single():
	def __init__(self,data,allelelist,batch_size,n_support,n_query,n_episode=100,mode='train'):
		self.mode=mode
		self.batch_size=batch_size
		self.n_support=n_support
		self.n_query=n_query
		self.n_episode=n_episode
		self.allelelist=allelelist
		self.data=data
	def get_data_loader(self):
		metad=meta_dataset_single(self.data,self.allelelist,self.n_support,self.n_query)
		sampler=EpisodicBatchSampler(metad,self.batch_size,self.n_episode)
		if self.mode=='test':
			data_loader=DataLoader(metad,batch_size=1,shuffle=False)
		else:
			data_loader=DataLoader(metad,batch_sampler=sampler)
		return data_loader

