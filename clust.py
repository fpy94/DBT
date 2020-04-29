import numpy as np
import pickle
import os
def read_dismat():
	datanum={}
	with open('../../data/iedb2013/mhcI_nonoverlap_allelenum_sort.tsv') as f:
		for l in f:
			ll=l.strip().split()
			datanum[ll[0]]=int(ll[1])
	keys=[]
	clumat=[]
	i=0
	with open('../../data/iedb2013/psudo_nonoverlap.fa.mat') as f:
		for l in f:
			if i==0:
				i+=1
				continue
			else:
				ll=l.strip().split()
				if datanum[ll[0]]>10:
					keys.append(ll[0])
					clumat.append(np.array([float(j) for j in ll[1:]]))
				i+=1
	allelenum=len(keys)
	clumat=np.array(clumat)
	return keys,clumat

def MHCKNN(K=5):
	keys,clumat=read_dismat()
	sort=np.argsort(clumat,-1)
	k_near=[]
	i=0
	for i in range(sort.shape[0]):
		row=[]
		for j in range(sort.shape[1]):
			if sort[i,j]!=i:
				row.append(keys[sort[i,j]])
			if len(row)==K:
				break
		k_near.append(row)
	return keys,k_near

	

def selectdata(data,testallele,testidx,pratio,n_support,topallele):
	savedir='./rs_model_file/'+testallele
	# allele list
	allele_list,allele_near=MHCKNN(K=1)
	taskidx=[0]
	tasklist=[]
	totaldata=[]
	for d in allele_list:
		if d!=testallele:
			taskidx.append(taskidx[-1]+len(data[d]))
			tasklist.append(d)
			totaldata+=data[d]
	#load testscore trainsample*testsample
	with open(os.path.join(savedir,'testscore.pkl'),'rb') as f:
		testscore=pickle.load(f)
	abstestscore=np.abs(testscore)
	abstest=np.concatenate([np.expand_dims(abstestscore[i],0) for i in testidx[:n_support]],0)

	topN=int(pratio*testscore.shape[1])#select top N training samples
	absargsort=np.argsort(abstest,1)[:,::-1][:,:topN]

	#choose the overlaping
	trainidx_=list(np.reshape(absargsort,-1))
	trainidx=[]
	#remove overlap
	for i in trainidx_:
		if i not in trainidx:
			trainidx.append(i)
	trainidx=list(sorted(trainidx))

	#chosse the training data
	newdata={}
	datanum={}
	for idx in trainidx:
		#find this allele
		for i in range(len(taskidx)):
			if idx>=taskidx[i] and idx<taskidx[i+1]:
				allele=tasklist[i]
				break
		if allele not in newdata:
			newdata[allele]=[totaldata[idx]]
			datanum[allele]=1
		else:
			newdata[allele].append(totaldata[idx])
			datanum[allele]+=1
	datanumsort=sorted(datanum.items(), key=lambda k:k[1], reverse=True)[:topallele]
	newdatatop={}
	newdatalist,num=zip(*datanumsort)
	for allele in newdatalist:
		newdatatop[allele]=newdata[allele]

	#generate test dataset
	testdata=data[testallele]
	retestdata=[testdata[i] for i in testidx]
	seq,soh,mhcoh,score=zip(*retestdata)
	sx=np.concatenate([np.expand_dims(i,0) for i in soh],0)
	sx=sx.astype(np.float32)
	sx/=10
	mx=np.concatenate([np.expand_dims(i,0) for i in mhcoh],0)
	mx=mx.astype(np.float32)
	mx/=10

	y=np.expand_dims(np.array(score),1)
	y=y.astype(np.float32)

	testdata=[sx,mx,y]
	return newdatatop,newdatalist,testdata

def selectdata_cos(data,testallele,testidx,pratio,n_support):
	savedir='./rs_model_file/'+testallele
	# allele list
	allele_list,allele_near=MHCKNN(K=1)
	taskidx=[0]
	tasklist=[]
	totaldata=[]
	for d in allele_list:
		if d!=testallele:
			taskidx.append(taskidx[-1]+len(data[d]))
			tasklist.append(d)
			totaldata+=data[d]
	#load feature of trainsample&testsample
	with open(os.path.join(savedir,'decomp.pkl'),'rb') as f:
		Feature=pickle.load(f)
	fi=Feature['fi']
	testfeature=Feature['testfeature']
	prefi=np.matmul(testfeature,fi.T)/(np.matmul(np.linalg.norm(testfeature,axis=1,keepdims=True),np.linalg.norm(fi.T,axis=0,keepdims=True)))
	prefi_=np.concatenate([np.expand_dims(prefi[i],0) for i in testidx[:n_support]],0)#select query test samples score
	'''
	prefi_mean=np.mean(prefi_,0)
	trainidx=np.where(prefi_mean>pratio)[0]

	#chosse the training data
	newdata={}
	datanum={}
	for idx in trainidx:
		#find this allele
		for i in range(len(taskidx)):
			if idx>=taskidx[i] and idx<taskidx[i+1]:
				allele=tasklist[i]
				break
		if allele not in newdata:
			newdata[allele]=[totaldata[idx]]
			datanum[allele]=1
		else:
			newdata[allele].append(totaldata[idx])
			datanum[allele]+=1
	'''
	prefitask=[]
	for i in range(len(tasklist)):
		prefitask.append(np.mean(prefi[:,taskidx[i]:taskidx[i+1]]))
	prefitask=np.array(prefitask)
	if np.max(prefitask) >0.5:
		selected_task=np.where(prefitask>0.5)[0]
	elif np.max(prefitask) >0.4:
		selected_task=np.where(prefitask>0.4)[0]
	elif np.max(prefitask) >0.3:
		selected_task=np.where(prefitask>0.3)[0]
	newdata={}
	datanum={}
	for idx in selected_task:
		newdata[tasklist[idx]]=data[tasklist[idx]]
		datanum[tasklist[idx]]=len(data[tasklist[idx]])
	datanumsort=sorted(datanum.items(), key=lambda k:k[1], reverse=True)
	newdatalist,num=zip(*datanumsort)
	print(datanumsort)
	#generate test dataset
	testdata=data[testallele]
	retestdata=[testdata[i] for i in testidx]
	seq,soh,mhcoh,score=zip(*retestdata)
	sx=np.concatenate([np.expand_dims(i,0) for i in soh],0)
	sx=sx.astype(np.float32)
	sx/=10
	mx=np.concatenate([np.expand_dims(i,0) for i in mhcoh],0)
	mx=mx.astype(np.float32)
	mx/=10

	y=np.expand_dims(np.array(score),1)
	y=y.astype(np.float32)

	testdata=[sx,mx,y]
	return newdata,newdatalist,testdata

def selectdata_cos_single(data,testallele,testidx,pratio,n_support):
	single_data={}
	for d in data:
		seq,soh,mhcoh,score=zip(*data[d])
		single_data[d]=list(zip(seq,soh,score))
	data=single_data
	savedir='./rs_model_file/'+testallele
	# allele list
	allele_list,allele_near=MHCKNN(K=1)
	taskidx=[0]
	tasklist=[]
	totaldata=[]
	for d in allele_list:
		if d!=testallele:
			taskidx.append(taskidx[-1]+len(data[d]))
			tasklist.append(d)
			totaldata+=data[d]
	#load feature of trainsample&testsample
	with open(os.path.join(savedir,'decomp.pkl'),'rb') as f:
		Feature=pickle.load(f)
	fi=Feature['fi']
	testfeature=Feature['testfeature']
	prefi=np.matmul(testfeature,fi.T)/(np.matmul(np.linalg.norm(testfeature,axis=1,keepdims=True),np.linalg.norm(fi.T,axis=0,keepdims=True)))
	prefi_=np.concatenate([np.expand_dims(prefi[i],0) for i in testidx[:n_support]],0)#select query test samples score
	'''
	prefi_mean=np.mean(prefi_,0)
	trainidx=np.where(prefi_mean>pratio)[0]

	#chosse the training data
	newdata={}
	datanum={}
	for idx in trainidx:
		#find this allele
		for i in range(len(taskidx)):
			if idx>=taskidx[i] and idx<taskidx[i+1]:
				allele=tasklist[i]
				break
		if allele not in newdata:
			newdata[allele]=[totaldata[idx]]
			datanum[allele]=1
		else:
			newdata[allele].append(totaldata[idx])
			datanum[allele]+=1
	'''
	prefitask=[]
	for i in range(len(tasklist)):
		prefitask.append(np.mean(prefi[:,taskidx[i]:taskidx[i+1]]))
	prefitask=np.array(prefitask)
	if np.max(prefitask) >0.5:
		selected_task=np.where(prefitask>0.5)[0]
	elif np.max(prefitask) >0.4:
		selected_task=np.where(prefitask>0.4)[0]
	elif np.max(prefitask) >0.3:
		selected_task=np.where(prefitask>0.3)[0]
	newdata={}
	datanum={}
	for idx in selected_task:
		newdata[tasklist[idx]]=data[tasklist[idx]]
		datanum[tasklist[idx]]=len(data[tasklist[idx]])
	datanumsort=sorted(datanum.items(), key=lambda k:k[1], reverse=True)
	newdatalist,num=zip(*datanumsort)
	print(datanumsort)
	#generate test dataset
	testdata=data[testallele]
	retestdata=[testdata[i] for i in testidx]
	seq,soh,score=zip(*retestdata)
	sx=np.concatenate([np.expand_dims(i,0) for i in soh],0)
	sx=sx.astype(np.float32)
	sx/=10

	y=np.expand_dims(np.array(score),1)
	y=y.astype(np.float32)

	testdata=[sx,y]
	return newdata,newdatalist,testdata
