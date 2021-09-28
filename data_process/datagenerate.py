import pickle
import numpy as np
import re

def read_blosum(path):
    '''
    Read the blosum matrix from the file blosum50.txt
    Args:
        1. path: path to the file blosum50.txt
    Return values:
        1. The blosum50 matrix
    '''
    f = open(path,"r")
    blosum = []
    for line in f:
        blosum.append([(float(i))/10 for i in re.split("\t",line)])
        #The values are rescaled by a factor of 1/10 to facilitate training
    f.close()
    return blosum

blosum_matrix = read_blosum('blosum50.txt')
aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
name=[]
seq=[]
with open('psudo.fa') as f:
    for l in f:
        if '>' in l:
            name.append(l.strip()[1:])
        else:
            seq.append(l.strip())
pseudo=dict((zip(name,seq)))

data={}
with open('mhcI.tsv') as f:
    for l in f:
        ll=l.strip().split('\t')
        allele=ll[1]
        seq=ll[3]
        score=1-(np.log(np.clip(float(ll[5]),0,50000))/np.log(50000))
        score=score.astype(np.float32)
        if allele in pseudo:
            mhc=pseudo[allele]
            if allele not in data:
                data[allele]=[[seq,mhc,score]]
            else:
                data[allele].append([seq,mhc,score])

edata={}
for d in data:
    seq,mhc,score=zip(*data[d])
    seqe=np.zeros((len(seq),24,20),dtype=np.float32)
    mhce=np.zeros((len(mhc),34,20),dtype=np.float32)
    for i,pep in enumerate(seq):
        for residue_index in range(34):
            mhce[i,residue_index]=blosum_matrix[aa[mhc[i][residue_index]]]
        pep_blosum = []#Encoded peptide seuqence
        for residue_index in range(12):
            if residue_index < len(pep):
                pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
            else:
                pep_blosum.append(np.zeros(20))
        for residue_index in range(12):
            if 12 - residue_index > len(pep):
                pep_blosum.append(np.zeros(20)) 
            else:
                pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 12 + residue_index]]])
        seqe[i]=np.array(pep_blosum)
    new=list(zip(seq,seqe,mhce,score,[d for _ in range(len(seq))]))
    edata[d]=new

with open('data_pan_DBT.pkl','wb') as f:
    pickle.dump(edata,f)
