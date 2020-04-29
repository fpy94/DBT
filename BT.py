import ete3
from ete3 import Tree
import numpy as np
def BTQuery(q,tree,dist,k=10):
    currentname=tree.get_tree_root().name
    while True:
        children=tree.search_nodes(name=currentname)[0].get_children()
        if len(children)<k:
            children.append(tree.search_nodes(name=currentname)[0])
        dmin=np.inf
        minnode=None
        for c in children:
            name=c.name
            d=dist[q,name]
            if d<dmin:
                dmin=d
                minnode=c.name
        if minnode==currentname:
            break
        currentname=minnode
    return currentname
def BTtrain(dist,label,eps=0.1):
    Nnodes=dist.shape[0]
    Nodelist=np.random.permutation(Nnodes)
    tree=Tree(name=Nodelist[0])
    querynum=1
    nodenum=1
    for nodeidx in Nodelist[1:]:
        vmin=BTQuery(nodeidx,tree,dist)
        if np.abs(label[vmin]-label[nodeidx])>eps:
#             tree.search_nodes(name=vmin)[0].add_child(name=nodeidx,dist=np.abs(label[vmin]-label[nodeidx]))
            tree.search_nodes(name=vmin)[0].add_child(name=nodeidx)
            nodenum+=1
        querynum+=1
        #if querynum%2500==0:
            #print('querynum {}, treenum {}'.format(querynum,nodenum))
#     print('tree building finished, treenum {}'.format(nodenum))
    return tree

def distance(test,train):
    d1=-2*np.dot(test,train.T)
    d2=np.sum(np.square(test),axis=1,keepdims=True)
    d3=np.sum(np.square(train),axis=1)
    dist=d1+d2+d3
    dist[np.where(dist<0)]=0
    dist=np.sqrt(dist)
    return dist
