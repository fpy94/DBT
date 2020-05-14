import ete3
import argparse
from ete3 import Tree
import os
import pickle

def get_tree_edge_path(tree):
    edge=[]
    path=[]
    num=0
    for node in tree.traverse("preorder"):
        num+=1
        nodename=node.name
        e=[(nodename,n.name) for n in node.get_children()]
        edge.extend(e)
        path.append(tuple([n.name for n in tree.search_nodes(name=nodename)[0].iter_ancestors()][::-1]+[nodename]))
        if num%5000==0:
            print(num)
    return edge,path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-tpath','--tree_path',default=None,type=str,help='continuning training path')
    #parser.add_argument('-savedir','--save_dir',default=None,type=str,help='saving dir')
    args=parser.parse_args()
    savepath=args.tree_path.split('.pkl')[0]+'_edge.pkl'
    print(savepath)
    with open(args.tree_path, 'rb') as f:
        tree=pickle.load(f)['tree']
    edge,path=get_tree_edge_path(tree)
    with open(savepath, 'wb') as f:
        pickle.dump({'edge':edge,'path':path},f)
