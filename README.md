# DBT
Differential boundary tree implemented in pytorch.

# Requiements:
python3<br>
pytorch >1.1.0<br>
numpy >1.17.2<br>
ete3 >3.1.1<br>
scipy >1.3.1<br>

# Usage and hyper-parameters:
## Pretrain
To pretrian the model, run: <br>
python pretrain.py<br>
We only pretrin one epoch using backbone model, the pretrain model will be saved in pretrain/ dir.<br> 
## Train DBT
python train_cv.py -lr 0.001 -bs 1000 -nfor 5 -eps 0.1 -maxepoch 25 -bs 1000<br>
-lr learning rate <br>
-bs batch size, 1000 as recomand, in which 500 are used to build tree and the other are used to query the tree and train the model <br>
-nfor number of trees in each mini-batch. Generally, each mini-batch will randomly build several trees to train the model, default 1<br>
-maxepoch max epochs<br>
-eps the absolute distance threshold of label between query node and its nearest node in tree<br>
-savedir the output directory of trained model<br>
-mpath continuning training model file<br>
## Build boundary tree using trained model
python build_tree.py -cv 0 -datadir ./data_process/cvdata.pkl -rep 0 -eps 0.05 -mpath MODEL_PATH -savedir SAVE_PATH<br>
-cv cross_validation index<br>
-savedir the output directory of built boundary tree<br>
-mpath the trained model used to build boundary tree<br>
-rep specify the tree index, we recommend five trees for one model<br>
## Query the boundary tree
python query_tree.py -cv 0 -eps 0.05 -rep 0 -mpath MODEL_PATH -datadir ./data_process/cvdata.pkl -tpath TREE_PATH -savedir SAVE_PATH <br>
-rep the tree index<br>

