# DBT
Differential boundary tree implemented in pytorch.

# Requiements:
python3<br>
pytorch >1.1.0<br>
numpy >1.17.2<br>
ete3 >3.1.1<br>
scipy >1.3.1<br>

# Usage and hyper-parameters:
## Train DBT
python3 train_init.py -savedir ./DBT_model_init -nfor 5 -cv 0 -bs 1000 -maxepoch 40 -gpu 1<br>
-lr learning rate <br>
-bs batch size, 1000 as recomand, in which 500 are used to build tree and the other are used to query the tree and train the model <br>
-nfor number of trees in each mini-batch. Generally, each mini-batch will randomly build several trees to train the model, default 5<br>
-maxepoch max epochs<br>
-eps the absolute distance threshold of label between query node and its nearest node in tree<br>
-savedir the output directory of trained model<br>
-mpath continuning training model file<br>
## Build boundary tree using trained model
python3 -u build_tree.py -cv 0 -eps 0.05 -datadir ./data_process/cvdata.pkl -mpath ./DBT_model_init_rep/checkpoint_cv_0_epoch_25_bs_1000_lr_0.001_rep_0.pkl -savedir ./tree_nfor_rep_0/<br>
-cv cross_validation index<br>
-savedir the output directory of built boundary tree<br>
-mpath the trained model used to build boundary tree
