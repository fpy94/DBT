# DBT
Differential boundary tree implemented in pytorch.

# Requiements:
python3<br>
pytorch >1.1.0<br>
numpy >1.17.2<br>
ete3 >3.1.1<br>
scipy >1.3.1<br>

# usage and hyper-parameters:
python3 train.py -lr 0.001 -bs 1000 -nfor 1 -eps 0.1<br>
-lr learning rate <br>
-bs batch size, 1000 as recomand, in which 500 are used to build tree and the other are used to query the tree and train the model <br>
-nfor number of trees in each mini-batch. Generally, each mini-batch will randomly build several trees to train the model, default 1<br>
-maxepoch max epochs<br>
-eps the absolute distance threshold of label between query node and its nearest node in tree<br>

