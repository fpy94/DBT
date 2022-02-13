for ti in $(seq 0 4)
do
echo $ti
python query_tree_alldata.py -eps 0.05 -treeindex $ti -mpath ./data_for_test/checkpoint_epoch_25_bs_1000_lr_0.001.pkl -datadir ./test_data_process/testdata.pkl -tpath ./data_for_test/tree -savedir ./
done
