# extract the mhc-peptides in human species in length 8-15
awk '{if($1=="human" && length($4)<=15 && length($4)>=8) print $0}' bdata.20130222.mhci.txt > mhcI.tsv

# generate and encode the data
python datagenerate.py

#split the data into 5 fold for cross-validation, the split information is saved in 'cvindex.pkl' and the splited data is saved in 'cvdata.pkl'
python cvsplit.py
