# RoBERT Recurrence over BERT
pyTorch implementation of Recurrence over BERT (RoBERT) based on this paper https://arxiv.org/abs/1910.10781
and comparison with pyTorch implementation of other Hierarchical Methods (Mean Pooling and Max Pooling) and Truncation Methods (Head Only and Tail Only) presented in this paper https://arxiv.org/abs/1905.05583 

[original git link](https://github.com/helmy-elrais/RoBERT_Recurrence_over_BERT/blob/master/train.ipynb)

## Install
Tested on python 3.8.8.

`conda env create -f environment.yml`


## Running
```CUDA_VISIBLE_DEVICES=6 python train_imdb.py --sentlen 100 --adj_method kk --level sent --graph_type gcn --epoch 10 --dataset imdb --lr 5e-7```


default dataset: imdb

## Main Scripts
`train_imdb.py`: main function.

`Dataset_Split_Class.py`: data loading.

`Bert_Classification.py`: modeling for BERT and graphs.

`Graph_Models.py`: graph model classes.

`Graph_Models_utils.py`: helper functions for graph model classes.
