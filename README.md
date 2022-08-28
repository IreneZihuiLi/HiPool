# HiPool

pyTorch implementation of HiPool submission paper **HiPool: Hierarchical Pooling for Long Document Classification**.

The code is based on this implementation:
[git link](https://github.com/helmy-elrais/RoBERT_Recurrence_over_BERT/blob/master/train.ipynb)

## Install
Tested on python 3.8.8.

`conda env create -f environment.yml`


## Running
```CUDA_VISIBLE_DEVICES=1 python train_imdb.py --sentlen 50 --adj_method bigbird --level sent --graph_type gat --epoch 10```

default dataset: consumer_complaints

## Main Scripts
`train_imdb.py`: main function.

`Dataset_Split_Class.py`: data loading.

`Bert_Classification.py`: modeling for BERT and graphs.

`Graph_Models.py`: graph model classes.

`Graph_Models_utils.py`: helper functions for graph model classes.
