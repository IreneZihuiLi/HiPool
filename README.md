# RoBERT Recurrence over BERT
pyTorch implementation of Recurrence over BERT (RoBERT) based on this paper https://arxiv.org/abs/1910.10781
and comparison with pyTorch implementation of other Hierarchical Methods (Mean Pooling and Max Pooling) and Truncation Methods (Head Only and Tail Only) presented in this paper https://arxiv.org/abs/1905.05583 

[git link](https://github.com/helmy-elrais/RoBERT_Recurrence_over_BERT/blob/master/train.ipynb)




```CUDA_VISIBLE_DEVICES=1 python train_imdb.py --sentlen 50 --adj_method bigbird --level sent --graph_type gat --epoch 3```
