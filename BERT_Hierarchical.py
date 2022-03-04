##############################################################
#
# BERT_Hierarchical.py
# This file contains the code to fine-tune BERT by computing
# segment tensors as a pooled result from all the segments
# obtained after tokenization.
#
##############################################################
import pandas as pd
import numpy as np
import time
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler

import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
# get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
from TransformerLayer import  BERT


class BERT_Hierarchical_Model(nn.Module):

    def __init__(self, device,pooling_method="mean"):
        super(BERT_Hierarchical_Model, self).__init__()

        self.pooling_method = pooling_method
        self.device = device

        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.out = nn.Linear(768, 10)

    def forward(self, ids, mask, token_type_ids, lengt):

        # import pdb;pdb.set_trace()
        results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        chunks_emb = results[1].split_with_sizes(lengt)



        if self.pooling_method == "mean":
            emb_pool = torch.stack([torch.mean(x, 0) for x in chunks_emb])
        elif self.pooling_method == "max":
            emb_pool = torch.stack([torch.max(x, 0)[0] for x in chunks_emb])
        # emb_pool: torch.Size([3, 768])

        return self.out(emb_pool)


class BERT_Hierarchical_LSTM_Model(nn.Module):

    def __init__(self, device, pooling_method="mean",lstm_layer_number=1,lstm_hidden_size=64):
        super(BERT_Hierarchical_LSTM_Model, self).__init__()

        self.pooling_method = pooling_method
        self.device = device

        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = lstm_layer_number
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layer_number,
            dropout=0.2,
        )

        self.out = nn.Linear(self.lstm_hidden_size, 10)


    def forward(self, ids, mask, token_type_ids, lengt):
        # lengt is a list [2,2,2]


        results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = results[1].split_with_sizes(lengt)


        'lstm starts'
        max_step = max(lengt)
        batch_size = len(lengt)
        lstm_input = torch.zeros(batch_size,max_step,768).to(self.device)

        for current_id, element in enumerate(lstm_input):
            'todo: deal with different shapes'
            lstm_input[current_id] = chunks_emb[current_id]

        lstm_input = lstm_input.permute(1,0,2)

        h0 = c0 = torch.zeros(self.lstm_layer_number,batch_size,self.lstm_hidden_size).to(self.device)
        outputs, (ht, ct) = self.lstm(lstm_input, (h0, c0))

        emb_pool = outputs[-1]
        'lstm ends'
        import pdb;pdb.set_trace()
        'outputs.shape torch.Size([2, 3, 64]),emb_pool shape torch.Size([3, 64])'

        return self.out(emb_pool)


class BERT_Hierarchical_BERT_Model(nn.Module):

    def __init__(self, device, pooling_method="mean",lstm_layer_number=1,lstm_hidden_size=32):
        super(BERT_Hierarchical_BERT_Model, self).__init__()

        self.pooling_method = pooling_method
        self.device = device

        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = lstm_layer_number
        self.lstm_hidden_size = lstm_hidden_size

        self.mapping = nn.Linear(768,lstm_hidden_size)
        self.BERTLayer = BERT(hidden=lstm_hidden_size, n_layers=1, attn_heads=8).to(device)
        self.out = nn.Linear(self.lstm_hidden_size, 10)


    def forward(self, ids, mask, token_type_ids, lengt):
        # lengt is a list [2,2,2]


        results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = results[1].split_with_sizes(lengt)


        'lstm starts'
        max_step = max(lengt)
        batch_size = len(lengt)
        lstm_input = torch.zeros(batch_size,max_step,768).to(self.device)

        for current_id, element in enumerate(lstm_input):
            'todo: deal with different shapes'
            lstm_input[current_id] = chunks_emb[current_id]

        lstm_input = lstm_input.permute(1,0,2)
        # shape: torch.Size([2, 3, 768]) [len, batch_size, dim]

        'lstm ends'
        # import pdb;pdb.set_trace()
        lstm_input = self.mapping(lstm_input)
        lstm_output=self.BERTLayer(lstm_input)
        'outputs.shape torch.Size([2, 3, 64]),emb_pool shape torch.Size([3, 64])'

        return self.out(lstm_output[-1])


class BERT_Hierarchical_BERT_Model(nn.Module):

    def __init__(self, device, pooling_method="mean",lstm_layer_number=1,lstm_hidden_size=32):
        super(BERT_Hierarchical_BERT_Model, self).__init__()

        self.pooling_method = pooling_method
        self.device = device

        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = lstm_layer_number
        self.lstm_hidden_size = lstm_hidden_size

        self.mapping = nn.Linear(768,lstm_hidden_size)
        self.BERTLayer = BERT(hidden=lstm_hidden_size, n_layers=1, attn_heads=8).to(device)
        self.out = nn.Linear(self.lstm_hidden_size, 10)


    def forward(self, ids, mask, token_type_ids, lengt):
        # lengt is a list [2,2,2]


        results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = results[1].split_with_sizes(lengt)


        'lstm starts'
        max_step = max(lengt)
        batch_size = len(lengt)
        lstm_input = torch.zeros(batch_size,max_step,768).to(self.device)

        for current_id, element in enumerate(lstm_input):
            'todo: deal with different shapes'
            lstm_input[current_id] = chunks_emb[current_id]

        lstm_input = lstm_input.permute(1,0,2)
        # shape: torch.Size([2, 3, 768]) [len, batch_size, dim]

        'lstm ends'
        # import pdb;pdb.set_trace()
        lstm_input = self.mapping(lstm_input)
        lstm_output=self.BERTLayer(lstm_input)
        'outputs.shape torch.Size([2, 3, 64]),emb_pool shape torch.Size([3, 64])'

        return self.out(lstm_output[-1])



# layer num, head num
# 1,1, 0.823220536756126
# 2,1, 0.8165110851808635
# 1,8,  0.8211785297549592
