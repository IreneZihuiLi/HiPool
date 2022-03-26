##############################################################
#
# Custom_Dataset_Class.py
# This file contains the code to load and prepare the dataset
# for use by BERT.
# It does preprocessing, segmentation and BERT features extraction
#
##############################################################

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import math
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
# get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time


class DatasetSplit(Dataset):
    """ Make preprocecing, tokenization and transform consumer complaints
    dataset into pytorch DataLoader instance.

    Parameters
    ----------
    tokenizer: BertTokenizer
        transform data into feature that bert understand
    max_len: int
        the max number of token in a sequence in bert tokenization. 
    overlap_len: int
        the maximum number of overlap token.
    chunk_len: int
        define the maximum number of word in a single chunk when spliting sample into a chumk
    approach: str
        define how to handle overlap token after bert tokenization.
    max_size_dataset: int
        define the maximum number of sample to used from data.
    file_location: str
        the path of the dataset.

    Attributes
    ----------
    data: array of shape (n_keept_sample,)
        prepocess data.
    label: array of shape (n_keept_sample,)
        data labels
    """
    def __init__(self, tokenizer, max_len, chunk_len=200, overlap_len=50, approach="all", max_size_dataset=None, file_location="./us-consumer-finance-complaints/consumer_complaints.csv", min_len=249):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len
        self.approach = approach
        self.min_len = min_len
        self.max_size_dataset = max_size_dataset
        self.data, self.label = self.process_data(file_location,)
        self.num_class = 10



    def process_data(self, file_location):
        """ Preprocess the text and label columns as describe in the paper.

        Parameters
        ----------
        file_location: str
            the path of the dataset file.

        Returns
        -------
        texts: array of shape (n_kept_sample,)
            preprocessed  sample
        labels: array (n_kept_sample,)
            samples labels transform into a numerical value
        """

        # Load the dataset into a pandas dataframe.
        if file_location.startswith('complaints'):
            file_location = './my_datasets/us-consumer-finance-complaints/consumer_complaints.csv'
            df = pd.read_csv(file_location, dtype="unicode")
            train_raw = df[df.consumer_complaint_narrative.notnull()]
            train_raw = train_raw.assign(
                len_txt=train_raw.consumer_complaint_narrative.apply(lambda x: len(x.split())))
            train_raw = train_raw[train_raw.len_txt > self.min_len]
            train_raw = train_raw[['consumer_complaint_narrative', 'product']]
            train_raw.reset_index(inplace=True, drop=True)
            train_raw.at[train_raw['product'] == 'Credit reporting',
                         'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
            train_raw.at[train_raw['product'] == 'Credit card',
                         'product'] = 'Credit card or prepaid card'
            train_raw.at[train_raw['product'] == 'Prepaid card',
                         'product'] = 'Credit card or prepaid card'
            train_raw.at[train_raw['product'] == 'Payday loan',
                         'product'] = 'Payday loan, title loan, or personal loan'
            train_raw.at[train_raw['product'] == 'Virtual currency',
                         'product'] = 'Money transfer, virtual currency, or money service'
            train_raw = train_raw.rename(
                columns={'consumer_complaint_narrative': 'text', 'product': 'label'})
        elif file_location.startswith('./IMDB'):
            df = pd.read_csv("./my_datasets/IMDB/IMDB.csv")
            train_raw = df[df.review.notnull()]
            train_raw = train_raw[['review', 'sentiment']]
            train_raw.reset_index(inplace=True, drop=True)
            train_raw = train_raw.rename(columns={'review': 'text', 'sentiment': 'label'})



        LE = LabelEncoder()
        train_raw['label'] = LE.fit_transform(train_raw['label'])
        train = train_raw.copy()
        if(self.max_size_dataset):
            train = train.loc[0:self.max_size_dataset, :]
        train = train.reindex(np.random.permutation(train.index))
        train['text'] = train.text.apply(self.clean_txt)
        'return string list in an object ndarrary, ad an int arrary for labels'
        self.num_class = len(set(train['label'].values))
        return train['text'].values, train['label'].values

    def clean_txt(self, text):
        """ Remove special characters from text """

        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text

    def long_terms_tokenizer(self, data_tokenize, targets):
        """  tranfrom tokenized data into a long token that take care of
        overflow token according to the specified approach.

        Parameters
        ----------
        data_tokenize: dict
            an tokenized result of a sample from bert tokenizer encode_plus method.
        targets: array
            labels of each samples.

        Returns
        _______
        long_token: dict
            a dictionnary that contains
             - [ids]  tokens ids
             - [mask] attention mask of each token
             - [token_types_ids] the type ids of each token. note that each token in the same sequence has the same type ids
             - [targets_list] list of all sample label after add overlap token as sample according to the approach used
             - [len] length of targets_list
        """

        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        print(data_tokenize["input_ids"].shape)
        print(previous_input_ids.shape)

        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids = data_tokenize["token_type_ids"].reshape(-1)
        print(previous_attention_mask.shape)
        remain = data_tokenize.get("overflowing_tokens")
        targets = torch.tensor(targets, dtype=torch.int)

        # input_ids_list.append(previous_input_ids)
        # attention_mask_list.append(previous_attention_mask)
        # token_type_ids_list.append(previous_token_type_ids)
        # targets_list.append(targets)



        'sementation new: Dec, 2021'


        start_token = torch.tensor([101], dtype=torch.long)
        end_token = torch.tensor([102], dtype=torch.long)

        total_token = len(previous_input_ids) -2 # remove head 101, tail 102
        stride = self.overlap_len - 2
        number_chunks = math.floor(total_token/stride)

        mask_list = torch.ones(self.chunk_len, dtype=torch.long)
        type_list = torch.zeros(self.chunk_len, dtype=torch.long)
        for current in range(number_chunks-1):
            input_ids = previous_input_ids[current*stride:current*stride+self.chunk_len-2]
            input_ids = torch.cat((start_token, input_ids, end_token))
            input_ids_list.append(input_ids)

            attention_mask_list.append(mask_list)
            token_type_ids_list.append(type_list)
            targets_list.append(targets)

        if len(input_ids_list) == 0:
            input_ids = torch.ones(self.chunk_len-2, dtype=torch.long)
            input_ids = torch.cat((start_token, input_ids, end_token))
            input_ids_list.append(input_ids)

            attention_mask_list.append(mask_list)
            token_type_ids_list.append(type_list)
            targets_list.append(targets)

        return({
            'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
            # torch.tensor(mask, dtype=torch.long),
            'mask': attention_mask_list,
            # torch.tensor(token_type_ids, dtype=torch.long),
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })


    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""
        
        consumer_complaint = str(self.data[idx])
        targets = int(self.label[idx])
        data = self.tokenizer.encode_plus(
            consumer_complaint,
            max_length=self.max_len,
            pad_to_max_length=False,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            truncation=True,
            return_tensors='pt')


        previous_input_ids = torch.tensor(data["input_ids"].reshape(-1), dtype=torch.long)
        previous_attention_mask = torch.tensor(data["attention_mask"].reshape(-1), dtype=torch.long)
        previous_token_type_ids = torch.tensor(data["token_type_ids"].reshape(-1), dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.int)
        a = {
            'ids': previous_input_ids, 
            'mask': previous_attention_mask,
            'token_type_ids':  previous_token_type_ids,
            'targets': targets
        }
        return a
    def __len__(self):
        """ Return data length """
        return self.label.shape[0]
