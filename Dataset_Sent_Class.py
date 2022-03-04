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
import json
from sklearn.preprocessing import LabelEncoder
import re
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


class DatasetSent(Dataset):
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

    def __init__(self, tokenizer, max_len, sentence_group_num, chunk_len,max_size_dataset=None, file_location="complaints"):
        self.tokenizer = tokenizer # bert tokenizer
        self.max_len = max_len
        self.sentence_group_num = sentence_group_num
        self.num_class = 10
        self.chunk_len = chunk_len
        self.max_size_dataset = max_size_dataset
        self.data, self.label = self.process_data(file_location,)



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
            # df = pd.read_csv(file_location, dtype="unicode")
            # train_raw = df[df.consumer_complaint_narrative.notnull()]
            # train_raw = train_raw.assign(
            #     len_txt=train_raw.consumer_complaint_narrative.apply(lambda x: len(x.split())))
            # train_raw = train_raw[train_raw.len_txt > self.min_len]
            # train_raw = train_raw[['consumer_complaint_narrative', 'product']]
            # train_raw.reset_index(inplace=True, drop=True)
            # train_raw.at[train_raw['product'] == 'Credit reporting',
            #              'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
            # train_raw.at[train_raw['product'] == 'Credit card',
            #              'product'] = 'Credit card or prepaid card'
            # train_raw.at[train_raw['product'] == 'Prepaid card',
            #              'product'] = 'Credit card or prepaid card'
            # train_raw.at[train_raw['product'] == 'Payday loan',
            #              'product'] = 'Payday loan, title loan, or personal loan'
            # train_raw.at[train_raw['product'] == 'Virtual currency',
            #              'product'] = 'Money transfer, virtual currency, or money service'
            # train_raw = train_raw.rename(
            #     columns={'consumer_complaint_narrative': 'text', 'product': 'label'})
            with open('./us-consumer-finance-complaints/consumer_complaints_sent.json') as f:
                load_data = json.load(f)

            labels = [x['label'] for x in load_data]

        # elif file_location.startswith('imdb'):
        else:
            # df = pd.read_csv("./IMDB/IMDB_sent.json")
            # with open('./us-consumer-finance-complaints/consumer_complaints_sent.json') as f:
            with open('./IMDB/imdb_sent.json') as f:
                load_data = json.load(f)

            labels = [x['label'] for x in load_data]
        'want all things in list, not df'

        LE = LabelEncoder()
        train_raw_labels = LE.fit_transform(labels)
        # train_raw_sentences = [self.clean_txt(x['Sentences']) for x in load_data]
        train_raw_sentences = [self.clean_txt(self.extend_sentence_length(x['Sentences'])) for x in load_data]
        # import pdb;pdb.set_trace()

        if(self.max_size_dataset):
            train_raw_labels = train_raw_labels[0:self.max_size_dataset]
            train_raw_sentences = train_raw_sentences[0:self.max_size_dataset]


        'return string list in an object ndarrary, ad an int arrary for labels'
        # return train['text'].values, train['label'].values

        self.num_class = len(set(train_raw_labels))

        return train_raw_sentences,train_raw_labels

    def extend_sentence_length(self, sentences):
        '''this is to combine short sentences'''
        sentences_combined = []
        sub_combined = []
        for sentence in sentences:
            if len(sub_combined) <= self.sentence_group_num:
                sub_combined.append(sentence)
            else:
                sentences_combined.append(' '.join(sub_combined))
                sub_combined = []
        sentences_combined.append(' '.join(sub_combined))
        return sentences_combined

    def clean_txt(self, sentences):
        """ Remove special characters from text """

        for i,text in enumerate(sentences):
            text = re.sub("'", "", text)
            text = re.sub("<br />","", text)
            text = re.sub("(\\W)+", " ", text)
            sentences[i] = text

        return sentences



    def sentence_tokenizer(self,idx):

        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        for item in self.data[idx]:
            single_data = str(item)

            targets = int(self.label[idx])
            'encode plus returns more info: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus'
            '''data is a dict
            dict_keys(['overflowing_tokens', 'num_truncated_tokens', 'input_ids', 'token_type_ids', 'attention_mask'])
            work on a single sentence level
            '''
            data = self.tokenizer.encode_plus(
                single_data,
                max_length=self.chunk_len,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_length=True,
                return_tensors='pt')
            if len(targets_list) < 1:
                targets = torch.tensor(targets, dtype=torch.int)
                targets_list.append(targets)
            input_ids_list.append(data['input_ids'].reshape(-1))
            attention_mask_list.append(data['attention_mask'].reshape(-1))
            token_type_ids_list.append(data['token_type_ids'].reshape(-1))



        return ({
            'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
            # torch.tensor(mask, dtype=torch.long),
            'mask': attention_mask_list,
            # torch.tensor(token_type_ids, dtype=torch.long),
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(input_ids_list), dtype=torch.long)]
        })


    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""

        # consumer_complaint = str(self.data[idx][0])
        long_token = self.sentence_tokenizer(idx)
        'this is for testing only'
        # consumer_complaint = str(' '.join(self.data[idx])) +  str(' '.join(self.data[idx]))
        #
        # targets = int(self.label[idx])
        # 'encode plus returns more info: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus'
        # '''data is a dict
        # dict_keys(['overflowing_tokens', 'num_truncated_tokens', 'input_ids', 'token_type_ids', 'attention_mask'])
        # work on a single sentence level
        # '''
        # data = self.tokenizer.encode_plus(
        #     consumer_complaint,
        #     max_length=self.chunk_len,
        #     pad_to_max_length=True,
        #     add_special_tokens=True,
        #     return_attention_mask=True,
        #     return_token_type_ids=True,
        #     return_overflowing_tokens=True,
        #     return_length=True,
        #     return_tensors='pt')
        # long_token = self.long_terms_tokenizer(data, targets)


        return long_token

    def __len__(self):
        """ Return data length """
        return self.label.shape[0]
