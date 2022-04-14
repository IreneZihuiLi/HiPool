##############################################################
#
# Bert_Classification.py
# This file contains the code for fine-tuning BERT using a
# simple classification head.
#
##############################################################
import torch
import networkx as nx
import torch.nn as nn
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
import numpy as np
import transformers
# get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time
from torch.nn.utils.rnn import pad_sequence
from TransformerLayer import  BERT

from utils import kronecker_generator


class Bert_Classification_Model(nn.Module):
    """ A Model for bert fine tuning """

    def __init__(self):
        super(Bert_Classification_Model, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        # self.bert_drop=nn.Dropout(0.2)
        # self.fc=nn.Linear(768,256)
        # self.out=nn.Linear(256,10)
        self.out = nn.Linear(768, 10)
        # self.relu=nn.ReLU()

    def forward(self, ids, mask, token_type_ids):
        """ Define how to perfom each call

        Parameters
        __________
        ids: array
            -
        mask: array
            - 
        token_type_ids: array
            -

        Returns
        _______
            - 
        """
        import pdb;pdb.set_trace()
        'original'
        results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)


        return self.out(results[1])


class Hi_Bert_Classification_Model(nn.Module):
    """ A Model for bert fine tuning """

    def __init__(self,num_class,device,pooling_method='mean'):
        super(Hi_Bert_Classification_Model, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.out = nn.Linear(768, num_class)
        self.device = device
        self.pooling_method=pooling_method


    def forward(self, ids, mask, token_type_ids):



        if self.pooling_method == "mean":
            emb_pool = torch.stack([torch.mean(x.float(), 0) for x in ids]).long().to(self.device)
        elif self.pooling_method == "max":
            emb_pool = torch.stack([torch.max(x.float(), 0)[0] for x in ids]).long().to(self.device)
        emb_mask = torch.stack([x[0] for x in mask]).long().to(self.device)
        emb_token_type_ids = torch.stack([x[0] for x in token_type_ids]).long().to(self.device)

        'original'
        results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)


        return self.out(results[1]) # (batch_size, class_number)


class Hi_Bert_Classification_Model_LSTM(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self,num_class,device,pooling_method='mean'):
        super(Hi_Bert_Classification_Model_LSTM, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        self.lstm_hidden_size = 128

        self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method=pooling_method


        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layer_number,
            dropout=0.2,
        )
        self.out = nn.Linear(self.lstm_hidden_size, num_class)

    def forward(self, ids, mask, token_type_ids):

        'encode bert'
        bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
        batch_bert = []
        for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[1])

        sent_bert = self.bert_lstm(torch.stack(batch_bert, 0)) # (batch, step, 128)


        'lstm starts'
        batch_size = sent_bert.shape[0]
        lstm_input = sent_bert.permute(1,0,2)

        h0 = c0 = torch.zeros(self.lstm_layer_number, batch_size, self.lstm_hidden_size).to(self.device)

        outputs, (ht, ct) = self.lstm(lstm_input, (h0, c0))

        lstm_out = self.out(outputs[-1]) # shape torch.Size([batch, 128])
        'lstm ends'


        return lstm_out # (batch_size, class_number)


class Hi_Bert_Classification_Model_BERT(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self,num_class,device,pooling_method='mean'):
        super(Hi_Bert_Classification_Model_BERT, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        self.lstm_hidden_size = 128

        # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method=pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)
        self.BERTLayer = BERT(hidden=self.lstm_hidden_size, n_layers=1, attn_heads=8).to(device)
        self.out = nn.Linear(self.lstm_hidden_size, num_class).to(device)

    def forward(self, ids, mask, token_type_ids):

        'encode bert'
        bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
        batch_bert = []

        for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[1])

        sent_bert = torch.stack(batch_bert, 0)

        'BERT starts'
        lstm_input = sent_bert.permute(1,0,2)


        lstm_input = self.mapping(lstm_input)
        lstm_output = self.BERTLayer(lstm_input)
        'lstm ends'

        # import pdb;
        # pdb.set_trace()
        return self.out(lstm_output[-1]) # (batch_size, class_number)


from Graph_Models import GCN,GAT,GraphSAGE,SimpleRank,LinearFirst,DiffPool,HiPool
class Hi_Bert_Classification_Model_GCN(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self,args,num_class,device,adj_method,pooling_method='mean'):
        super(Hi_Bert_Classification_Model_GCN, self).__init__()
        self.args = args

        if self.args.tf_base.startswith('roberta'):
            self.bert_path = 'roberta-base'
            self.bert = transformers.RobertaModel.from_pretrained(self.bert_path)
            print ('Using roberta-base')
        else:
            self.bert_path = 'bert-base-uncased'
            self.bert = transformers.BertModel.from_pretrained(self.bert_path)
            print ('Using bert-base')



        self.lstm_layer_number = 2
        'default 128 and 32'
        self.lstm_hidden_size = args.lstm_dim
        self.hidden_dim = args.hid_dim

        # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method=pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)

        'start GCN'
        if self.args.graph_type == 'gcn':
            self.gcn = GCN(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'gat':
            self.gcn = GAT(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'graphsage':
            self.gcn = GraphSAGE(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'linear':
            self.gcn = LinearFirst(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'rank':
            self.gcn = SimpleRank(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'diffpool':
            self.gcn = DiffPool(self.device,max_nodes=10,input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'hipool':
            self.gcn = HiPool(self.device,input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)

        self.adj_method = adj_method

        'conv from token to sent'
        self.conv1d = nn.Conv1d(in_channels=args.sentlen, out_channels=1, kernel_size=1, stride=1)


    def forward(self, ids, mask, token_type_ids):


        'encode bert'
        bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
        batch_bert = []

        for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[1])

        # april overlap: this is not correct
        # if not self.args.overlap:
        #     for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
        #         results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
        #         batch_bert.append(results[1])
        #
        #         # apply cnn for BERT sentence, or just a summation representation
        #         # if self.args.apply_conv:
        #         #     # add cnn
        #         #     batch_bert.append(self.conv1d(results[0]).squeeze(1))
        #         # else:
        #         #     # original
        #         #     batch_bert.append(results[1])
        #         #     # results[1] torch.Size([num_sent, 768])
        #         #     # results[0] torch.Size([num_sent, length, 768])
        # else:
        #     for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
        #         results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
        #
        #         'compute overlap'
        #         bridge_nodes = results[0].reshape((-1,768))[int(self.args.sentlen/2):-int(self.args.sentlen/2)]
        #         bridge_nodes = bridge_nodes.reshape((-1,self.args.sentlen,768))
        #         bridge_nodes = torch.sum(bridge_nodes,1) # sum over the dim of sentlen
        #
        #         'add overlap'
        #         bert_nodes = results[1] # this is to only take the [cls]
        #         overlap_nodes = bridge_nodes
        #         reorder_all = []
        #         tb_i = 0
        #         ta_i = 0
        #         for ind in range(len(bert_nodes) + len(overlap_nodes)):
        #             if ind % 2 == 0:
        #                 reorder_all.append(bert_nodes[tb_i])
        #                 tb_i += 1
        #             else:
        #                 reorder_all.append(overlap_nodes[ta_i])
        #                 ta_i += 1
        #         reorder_all = torch.stack(reorder_all, 0)
        #
        #         batch_bert.append(reorder_all)


        sent_bert = torch.stack(batch_bert, 0)



        'GCN starts'
        sent_bert = self.mapping(sent_bert)
        node_number = sent_bert.shape[1]



        'random, using networkx'

        if self.adj_method == 'random':
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)
        elif self.adj_method == 'er':
            generated_adj = nx.erdos_renyi_graph(node_number, node_number)
        elif self.adj_method == 'binom':
            generated_adj = nx.binomial_graph(node_number, p=0.5)
        elif self.adj_method == 'path':
            generated_adj = nx.path_graph(node_number)
        elif self.adj_method == 'complete':
            generated_adj = nx.complete_graph(node_number)
        elif self.adj_method == 'kk':
            generated_adj = kronecker_generator(node_number)
        elif self.adj_method == 'watts':
            if node_number-1 > 0:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number-1, p=0.5)
            else:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number, p=0.5)
        elif self.adj_method == 'ba':
            if node_number - 1>0:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number-1)
            else:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number)
        elif self.adj_method == 'bigbird':

            # following are attention edges
            attention_adj = np.zeros((node_number, node_number))
            global_attention_step = 2
            attention_adj[:, :global_attention_step] = 1
            attention_adj[:global_attention_step, :] = 1
            np.fill_diagonal(attention_adj,1) # fill diagonal with 1
            half_sliding_window_size = 1
            np.fill_diagonal(attention_adj[:,half_sliding_window_size:], 1)
            np.fill_diagonal(attention_adj[half_sliding_window_size:, :], 1)
            generated_adj = nx.from_numpy_matrix(attention_adj)

        else:
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)


        nx_adj = from_networkx(generated_adj)
        adj = nx_adj['edge_index'].to(self.device)

        'combine starts'
        # generated_adj2 = nx.dense_gnm_random_graph(node_number,node_number)
        # nx_adj = from_networkx(generated_adj)
        # adj = nx_adj['edge_index'].to(self.device)
        # nx_adj2 = from_networkx(generated_adj2)
        # adj2 = nx_adj2['edge_index'].to(self.device)
        # adj = torch.cat([adj2, adj], 1)
        'combine ends'


        if self.adj_method == 'complete':
            'complete connected'
            adj = torch.ones((node_number,node_number)).to_sparse().indices().to(self.device)

        if self.args.graph_type.endswith('pool'):
            'diffpool only accepts dense adj'
            adj_matrix = nx.adjacency_matrix(generated_adj).todense()
            adj_matrix = torch.from_numpy(np.asarray(adj_matrix)).to(self.device)
            adj = (adj,adj_matrix)


        # sent_bert shape torch.Size([batch_size, 3, 768])
        gcn_output_batch = []
        'todo: make this a batch, not a for-loop; right now this is to calculate for each single document'
        for node_feature in sent_bert:


            gcn_output=self.gcn(node_feature, adj)

            'graph-level read out, summation'
            gcn_output = torch.sum(gcn_output,0)
            gcn_output_batch.append(gcn_output)

        # import pdb;
        # pdb.set_trace()

        gcn_output_batch = torch.stack(gcn_output_batch, 0)

        'GCN ends'


        return gcn_output_batch,generated_adj # (batch_size, class_number)



# class Hi_Bert_Classification_Model_GCN_tokenlevel(nn.Module):
#     """ A Model for bert fine tuning, put an lstm on top of BERT encoding """
#
#     def __init__(self,num_class,device,adj_method,pooling_method='mean'):
#         super(Hi_Bert_Classification_Model_GCN_tokenlevel, self).__init__()
#
#         self.bert_path = 'bert-base-uncased'
#         self.bert = transformers.BertModel.from_pretrained(self.bert_path)
#
#         self.lstm_layer_number = 2
#         self.lstm_hidden_size = 128
#         self.max_len = 1024
#
#         # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
#         self.device = device
#         self.pooling_method=pooling_method
#
#         self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)
#
#         'start GCN'
#         # self.gcn = GCN(input_dim=self.lstm_hidden_size,hidden_dim=32,output_dim=num_class).to(device)
#         self.gcn = GAT(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
#         self.adj_method = adj_method
#
#
#     def forward(self, ids, mask, token_type_ids):
#
#
#         batch_size = len(ids)
#
#
#
#         reshape_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
#         reshape_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
#         reshape_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
#
#         # reshape_ids = torch.stack(ids, 0).reshape(batch_size, -1).to(self.device)
#         # reshape_mask = torch.stack(mask, 0).reshape(batch_size, -1).to(self.device)
#         # reshape_token_type_ids = torch.stack(token_type_ids, 0).reshape(batch_size, -1).to(self.device)
#
#
#
#         batch_bert = []
#         for emb_pool, emb_mask, emb_token_type_ids in zip(reshape_ids, reshape_mask, reshape_token_type_ids):
#             results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
#             batch_bert.append(results[0]) # results[0] shape: (length,chunk_len, 768)
#
#
#         sent_bert = torch.stack(batch_bert, 0).reshape(batch_size,-1,768)[:,:self.max_len,:]
#
#         # import pdb;pdb.set_trace()
#         # res,not_use = self.bert(reshape_ids,attention_mask=reshape_mask, token_type_ids=reshape_token_type_ids)
#         # sent_bert shape: (batch_size, seq_len, 768)
#
#
#
#         'encode bert'
#         # bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
#         # bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
#         # bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
#         # batch_bert = []
#         # for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
#         #     results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
#         #     batch_bert.append(results[1])
#         #
#         # sent_bert = torch.stack(batch_bert, 0)
#
#
#
#         'GCN starts'
#         sent_bert = self.mapping(sent_bert)
#         node_number = sent_bert.shape[1]
#
#
#
#         'random, using networkx'
#
#         if self.adj_method == 'random':
#             generated_adj = nx.dense_gnm_random_graph(node_number, node_number)
#         elif self.adj_method == 'er':
#             generated_adj = nx.erdos_renyi_graph(node_number, node_number)
#         elif self.adj_method == 'binom':
#             generated_adj = nx.binomial_graph(node_number, p=0.5)
#         elif self.adj_method == 'path':
#             generated_adj = nx.path_graph(node_number)
#         elif self.adj_method == 'complete':
#             generated_adj = nx.complete_graph(node_number)
#         elif self.adj_method == 'kk':
#             generated_adj = kronecker_generator(node_number)
#         elif self.adj_method == 'watts':
#             if node_number-1 > 0:
#                 generated_adj = nx.watts_strogatz_graph(node_number, k=node_number-1, p=0.5)
#             else:
#                 generated_adj = nx.watts_strogatz_graph(node_number, k=node_number, p=0.5)
#         elif self.adj_method == 'ba':
#             if node_number - 1>0:
#                 generated_adj = nx.barabasi_albert_graph(node_number, m=node_number-1)
#             else:
#                 generated_adj = nx.barabasi_albert_graph(node_number, m=node_number)
#         else:
#             generated_adj = nx.dense_gnm_random_graph(node_number, node_number)
#
#
#         nx_adj = from_networkx(generated_adj)
#         adj = nx_adj['edge_index'].to(self.device)
#
#         'combine starts'
#         # generated_adj2 = nx.dense_gnm_random_graph(node_number,node_number)
#         # nx_adj = from_networkx(generated_adj)
#         # adj = nx_adj['edge_index'].to(self.device)
#         # nx_adj2 = from_networkx(generated_adj2)
#         # adj2 = nx_adj2['edge_index'].to(self.device)
#         # adj = torch.cat([adj2, adj], 1)
#         'combine ends'
#
#
#
#         if self.adj_method == 'complete':
#             'complete connected'
#             adj = torch.ones((node_number,node_number)).to_sparse().indices().to(self.device)
#
#         # sent_bert shape torch.Size([batch_size, 3, 768])
#         gcn_output_batch = []
#         for node_feature in sent_bert:
#             gcn_output=self.gcn(node_feature, adj)
#
#             'graph-level read out, summation'
#             gcn_output = torch.sum(gcn_output,0)
#             gcn_output_batch.append(gcn_output)
#
#
#
#         gcn_output_batch = torch.stack(gcn_output_batch, 0)
#
#         'GCN ends'
#
#         # import pdb;
#         # pdb.set_trace()
#         return gcn_output_batch,generated_adj # (batch_size, class_number)
