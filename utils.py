##############################################################
#
# utils.py
# This file contains various functions that are applied in
# the training loops.
# They convert batch data into tensors, feed them to the models,
# compute the loss and propagate it.
#
##############################################################
import math
import torch
import networkx as nx
import numpy as np

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
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time
import timeit


def my_collate1(batches):
    # return batches

    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]
    # all_data = []
    # for batch in batches:
    #     data = {}
    #     for key, value in batch.items():
    #         import pdb;pdb.set_trace()
    #         data[key] = torch.stack(value)
    #     all_data.append(data)
    #
    # return all_data

def loss_fun(outputs, targets):
    loss = nn.CrossEntropyLoss()
    return loss(outputs, targets)
    # return nn.BCEWithLogitsLoss()(outputs, targets)


def evaluate(target, predicted):
    true_label_mask = [1 if (np.argmax(x)-target[i]) ==
                       0 else 0 for i, x in enumerate(predicted)]
    nb_prediction = len(true_label_mask)
    true_prediction = sum(true_label_mask)
    false_prediction = nb_prediction-true_prediction
    accuracy = true_prediction/nb_prediction
    return{
        "accuracy": accuracy,
        "nb exemple": len(target),
        "true_prediction": true_prediction,
        "false_prediction": false_prediction,
    }


def train_loop_fun1_orig(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses = []
    #import pdb;pdb.set_trace()
    for batch_idx, batch in enumerate(data_loader):
        #         model.half()
        #         ids_batch=[data["ids"] for data in batch]
        #         mask_batch=[data["mask"] for data in batch]
        #         token_type_ids_batch = [data["token_type_ids"] for data in batch]
        #         targets_batch = [data["targets"] for data in batch]
        #         lengt_batch=[data['len'] for data in batch]

        ids = [data["ids"] for data in batch] # size of 8
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch] # length: 8
        length = [data['len'] for data in batch] # [tensor([3]), tensor([7]), tensor([2]), tensor([4]), tensor([2]), tensor([4]), tensor([2]), tensor([3])]



        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.cat(targets)
        length = torch.cat(length)

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]



        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fun(outputs, targets)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 500 == 0:
            print(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()
    return losses

def get_graph_features(graph):
    'more https://networkx.org/documentation/stable/reference/algorithms/approximation.html'

    try:

        # import pdb;pdb.set_trace()
        node_number = nx.number_of_nodes(graph)  # int
        centrality = nx.degree_centrality(graph) # a dictionary
        centrality = sum(centrality.values())/node_number
        edge_number = nx.number_of_edges(graph) # int
        degrees = dict(graph.degree) # a dictionary
        degrees = sum(degrees.values()) /edge_number
        density = nx.density(graph) # a float
        clustring_coef = nx.average_clustering(graph) # a float Compute the average clustering coefficient for the graph G.
        closeness_centrality = nx.closeness_centrality(graph) # dict
        closeness_centrality = sum(closeness_centrality.values())/len(closeness_centrality)
        number_triangles = nx.triangles(graph) # dict
        number_triangles = sum(number_triangles.values())/len(number_triangles)
        number_clique = nx.graph_clique_number(graph) # a float Returns the number of maximal cliques in the graph.
        number_connected_components = nx.number_connected_components(graph) # int Returns the number of connected components.
        # avg_shortest_path_len = nx.average_shortest_path_length(graph) # float Return the average shortest path length; The average shortest path length is the sum of path lengths d(u,v) between all pairs of nodes (assuming the length is zero if v is not reachable from v) normalized by n*(n-1) where n is the number of nodes in G.
        # diameter = nx.distance_measures.diameter(graph) # int The diameter is the maximum eccentricity.
        return {'node_number': node_number, 'edge_number': edge_number, 'centrality': centrality, 'degrees': degrees,
                'density': density, 'clustring_coef': clustring_coef, 'closeness_centrality': closeness_centrality,
                'number_triangles': number_triangles, 'number_clique': number_clique,
                'number_connected_components': number_connected_components,
                'avg_shortest_path_len': 0, 'diameter': 0}
    except:
        return {'node_number': 1, 'edge_number': 1, 'centrality': 0, 'degrees': 0,
                'density': 0, 'clustring_coef': 0, 'closeness_centrality': 0,
                'number_triangles': 0, 'number_clique': 0,
                'number_connected_components': 0,
                'avg_shortest_path_len': 0, 'diameter': 0}


def graph_feature_stats(graph_feature_list):
    total_number = len(graph_feature_list)
    stats = {k:[] for k in graph_feature_list[0].keys()}
    for feature_dict in graph_feature_list:
        for key in stats.keys():
            stats[key].append(feature_dict[key])
    'get mean'
    stats_mean = {k:sum(v)/len(v) for (k,v) in stats.items()}
    return stats_mean

def train_loop_fun1(data_loader, model, optimizer, device, scheduler=None):
    '''optimized function for Hi-BERT'''

    model.train()
    t0 = time.time()
    losses = []
    #import pdb;pdb.set_trace()

    graph_features = []
    for batch_idx, batch in enumerate(data_loader):

        ids = [data["ids"] for data in batch] # size of 8
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch] # length: 8
        length = [data['len'] for data in batch] # [tensor([3]), tensor([7]), tensor([2]), tensor([4]), tensor([2]), tensor([4]), tensor([2]), tensor([3])]


        'cat is not working for hi-bert'
        # ids = torch.cat(ids)
        # mask = torch.cat(mask)
        # token_type_ids = torch.cat(token_type_ids)
        # targets = torch.cat(targets)
        # length = torch.cat(length)


        # ids = ids.to(device, dtype=torch.long)
        # mask = mask.to(device, dtype=torch.long)
        # token_type_ids = token_type_ids.to(device, dtype=torch.long)
        # targets = targets.to(device, dtype=torch.long)

        target_labels = torch.stack([x[0] for x in targets]).long().to(device)

        optimizer.zero_grad()

        # measure time
        start = timeit.timeit()
        outputs,adj_graph = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        end = timeit.timeit()
        model_time = end - start


        loss = loss_fun(outputs, target_labels)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 500 == 0:
            print(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()

        graph_features.append(get_graph_features(adj_graph))


    stats_mean = graph_feature_stats(graph_features)
    import pprint
    pprint.pprint(stats_mean)
    return losses


def eval_loop_fun1(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    for batch_idx, batch in enumerate(data_loader):
        ids = [data["ids"] for data in batch]  # size of 8
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]  # length: 8


        with torch.no_grad():
            target_labels = torch.stack([x[0] for x in targets]).long().to(device)
            outputs, _ = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fun(outputs, target_labels)
            losses.append(loss.item())

        fin_targets.append(target_labels.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses
#     return np.vstack(fin_outputs), np.vstack(fin_targets), losses



def eval_loop_fun1_orgi(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    for batch_idx, batch in enumerate(data_loader):

        #         model.half()
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]
        lengt = [data['len'] for data in batch]
#         ids_batch=[data["ids"] for data in batch]
#         mask_batch=[data["mask"] for data in batch]
#         token_type_ids_batch = [data["token_type_ids"] for data in batch]
#         targets_batch = [data["targets"] for data in batch]
#         lengt_batch=[data['len'] for data in batch]

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.cat(targets)
        lengt = torch.cat(lengt)

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fun(outputs, targets)
            losses.append(loss.item())

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(
            outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses
#     return np.vstack(fin_outputs), np.vstack(fin_targets), losses



def rnn_train_loop_fun1(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses = []
    for batch_idx, batch in enumerate(data_loader):
        #         model.half()
        #         ids_batch=[data["ids"] for data in batch]
        #         mask_batch=[data["mask"] for data in batch]
        #         token_type_ids_batch = [data["token_type_ids"] for data in batch]
        #         targets_batch = [data["targets"] for data in batch]
        #         lengt_batch=[data['len'] for data in batch]
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"][0] for data in batch]
        lengt = [data['len'] for data in batch]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.stack(targets)
        lengt = torch.cat(lengt)
        lengt = [x.item() for x in lengt]

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask,
                        token_type_ids=token_type_ids, lengt=lengt)
        loss = loss_fun(outputs, targets)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 640 == 0:
            print(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()
    return losses


def rnn_eval_loop_fun1(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    for batch_idx, batch in enumerate(data_loader):

        #         model.half()
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"][0] for data in batch]
        lengt = [data['len'] for data in batch]
#         ids_batch=[data["ids"] for data in batch]
#         mask_batch=[data["mask"] for data in batch]
#         token_type_ids_batch = [data["token_type_ids"] for data in batch]
#         targets_batch = [data["targets"] for data in batch]
#         lengt_batch=[data['len'] for data in batch]

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.stack(targets)
        lengt = torch.cat(lengt)
        lengt = [x.item() for x in lengt]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(ids=ids, mask=mask,
                            token_type_ids=token_type_ids, lengt=lengt)
            loss = loss_fun(outputs, targets)
            losses.append(loss.item())

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(
            outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses
#     return np.vstack(fin_outputs), np.vstack(fin_targets), losses


def kronecker_generator(node_number):
    """Return the Kron graph
    """
    n = math.ceil(math.sqrt(node_number))

    # binomial edge
    nb, pb = 10, .5
    A = np.random.binomial(nb, pb, n*n).reshape(n,n) / 10
    B = np.random.binomial(nb, pb, n*n).reshape(n,n) / 10
    A[A < 0.5] = 0
    B[B < 0.5] = 0
    prod = np.kron(A,B)
    prod[prod > 0] = 1

    # return truncated version
    prod = prod[:node_number,:node_number]

    G = nx.from_numpy_matrix(prod)

    # add noise

    return G

