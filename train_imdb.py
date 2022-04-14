import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW# get_linear_schedule_with_warmup
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

import os
from utils import *
from Dataset_Sent_Class import DatasetSent
from Dataset_Split_Class import DatasetSplit
from Bert_Classification import Bert_Classification_Model, Hi_Bert_Classification_Model,Hi_Bert_Classification_Model_LSTM,Hi_Bert_Classification_Model_BERT,Hi_Bert_Classification_Model_GCN,Hi_Bert_Classification_Model_GCN_tokenlevel
from RoBERT import RoBERT_Model

from BERT_Hierarchical import BERT_Hierarchical_Model
import warnings
warnings.filterwarnings("ignore")


# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import argparse

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--dataset', type=str, default='complaints',
                    help='choose from [complaints, imdb]')
parser.add_argument('--lstm_dim', type=int, default=128, help='Hidden dim for entering graph.')
parser.add_argument('--hid_dim', type=int, default=32, help='Hidden dim for graph models.')
parser.add_argument('--sentlen', type=int, default=20, help='Sentence length.')
parser.add_argument('--epoch', type=int, default=10, help='Number of epoch.')
parser.add_argument('--max_node', type=int, default=10, help='Number of sentence in an input document.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size.')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning Rate.')
parser.add_argument('--graph_type', type=str, default='graphsage', help='Graph encoder type: gcn, gat, graphsage, randomwalk,linear')
parser.add_argument('--adj_method', type=str, default='path_graph',
                    help='choose from [fc,dense_gnm_random_graph,erdos_renyi_graph,binomial_graph,path_graph,complete]')
parser.add_argument('--level', type=str, default='sent', help='level: sent or tok')
parser.add_argument('--tf_base', type=str, default='bert', help='Transformer base model: bert, roberta')
parser.add_argument('--apply_conv',action='store_true', help='If apply a conv1d layer when aggregating tokens to sentences after '
                                                             'BERT')
parser.add_argument('--overlap',action='store_true', help='If apply the overlap BERT embedding to add more bridges.')


# parser.add_argument('--model_dir', type=str, default='complaints',
#                     help='the dir for saving models')
args = parser.parse_args()

model_dir = args.dataset
if not os.path.exists("models/"+model_dir):
    os.mkdir("models/"+model_dir)
    print ('Making model dir...')
# set result path
res_path = "models/"+model_dir+"/model_"+str(args.sentlen)+"_"+args.graph_type+'_'+args.adj_method
if not os.path.exists(res_path):
    os.mkdir(res_path)
    print ('Making res dir...')


TRAIN_BATCH_SIZE=args.batch_size
EPOCH=args.epoch
validation_split = .2
shuffle_dataset = True
# random_seed= 42
random_seed= 1024

MAX_LEN = 100000
# MAX_LEN = 1024
GROUP_NUM = 10
'group_num 50, simple model 82%'


# CHUNK_LEN=200
CHUNK_LEN=args.sentlen # sentence-level
OVERLAP_LEN = int(args.sentlen/2)
MAX_NODE = args.max_node

# lr=2e-5#1e-3

print('Loading BERT tokenizer...')
if args.tf_base.startswith('roberta'):
    bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
else:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


print ('Loading data...',args.dataset)


dataset=DatasetSplit(
    tokenizer=bert_tokenizer,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    overlap_len=OVERLAP_LEN,
    max_node=MAX_NODE,
    #max_size_dataset=MAX_SIZE_DATASET,
    # file_location='./IMDB',
    file_location=args.dataset)



#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# Creating data indices for training and validation splits:
# if shuffle_dataset :
if dataset.train_size == 0 :
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
else:
    train_indices = [_ for _ in range(0,dataset.train_size)]

    'IMDB too large..'
    val_indices = [_ for _ in range(dataset.train_size,dataset.train_size+dataset.val_size)]
    # val_indices = [_ for _ in range(0,dataset.val_size)]
    val_indices = val_indices[:int(dataset.val_size * 0.05)] + val_indices[-int(dataset.val_size * 0.05):]


#
# import pdb;
# pdb.set_trace()


# train_indices is a list of int
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

from utils import my_collate1
train_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=train_sampler,collate_fn=my_collate1)


valid_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,collate_fn=my_collate1)

print ('Model building done.')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

num_training_steps=int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)


from utils import train_loop_fun1,evaluate,eval_loop_fun1


# model=Bert_Classification_Model().to(device)
# model=Hi_Bert_Classification_Model(num_class=dataset.num_class,device=device).to(device)

# model=Hi_Bert_Classification_Model_LSTM(num_class=dataset.num_class,device=device).to(device)
# model=Hi_Bert_Classification_Model_BERT(num_class=dataset.num_class,device=device).to(device)
if args.level == 'sent':
    model=Hi_Bert_Classification_Model_GCN(args=args,num_class=dataset.num_class,device=device,adj_method=args.adj_method).to(device)
else:
    model=Hi_Bert_Classification_Model_GCN_tokenlevel(num_class=dataset.num_class,device=device,adj_method=args.adj_method).to(device)

# optimizer=AdamW(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps = 0,
                                        num_training_steps = num_training_steps)
val_losses=[]
batches_losses=[]
val_acc=[]
avg_running_time = []
for epoch in range(EPOCH):

    t0 = time.time()
    print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp=train_loop_fun1(train_data_loader, model, optimizer, device)
    epoch_loss=np.mean(batches_losses_tmp)
    print ("\n ******** Running time this step..",time.time()-t0)
    avg_running_time.append(time.time()-t0)
    print(f"\n*** avg_loss : {epoch_loss:.4f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1=time.time()
    output, target, val_losses_tmp=eval_loop_fun1(valid_data_loader, model, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.4f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate=evaluate(target.reshape(-1), output,epoch = epoch, path=res_path)
    print(f"=====>\t{tmp_evaluate}")
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)


print("\n\n$$$$ average running time per epoch (sec)..", sum(avg_running_time)/len(avg_running_time))
torch.save(model, "models/"+model_dir+"/model_"+str(args.sentlen)+"_"+args.graph_type+'_'+args.adj_method+"_"+str(args.epoch)+".pt")
print("\t§§ model has been saved §§")


# evaluate on the full test dataset
if dataset.train_size !=0 :
    # val_indices = [_ for _ in range(0, dataset.val_size)]
    val_indices = [_ for _ in range(dataset.train_size, dataset.train_size+dataset.val_size)]
    valid_sampler = SubsetRandomSampler(val_indices)
    valid_data_loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=valid_sampler, collate_fn=my_collate1)

    t1 = time.time()
    output, target, val_losses_tmp = eval_loop_fun1(valid_data_loader, model, device)
    print(f"==> Final evaluation on the full dataset : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time() - t1:.2f} sec\n")
    tmp_evaluate = evaluate(target.reshape(-1), output, final=True, path=res_path)
    print(f"=====> Final result: \t{tmp_evaluate}")

'''
50 and 50: 0.82644

'''