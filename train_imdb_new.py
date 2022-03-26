import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW# get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
import logging
import os
from utils import *
from Dataset_Split_Class import DatasetSplit
import argparse
import warnings
warnings.filterwarnings("ignore")
from models.hieformer_help import HieformerConfig

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


parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--dataset', type=str, default='complaints',
                    help='choose from [complaints, imdb]')
parser.add_argument('--model_path', type=str, default='./longformer-large-4096',
                    help='')
parser.add_argument('--lstm_dim', type=int, default=128, help='Hidden dim for entering graph.')
parser.add_argument('--hid_dim', type=int, default=32, help='Hidden dim for graph models.')
parser.add_argument('--sentlen', type=int, default=20, help='Sentence length.')
parser.add_argument('--epoch', type=int, default=10, help='Number of epoch.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size.')
parser.add_argument('--graph_type', type=str, default='graphsage', help='Graph encoder type: gcn, gat, graphsage, randomwalk,linear')
parser.add_argument('--adj_method', type=str, default='path_graph',
                    help='choose from [fc,dense_gnm_random_graph,erdos_renyi_graph,binomial_graph,path_graph,complete]')
parser.add_argument('--level', type=str, default='sent', help='level: sent or tok')
parser.add_argument('--exp_name', type=str, default='longformer_log', help='level: sent or tok')

args = parser.parse_args()

TRAIN_BATCH_SIZE=args.batch_size
EPOCH=args.epoch
validation_split = .2
shuffle_dataset = True
random_seed= 42
MAX_LEN = 1024
GROUP_NUM = 10
'group_num 50, simple model 82%'

lr=2e-5#1e-3


log_file = args.exp_name 

logging.basicConfig(filename=log_file,  level=logging.INFO)
logging.info('*'*40)

print('Loading Longformer tokenizer...')
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_tokenizer = transformers.LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")


def train_epoch(data_loader, model,  optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses = []
    for batch_idx, (input_ids,input_mask,input_token_type_ids,targets) in enumerate(data_loader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        input_token_type_ids = input_token_type_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        logits =  model(input_ids, attention_mask=input_mask, return_dict=True)
        loss = loss_fun(logits, targets)
        loss.backward()

        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 500 == 0:
            print(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            logging.info(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            
            t0 = time.time()
    return losses

def eval_epoch(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    for batch_idx, (input_ids,input_mask,input_token_type_ids,targets) in enumerate(data_loader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        input_token_type_ids = input_token_type_ids.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            logits =  model(
                    input_ids,
                    attention_mask=input_mask,
                    return_dict=True,
                )
      
            loss = loss_fun(logits, targets)
            losses.append(loss.item())

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(logits, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

print ('Loading data...',args.dataset)
dataset=DatasetSplit(
    tokenizer=bert_tokenizer,
    max_len=MAX_LEN,
    file_location=args.dataset)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

from utils import my_collate1
train_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=train_sampler,collate_fn=my_collate)

valid_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,collate_fn=my_collate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



# model = transformers.LongformerForSequenceClassification.from_pretrained("./longformer-base-4096-hugging")

# from longformer.longformer import Longformer
# model_path = "pretrained_models/longformer-base-4096"
# model = Longformer.from_pretrained(model_path)
# for name, param in model.named_parameters():
#     print(name)

from models.hieformer import Hieformer, HieformerForSequenceClassification
model_path = "pretrained_models/longformer-base-4096"
config = HieformerConfig.from_json_file(os.path.join(model_path, "config.json"))
config.num_labels = dataset.num_class
model = HieformerForSequenceClassification.from_pretrained(model_path,config = config)
# for name, param in model.named_parameters():
#     print(name)

model=model.to(device)
optimizer=AdamW(model.parameters(), lr=lr)
num_training_steps=int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)
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
    logging.info(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp=train_epoch(train_data_loader, model, optimizer, device)
    epoch_loss=np.mean(batches_losses_tmp)
    print ("\n ******** Running time this step..{}".format(time.time()-t0))
    logging.info("\n ******** Running time this step..{}".format(time.time()-t0))
    avg_running_time.append(time.time()-t0)
    print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    logging.info(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1=time.time()
    output, target, val_losses_tmp=eval_epoch(valid_data_loader, model, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    logging.info(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate=evaluate(target.reshape(-1), output)
    print(f"=====>\t{tmp_evaluate}")
    logging.info(f"=====>\t{tmp_evaluate}")
    val_acc.append(tmp_evaluate['accuracy'])
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)
    print("\t§§ model has been saved §§")
    logging.info("\t§§ model has been saved §§")

logging.info("\n\n$$$$ average running time per epoch (sec)..{}".format(sum(avg_running_time)/len(avg_running_time)))
print("\n\n$$$$ average running time per epoch (sec)..", sum(avg_running_time)/len(avg_running_time))
    # torch.save(model, "models/"+model_dir+"/model_epoch{epoch+1}.pt")


