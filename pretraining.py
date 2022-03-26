import logging
import os
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from models.hieformer import HieformerForMaskedLM, HieformerForSequenceClassification
from transformers import LongformerForMaskedLM, LongformerTokenizer
from models.hieformer_help import HieformerConfig


logfile = "log_pretraining"
logging.basicConfig(filename=logfile,  level=logging.INFO)
logging.info('\n')
logging.info('*'*40)


@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))

training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', 'pretraining_model_ckpt',
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '3000',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_gpu_eval_batch_size', '2',
    '--per_gpu_train_batch_size', '1',  # 32GB gpu with fp32
    '--gradient_accumulation_steps', '32',
    '--evaluate_during_training',
    '--do_train',
    '--do_eval',
])
training_args.val_datapath = './my_datasets/wikitext-103-raw/wiki.valid.raw'
training_args.train_datapath = './my_datasets/wikitext-103-raw/wiki.train.raw'

def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=tokenizer.model_max_length)
    if eval_only:
        train_dataset = val_dataset
    else:
        logging.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    block_size=tokenizer.model_max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logging.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
    
    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logging.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')


roberta_base = RobertaForMaskedLM.from_pretrained('roberta-base')
roberta_base_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
pretrain_and_evaluate(training_args, roberta_base, roberta_base_tokenizer, eval_only=True, model_path=None)


model_path = "pretrained_models/longformer-base-4096"
logging.info(f'Loading the model from {model_path}')
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = HieformerForMaskedLM.from_pretrained(model_path,return_dict=True)

logging.info(f'Pretraining based on longformer-base-{model_args.max_pos} ... ')
training_args.max_steps = 2 
pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)