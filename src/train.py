"""
NOTE
Due to some problems with the CUDA/GPU memory loads, 
we had to change some code in the torch library.
Namely, in the rnn.py and functional.py scripts in torch
there are some added cpu/cuda assignments. 
The code will work on both cpu and gpu. 
Functions pack_padded_sequence in rnn.py
and nll_loss in function.py 
were edited. 

Fix source: https://discuss.pytorch.org/t/error-with-lengths-in-pack-padded-sequence/35517/6
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import argparse
from os.path import getctime
from pathlib import Path
from time import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator  # , Iterator
from allennlp.modules.elmo import Elmo
import torch.utils.data as data
from torchtext.vocab import GloVe

from data import load_data
from dataset import FNNDataset, PadSortBatch, SNLIDataset, PadSortBatchSNLI
from models import HierarchicalAttentionNet
from utils import (create_directories, load_latest_checkpoint, plot_results,
                   print_dataset_sizes, print_flags, print_model_parameters,
                   save_model, save_results, create_checkpoint, get_number_sentences)

# defaults
FLAGS = None
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')

ROOT_DIR = Path.cwd().parent
LEARNING_RATE = 0.05
MAX_EPOCHS = 10
BATCH_SIZE_FN = 10
BATCH_SIZE_NLI = 100
NUM_CLASSES_FN = 2

WORD_EMBED_DIM = 300
ELMO_EMBED_DIM = 1024
WORD_HIDDEN_DIM = 100
SENT_HIDDEN_DIM = 100


MODEL_TYPE_DEFAULT = 'STL'
DATA_DIR_DEFAULT = ROOT_DIR / 'data'
CHECKPOINTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'checkpoints'
MODELS_DIR_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'results'
DATA_PERCENTAGE_DEFAULT = 1.00

ELMO_DIR = Path().cwd().parent / 'data' / 'elmo'
ELMO_OPTIONS_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
ELMO_WEIGHT_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'


def train_batch_fn(batch, model, optimizer, loss_func_fn, loss_fn_weight):
    articles, article_dims, labels = batch
    optimizer.zero_grad()
    out = model(batch=articles, batch_dims=article_dims, task='FN')
    loss = loss_func_fn(out, labels)
    batch_loss = loss.item() * BATCH_SIZE_FN
    loss = loss_fn_weight * loss
    loss.backward()
    optimizer.step()
    batch_acc = (out.argmax(dim=1).to(DEVICE) == labels.to(DEVICE)).float().mean() 
    return batch_loss, batch_acc

def train_batch_nli(batch, model, optimizer, loss_func_nli, loss_nli_weight):
    premises, hypotheses, pre_dims, hyp_dims, labels = batch
    optimizer.zero_grad()
    out = model(batch=premises, batch_dims=pre_dims, task='NLI',
                batch_hyp=hypotheses, batch_hyp_dims=hyp_dims)
    loss = loss_func_nli(out, labels)
    batch_loss = loss.item() * BATCH_SIZE_NLI
    loss = loss_nli_weight * loss    
    loss.backward()
    optimizer.step()
    batch_acc = (out.argmax(dim=1).to(DEVICE) == labels.to(DEVICE)).float().mean() 
    return batch_loss, batch_acc
    
def train_epoch_fn(train_iter, model, optimizer, loss_func_fn):
    train_loss = 0.0
    train_acc = []
    for step, batch in enumerate(train_iter):
        articles, article_dims, labels = batch
        if step % 10 == 0:
            print(f'Processed {step} FN batches')
        optimizer.zero_grad()
        out = model(batch=articles, batch_dims=article_dims, task='FN')
        loss = loss_func_fn(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * BATCH_SIZE_FN
        acc = (out.argmax(dim=1).to(DEVICE) == labels.to(DEVICE)).float().mean()
        train_acc.append(acc)
    return train_loss, train_acc

def eval_epoch_fn(val_iter, model, loss_func_fn):
    val_acc = []
    val_loss = 0.0
    for step, batch in enumerate(val_iter):
        articles, article_dims, labels = batch
        out = model(batch=articles, batch_dims=article_dims, task='FN')
        loss = loss_func_fn(out, labels)
        val_loss += loss.item() * BATCH_SIZE_FN
        acc = (out.argmax(dim=1).to(DEVICE) == labels.to(DEVICE)).float().mean()
        val_acc.append(acc)
    return val_loss, val_acc

def eval_epoch_nli(val_iter, model, loss_func_nli):
    model.eval() # turn on evaluation mode
    val_acc = []
    val_loss = 0.0
    for step, batch in enumerate(val_iter):
        premises, hypotheses, pre_dims, hyp_dims, labels = batch
        out = model(batch=premises, batch_dims=pre_dims, task='NLI',
                batch_hyp=hypotheses, batch_hyp_dims=hyp_dims)
        loss = loss_func_nli(out, labels)
        val_loss += loss.item() * BATCH_SIZE_FN
        acc = (out.argmax(dim=1).to(DEVICE) == labels.to(DEVICE)).float().mean()
        val_acc.append(acc)
    return val_loss, val_acc
    
def train():
    model_type = FLAGS.model_type
    data_dir = Path(FLAGS.data_dir)
    checkpoints_dir = Path(FLAGS.checkpoints_dir)
    models_dir = Path(FLAGS.models_dir)
    results_dir = Path(FLAGS.results_dir)
    #data_percentage = FLAGS.data_percentage


    if model_type == 'STL':
        only_fn = True
        checkpoints_dir = checkpoints_dir / 'stl'
    else:
        only_fn = False
        checkpoints_dir = checkpoints_dir / 'mtl'
    
    # check if data directory exists
    if not data_dir.exists():
        raise ValueError('Data directory does not exist')
    
    # create other directories if they do not exist
    create_directories(checkpoints_dir, models_dir, results_dir)
    
    # load the data
    print('Loading the data...')

    # get the glove and elmo embeddings
    GloVe_vectors = GloVe()
    print('Uploaded GloVe embeddings.')
    ELMo = Elmo(
            options_file=ELMO_OPTIONS_FILE, 
            weight_file=ELMO_WEIGHT_FILE,
            num_output_representations=1, 
            requires_grad=False,
            dropout=0).to(DEVICE)
    print('Uploaded Elmo embeddings.')
    # get the fnn and snli data
    FNN = {}
    FNN_DL = {}

    for path in ['train', 'val', 'test']:
        FNN[path] = FNNDataset(data_dir / ('FNN_' + path + '.pkl'), 
           GloVe_vectors, ELMo)
        FNN_DL[path] = data.DataLoader(
                dataset=FNN[path],
                batch_size=BATCH_SIZE_FN,
                num_workers=0,
                shuffle=True,
                drop_last=True,
                collate_fn=PadSortBatch())
    print('Uploaded FNN data.')
    if not only_fn:
        SNLI = {}
        SNLI_DL = {}
        for path in ['train', 'val', 'test']:
            SNLI[path] = SNLIDataset(data_dir / ('SNLI_' + path + '.pkl'), 
               GloVe_vectors, ELMo)
            SNLI_DL[path] = data.DataLoader(
                    dataset=SNLI[path],
                    batch_size=BATCH_SIZE_NLI,
                    num_workers=0,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=PadSortBatchSNLI())
        print('Uploaded SNLI data.')
        snli_train_sent_no = len(SNLI['train']) * 2
        snli_train_len = len(SNLI_DL['train'])
    fnn_train_sent_no = get_number_sentences(data_dir / 'FNN_train.pkl')
    fnn_train_len = len(FNN_DL['train'])
    # initialize the model, according to the model type
    print('Initializing the model...', end=' ')
    if model_type == 'MTL':
        NUM_CLASSES_NLI = 3
        print("Loading an MTL HAN model.")
    elif model_type == 'STL':
        NUM_CLASSES_NLI = None
        print("Loading an STL HAN model.")
    elif model_type == 'Transfer':
        print("Nothing for now.")
    if ELMO_EMBED_DIM is not None:
        input_dim = WORD_EMBED_DIM + ELMO_EMBED_DIM 
    else:
        input_dim = WORD_EMBED_DIM
    model = HierarchicalAttentionNet(input_dim=input_dim , 
                                     hidden_dim=WORD_HIDDEN_DIM, 
                                     num_classes_task_fn=NUM_CLASSES_FN, 
                                     embedding=None, 
                                     num_classes_task_nli=NUM_CLASSES_NLI, 
                                     dropout=0).to(DEVICE)
    print('Working on: ', end='')
    print(DEVICE)
    print('Done!')
    print_model_parameters(model)
    print()

    # set the criterion and optimizer
    loss_func_fn = nn.CrossEntropyLoss()
    if not only_fn:
        loss_func_nli = nn.CrossEntropyLoss()
        temperature = 2
    optimizer = optim.Adam(
                params=model.parameters(),
                lr=LEARNING_RATE)        
    
    # load the last checkpoint (if it exists)
    epoch, results, best_accuracy = load_latest_checkpoint(checkpoints_dir, model, optimizer)
    results_fn = {'epoch':[], 'train_loss':[], 'train_accuracy':[], 'val_loss': [], 'val_accuracy': []}
    results_nli = {'epoch':[], 'train_loss':[], 'train_accuracy':[], 'val_loss': [], 'val_accuracy': []}
    results = {'fn': results_fn, 'nli': results_nli}
    if epoch == 0:
        print(f'Starting training at epoch {epoch + 1}...')
    else:
        print(f'Resuming training from epoch {epoch + 1}...')

    for i in range(epoch, MAX_EPOCHS):
        print(f'Epoch {i+1:0{len(str(MAX_EPOCHS))}}/{MAX_EPOCHS}:')
        model.train()
        # one epoch of training
        if only_fn:
            train_loss_fn, train_acc_fn = train_epoch_fn(FNN_DL['train'], model, 
                                                   optimizer, loss_func_fn)
        elif model_type == 'MTL':
            model.train()

            train_loss_fn = []
            train_acc_fn = []
            loss_fn_weight_gradnorm = 1            
            loss_fn_weight_dataset = 1 - fnn_train_sent_no / (fnn_train_sent_no + snli_train_sent_no)            
            train_loss_nli = []
            train_acc_nli = []
            loss_nli_weight_gradnorm = 1            
            loss_nli_weight_dataset = 1 - snli_train_sent_no / (fnn_train_sent_no + snli_train_sent_no)            

            chance_fn = 100 * fnn_train_len / (fnn_train_len + snli_train_len)
            iterator_fnn = enumerate(FNN_DL['train'])
            iterator_snli = enumerate(SNLI_DL['train'])
            done_fnn, done_snli = False, False
            step_fnn = 0
            step_snli = 0
            while not (done_fnn and done_snli):
                if len(train_loss_fn) > 1 and len(train_loss_nli) > 1:
                    # computes loss weights based on the loss from the previous iterations
                    loss_fn_ratio = train_loss_fn[len(train_loss_fn)-1]/train_loss_fn[len(train_loss_fn)-2]
                    loss_nli_ratio = train_loss_nli[len(train_acc_nli)-1]/train_loss_nli[len(train_loss_nli)-2]
                    loss_fn_exp = math.exp(loss_fn_ratio / temperature)
                    loss_nli_exp = math.exp(loss_nli_ratio / temperature)
                    loss_fn_weight_gradnorm =  loss_fn_exp / (loss_fn_exp + loss_nli_exp)
                    loss_nli_weight_gradnorm =  loss_nli_exp / (loss_fn_exp + loss_nli_exp)
                    loss_fn_weight = math.exp(loss_fn_weight_dataset * loss_fn_weight_gradnorm) / (math.exp(loss_fn_weight_dataset * loss_fn_weight_gradnorm) + math.exp(loss_nli_weight_dataset * loss_nli_weight_gradnorm)) 
                    loss_nli_weight = math.exp(loss_nli_weight_dataset * loss_nli_weight_gradnorm) / (math.exp(loss_fn_weight_dataset * loss_fn_weight_gradnorm) + math.exp(loss_nli_weight_dataset * loss_nli_weight_gradnorm)) 
                else:
                    loss_fn_weight = loss_fn_weight_dataset
                    loss_nli_weight = loss_nli_weight_dataset
                if np.random.randint(0,100) < chance_fn:
                    try:
                        step_fnn, batch_fnn = next(iterator_fnn)
                    except StopIteration:
                        done_fnn = True
                    else:
                        batch_loss_fn, batch_acc_fn = train_batch_fn(batch_fnn, model, optimizer, loss_func_fn, loss_fn_weight)
                        train_loss_fn.append(batch_loss_fn)
                        train_acc_fn.append(batch_acc_fn)
                else:
                    try:
                        step_snli, batch_snli = next(iterator_snli)
                    except StopIteration:
                        done_snli = True
                    else:
                        batch_loss_nli, batch_acc_nli = train_batch_nli(batch_snli, model, optimizer, loss_func_nli, loss_nli_weight)
                        train_loss_nli.append(batch_loss_nli)
                        train_acc_nli.append(batch_acc_nli)
                if step_fnn % 10 == 0 and step_fnn != 0:
                    print(f'Processed {step_fnn} FNN batches.')
                    print(f'Weight for loss for NLI is {loss_nli_weight}, loss for FN is {loss_fn_weight}.')
                if step_snli % 50 == 0 and step_fnn != 0:
                    print(f'Processed {step_snli} SNLIbatches.')
                    print(f'Weight for loss for NLI is {loss_nli_weight}, loss for FN is {loss_fn_weight}.')
        
        # one epoch of eval
        model.eval()
        val_loss_fn, val_acc_fn = eval_epoch_fn(FNN_DL['val'], model, 
                                              loss_func_fn)
        if model_type == 'MTL':
            val_loss_nli, val_acc_nli = eval_epoch_nli(SNLI_DL['val'], model, 
                                              loss_func_nli)
            
        for task in ['fn','nli']:
            results[task]['epoch'].append(i)
            if task == 'fn':
                temp_train_loss = train_loss_fn / len(FNN['train'])
                temp_val_loss = val_loss_fn / len(FNN['val'])
                temp_train_acc = train_acc_fn
                temp_val_acc = val_acc_fn
            elif task == 'nli':
                temp_train_loss = train_loss_nli / len(SNLI['train'])
                temp_val_loss = val_loss_nli / len(SNLI['val'])
                temp_train_acc = train_acc_nli
                temp_val_acc = val_acc_nli
                
            results[task]['train_loss'].append(temp_train_loss)        
            results[task]['train_accuracy'].append(torch.tensor(temp_train_acc).mean().item())
            results[task]['val_loss'].append(temp_val_loss)
            results[task]['val_accuracy'].append(torch.tensor(temp_val_acc).mean().item())
            print(results)
        
        best_accuracy = torch.tensor(temp_val_acc).max().item()
        create_checkpoint(checkpoints_dir, epoch, model, optimizer, results, best_accuracy)

    # save and plot the results
    save_results(results_dir, results, model)
    save_model(models_dir, model)
    plot_results(results_dir, results, model)



def main():
    # print all flags
    print_flags(FLAGS)

    # start the timer
    start_time = time()

    # train the model
    train()

    # end the timer
    end_time = time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f'Done training in {minutes}:{seconds} minutes.')


if __name__ == '__main__':
    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE_DEFAULT,
                        help='Train mode (i.e: STL, MTL or Transfer)')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Path of directory where the data is stored')
    parser.add_argument('--checkpoints_dir', type=str, default=CHECKPOINTS_DIR_DEFAULT,
                        help='Path of directory to store / load checkpoints')
    parser.add_argument('--models_dir', type=str, default=MODELS_DIR_DEFAULT,
                        help='Path of directory to store / load models')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR_DEFAULT,
                        help='Path of directory to store results')
    parser.add_argument('--data_percentage', type=float, default=DATA_PERCENTAGE_DEFAULT,
                        help='Percentage of data to be used (for training, testing, etc.)')
    FLAGS, unparsed = parser.parse_known_args()

    main()
