#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:37:52 2019

@author: azamatomu
"""

import argparse
from pathlib import Path

import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.elmo import Elmo
import torch.utils.data as data
from torchtext.vocab import GloVe

from dataset import FNNDataset, PadSortBatch, SNLIDataset, PadSortBatchSNLI
from models import HierarchicalAttentionNet
from utils import (create_directories, load_latest_checkpoint, plot_results,
                   print_dataset_sizes, print_flags, print_model_parameters,
                   save_model, save_results, create_checkpoint, get_number_sentences)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')

ROOT_DIR = Path.cwd().parent
LEARNING_RATE = 0.05
MAX_EPOCHS = 10
BATCH_SIZE_FN = 1
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

def main():
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

    # define the embeddings
    GloVe_vectors = GloVe()
    
    ELMo = Elmo(
            options_file=ELMO_OPTIONS_FILE, 
            weight_file=ELMO_WEIGHT_FILE,
            num_output_representations=1, 
            requires_grad=False,
            dropout=0).to(DEVICE)
    
    # define the dataset for SNLI
    FNN_test = FNNDataset(DATA_DIR_DEFAULT / ('FNN_test.pkl'),
       GloVe_vectors, ELMo)
    FNN_DL_test = data.DataLoader(
            dataset=FNN_test,
            batch_size=BATCH_SIZE_FN,
            num_workers=0,
            shuffle=True,
            drop_last=True,
            collate_fn=PadSortBatch())
    
    if not only_fn:
        SNLI_test = SNLIDataset(data_dir / ('SNLI_test.pkl'), 
            GloVe_vectors, ELMo)
        SNLI_DL_test = data.DataLoader(
                    dataset=SNLI_test,
                    batch_size=BATCH_SIZE_NLI,
                    num_workers=0,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=PadSortBatchSNLI())
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
    optimizer = optim.Adam(
            params=model.parameters(),
            lr=LEARNING_RATE)        
    
    #model.load_state_dict(torch.load(CHECKPOINTS_DIR_DEFAULT / 'HierarchicalAttentionNet_model.pt'))
    _, _, _ = load_latest_checkpoint(checkpoints_dir, model, optimizer)
    model.eval()
    y_pred_fn = []
    y_true_fn = []
    for step, batch in enumerate(FNN_DL_test):       
        articles, article_dims, labels = batch
        out = model(batch=articles, batch_dims=article_dims, task='FN')
        y_pred_fn.append(out.argmax(dim=1).to(DEVICE).item())
        y_true_fn.append(labels.to(DEVICE).item())
        if step % 100 == 0 and step != 0:
            print(sklearn.metrics.precision_recall_fscore_support(y_true_fn, y_pred_fn, average='micro'))
    print(sklearn.metrics.precision_recall_fscore_support(y_true_fn, y_pred_fn, average='micro'))
    print(sklearn.metrics.precision_recall_fscore_support(y_true_fn, y_pred_fn, average='macro'))        
    
    if not only_fn:
        y_pred_nli = []
        y_true_nli = []
        for step, batch in enumerate(SNLI_DL_test):       
            premises, hypotheses, pre_dims, hyp_dims, labels = batch
            out = model(batch=premises, batch_dims=pre_dims, task='NLI',
                    batch_hyp=hypotheses, batch_hyp_dims=hyp_dims)
            y_pred_nli.append(out.argmax(dim=1).to(DEVICE).item())
            y_true_nli.append(labels.to(DEVICE).item())
            if step % 100 == 0 and step != 0:
                print(sklearn.metrics.precision_recall_fscore_support(y_true_nli, y_pred_nli, average='micro'))
        print(sklearn.metrics.precision_recall_fscore_support(y_true_nli, y_pred_nli, average='micro'))
        print(sklearn.metrics.precision_recall_fscore_support(y_true_nli, y_pred_nli, average='macro'))        
            
    
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
