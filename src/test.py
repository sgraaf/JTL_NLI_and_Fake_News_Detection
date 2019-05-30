#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:37:52 2019

@author: azamatomu
"""

import argparse
from os.path import getctime
from pathlib import Path
from time import time
import math

import sklearn
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
CHECKPOINTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'checkpoints' / 'STL' / 'test_weighted_loss'
MODELS_DIR_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'results'
DATA_PERCENTAGE_DEFAULT = 1.00

ELMO_DIR = Path().cwd().parent / 'data' / 'elmo'
ELMO_OPTIONS_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
ELMO_WEIGHT_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

def train():
    model_type = FLAGS.model_type
    run_desc = FLAGS.run_desc
    data_dir = Path(FLAGS.data_dir)
    checkpoints_dir = Path(FLAGS.checkpoints_dir) / model_type / run_desc
    models_dir = Path(FLAGS.models_dir) / model_type / run_desc
    results_dir = Path(FLAGS.results_dir) / model_type / run_desc
    #data_percentage = FLAGS.data_percentage

    if model_type == 'STL':
        only_fn = True
    else:
        only_fn = False

    GloVe_vectors = GloVe()

    ELMo = Elmo(
            options_file=ELMO_OPTIONS_FILE, 
            weight_file=ELMO_WEIGHT_FILE,
            num_output_representations=1, 
            requires_grad=False,
            dropout=0).to(DEVICE)

    FNN_test = FNNDataset(DATA_DIR_DEFAULT / ('FNN_test.pkl'),
       GloVe_vectors, ELMo)
    FNN_DL_test = data.DataLoader(
            dataset=FNN_test,
            batch_size=BATCH_SIZE_FN,
            num_workers=0,
            shuffle=True,
            drop_last=True,
            collate_fn=PadSortBatch())

    input_dim = 300 + 1024
    NUM_CLASSES_NLI = None

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
    _, _, _ = load_latest_checkpoint(CHECKPOINTS_DIR_DEFAULT, model, optimizer)
    model.eval()
    loss_func_fn = nn.CrossEntropyLoss()
    y_pred = []
    y_true = []
    for step, batch in enumerate(FNN_DL_test):       
        articles, article_dims, labels = batch
        out = model(batch=articles, batch_dims=article_dims, task='FN')
        y_pred.append(out.argmax(dim=1).to(DEVICE).item())
        y_true.append(labels.to(DEVICE).item())
        if step % 100 == 0 and step != 0:
            print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=None))
    print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro'))
    print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='macro'))

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
    parser.add_argument('--run_desc', type=str, default=RUN_DESC_DEFAULT,
                        help='Run description, used to generate the subdirectory title')
    FLAGS, unparsed = parser.parse_known_args()

    main()


