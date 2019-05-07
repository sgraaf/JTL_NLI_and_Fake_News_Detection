# imports
import argparse
from os.path import getctime
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator

from data import load_data
from SentenceAttentionRNN import SentenceAttentionRNN
from utils import (create_directories, load_latest_checkpoint, plot_results,
                   print_dataset_sizes, print_flags, print_model_parameters,
                   save_results)
from WordAttentionRNN import WordAttentionRNN

# defaults
FLAGS = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = Path.cwd().parent
LEARNING_RATE = 0.05
MAX_EPOCHS = 20
BATCH_SIZE = 64

TRAIN_MODE_DEFAULT = 'MTL'
DATA_DIR_DEFAULT = ROOT_DIR / 'data'
CHECKPOINTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'checkpoints'
MODELS_DIR_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'results'
DATA_PERCENTAGE_DEFAULT = 1.00


def train():
    model_type = FLAGS.model_type
    data_dir = Path(FLAGS.data_dir)
    checkpoints_dir = Path(FLAGS.checkpoints_dir)
    models_dir = Path(FLAGS.models_dir)
    results_dir = Path(FLAGS.results_dir)
    data_percentage = FLAGS.data_percentage

    # check if data directory exists
    if not data_dir.exists():
        raise ValueError('Data directory does not exist')
    
    # create other directories if they do not exist
    create_directories(checkpoints_dir, models_dir, results_dir)

    # load the data
    print('Loading the data...')
    SNLI, FNN, TEXT, LABEL = load_data(data_dir, data_percentage)
    embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors)
    embedding.requires_grad = False
    print()

    # print the dataset sizes
    print_dataset_sizes(SNLI, data_percentage, 'SNLI')
    print_dataset_sizes(FNN, data_percentage, 'FakeNewsNet')

    # initialize the model, according to the model type
    print('Initializing the model...', end=' ')
    # TODO: Initialize the model properly
    if model_type == 'MTL':
        # model = ... 
    elif model_type == 'STL':
        # model = ...
    elif model_type == 'Transfer':
        # model = ...
    model.to(DEVICE)
    print('Done!')
    print_model_parameters(model)
    print()

    # set the criterion and optimizer
    # TODO: Do we need a custom loss function for our model(s)?
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE
    )

    # load the last checkpoint (if it exists)
    epoch, results, best_accuracy = load_latest_checkpoint(checkpoints_dir, model, optimizer)

    if epoch == 0:
        print(f'Starting training at epoch {epoch + 1}...')
    else:
        print(f'Resuming training from epoch {epoch + 1}...')

    for i in range(epoch, MAX_EPOCHS):
        print(f'Epoch {i+1:0{len(str(MAX_EPOCHS))}}/{MAX_EPOCHS}:')

        # TODO: train the model (and evaluate it throughout)
    
    # save and plot the results
    save_results(results_dir, results, model)
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
    parser.add_argument('--train_mode', type=str, default=TRAIN_MODE_DEFAULT,
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
