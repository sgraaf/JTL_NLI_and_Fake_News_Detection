# imports
import argparse
from os.path import getctime
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator #, Iterator

from data import load_data
from models import HierarchicalAttentionNet
from SentenceAttentionRNN import SentenceAttentionRNN
from utils import (create_directories, load_latest_checkpoint, plot_results,
                   print_dataset_sizes, print_flags, print_model_parameters,
                   save_results, save_model)
from WordAttentionRNN import WordAttentionRNN

# defaults
FLAGS = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = Path.cwd().parent
LEARNING_RATE = 0.05
MAX_EPOCHS = 5
BATCH_SIZE = 1
NUM_CLASSES_FN = 2

WORD_EMBED_DIM = 300
WORD_HIDDEN_DIM = 100
SENT_HIDDEN_DIM = 100


MODEL_TYPE_DEFAULT = 'STL'
DATA_DIR_DEFAULT = ROOT_DIR / 'data'
CHECKPOINTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'checkpoints'
MODELS_DIR_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'results'
DATA_PERCENTAGE_DEFAULT = 1.00


def doc_to_sents(doc, text):
    document, sent_tok, title_tok = [], [], []
    sentence_terminals = ['.','!','?']
    if len(doc.text[0]) > 1:
        for word in doc.text[0]:
            sent_tok.append(word)
            if text.vocab.itos[word] in sentence_terminals:
                if len(sent_tok) > 1:
                    document.append(torch.tensor(sent_tok))
                    sent_tok = []
                else: 
                    sent_tok = []
        if not document:
            document.append(torch.tensor(sent_tok))
    if len(doc.title[0]) > 1:
        for word in doc.title[0]:
            title_tok.append(word)
        document.append(torch.tensor(title_tok))
    return document

def print_doc(doc, text):
    for word in doc:
        print(text.vocab.itos[word], end=' ')

def train():
    model_type = FLAGS.model_type
    data_dir = Path(FLAGS.data_dir)
    checkpoints_dir = Path(FLAGS.checkpoints_dir)
    models_dir = Path(FLAGS.models_dir)
    results_dir = Path(FLAGS.results_dir)
    data_percentage = FLAGS.data_percentage
    if model_type == 'STL':
        only_fn = True
    else:
        only_fn = False

    # check if data directory exists
    if not data_dir.exists():
        raise ValueError('Data directory does not exist')
    
    # create other directories if they do not exist
    create_directories(checkpoints_dir, models_dir, results_dir)

    # load the data
    print('Loading the data...')
    if only_fn:
        FNN, TEXT, LABEL = load_data(data_dir, percentage=None, only_fn=only_fn)
    else:
        SNLI, FNN, TEXT, LABEL = load_data(data_dir, data_percentage, only_fn=only_fn)
    embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors)
    embedding.requires_grad = False

    # print the dataset sizes
    if not only_fn:
        print_dataset_sizes(SNLI, data_percentage, 'SNLI')
    print_dataset_sizes(FNN, data_percentage, 'FakeNewsNet')

    # initialize the model, according to the model type
    print('Initializing the model...', end=' ')
    # TODO: Initialize the model properly
    if model_type == 'MTL':
        # model = ... 
        print("Nothing for now.")
    elif model_type == 'STL':
        # model = ...
        print("Loading an STL HAN model.")
        model = HierarchicalAttentionNet(word_input_dim=WORD_EMBED_DIM, 
                                 word_hidden_dim=WORD_HIDDEN_DIM, 
                                 sent_hidden_dim=SENT_HIDDEN_DIM, 
                                 batch_size=BATCH_SIZE, 
                                 num_classes=NUM_CLASSES_FN, 
                                 embedding=embedding)
    elif model_type == 'Transfer':
        # model = ...
        print("Nothing for now.")

    model.to(DEVICE)
    print('Done!')
    print_model_parameters(model)
    print()

    # set the criterion and optimizer
    # TODO: Do we need a custom loss function for our model(s)?
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE
    )

    # load the last checkpoint (if it exists)
    epoch, results, best_accuracy = load_latest_checkpoint(checkpoints_dir, model, optimizer)
    results = {'epoch':[], 'train_loss':[], 'train_accuracy':[], 'val_loss': [], 'val_accuracy': []}
    if epoch == 0:
        print(f'Starting training at epoch {epoch + 1}...')
    else:
        print(f'Resuming training from epoch {epoch + 1}...')

    for i in range(epoch, MAX_EPOCHS):
        print(f'Epoch {i+1:0{len(str(MAX_EPOCHS))}}/{MAX_EPOCHS}:')

        train_i, val_i, test_i = BucketIterator.splits(
                datasets=(FNN['train'], FNN['val'], FNN['test']),
                batch_sizes=(BATCH_SIZE,BATCH_SIZE,BATCH_SIZE),
                sort_key=lambda x: len(x.text[0]), # the BucketIterator needs to be told what function it should use to group the data.
                #sort_within_batch=False,
                shuffle=True)
        model.train()
        train_loss = 0.0
        train_acc = []
        val_loss = 0.0
        val_acc = []
        batchn = 0
        fails = 0
        failed = []
        for one_doc in train_i:
            if batchn % 1000 == 0:
                print(f'Processed {batchn} batches')
            batchn += 1
            optimizer.zero_grad()
            model._init_hidden_state()
            document = doc_to_sents(one_doc, TEXT)
            try:
                preds = model(document)
            except:
                print("couldn't process!")
                failed.append(one_doc)
                fails += 1
                continue
            loss = loss_func(preds, one_doc.label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * BATCH_SIZE
            acc = (one_doc.label == preds.argmax(dim=1)).float()
            train_acc.append(acc)
            if batchn % 200 == 0:
                break
        model.eval() # turn on evaluation mode
        val_loss = 0.0
        valn = 0
        for one_doc in val_i:
            #print_doc(one_doc.text[0],TEXT)
            #print()
            valn += 1
            document = doc_to_sents(one_doc, TEXT)
            preds = model(document)
            loss = loss_func(preds, one_doc.label)
            val_loss += loss.item() * BATCH_SIZE
            acc = (one_doc.label == preds.argmax(dim=1)).float()
            val_acc.append(acc)
            valn += 1
        results['epoch'].append(i)
        results['train_loss'].append(train_loss / len(FNN["train"]))        
        results['train_accuracy'].append(torch.tensor(train_acc).mean().item())
        results['val_loss'].append(val_loss / len(FNN["val"]))
        results['val_accuracy'].append(torch.tensor(val_acc).mean().item())
        print(results)
        
    # save and plot the results
    save_results(results_dir, results, model)
    plot_results(results_dir, results, model)
    save_model(models_dir, model)


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
