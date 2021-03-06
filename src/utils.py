#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import getctime
from pathlib import Path
import pickle as pkl

import matplotlib.pyplot as plt
import torch
from pandas import DataFrame as df
from pandas import read_csv
from torch.nn.utils.rnn import PackedSequence

plt.style.use('seaborn-white')

def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)


def print_flags(FLAGS):
    """
    Prints all entries in FLAGS Namespace.

    :param Namespace FLAGS: the FLAGS Namespace
    """
    FLAGS_dict = vars(FLAGS)
    longest_key_length = max(len(key) for key in FLAGS_dict)
    print('Flags:')
    for key, value in vars(FLAGS).items():
        print(f'{key:<{longest_key_length}}: {value}')


def print_model_parameters(model):
    """
    Prints all model parameters and their values.

    :param nn.Module model: the model
    """
    # print(f'Model: {model.__class__.__name__}')
    print('Model parameters:')
    named_parameters = model.named_parameters()
    longest_param_name_length = max([len(named_param[0]) for named_param in named_parameters])
    for name, param in named_parameters:
        print(f' {name:<{longest_param_name_length}}: {param}')


def print_dataset_sizes(dataset, data_percentage, name):
    """
    Prints the sizes of the dataset splits.

    :param dict dataset: the dataset
    :param float data_percentage: the percentage of the data used
    :param str name: the name of the dataset
    """
    print(f'{name} dataset size (using {data_percentage * 100:.0f}% of the data):')
    longest_set_size = len(str(max(len(dataset['train']), len(dataset['val']), len(dataset['test']))))
    print(f'Train: {len(dataset["train"]):<{longest_set_size}} samples')
    print(f'Dev:   {len(dataset["val"]):<{longest_set_size}} samples')
    print(f'Test:  {len(dataset["test"]):<{longest_set_size}} samples')
    print()

def get_number_sentences(data_dir):
    with open(data_dir, 'rb') as f:
        x = pkl.load(f)
    num_sentences_fnn = []
    for i in x['articles']: 
        num_sentences_fnn.append(len(i))
    num_sentences_fnn = sum(num_sentences_fnn)
    return num_sentences_fnn

def get_class_balance(data_dir):
    with open(data_dir, 'rb') as f:
        x = pkl.load(f)
    real = 0
    fake = 0
    for i in x['labels']: 
        if i == 0:
            real += 1
        elif i == 1:
            fake += 1
    real_ratio = real / (real + fake)
    fake_ratio = fake / (real + fake)
    return real_ratio, fake_ratio

def matrix_matmul(seq, weight, bias=None):
    features = []
    i = 1
    for feature in seq:
        i += 1
        #print(feature.size())
        #print(feature.unsqueeze(0).size())
        #print(feature)
        #print(weight.size())
        #print(weight.transpose(1,0).size())
        feature = torch.mm(feature, weight)
        if bias is not None:
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        features.append(feature)
    return torch.cat(features, 0).squeeze()


def attention_mul(rnn_output, attention_weights):
    features = []
    for feature_1, feature_2 in zip(rnn_output, attention_weights):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        features.append(feature.unsqueeze(0))
    output = torch.cat(features, 0)

    return torch.sum(output, 0).unsqueeze(0)


def create_checkpoint(checkpoints_dir, epoch, model, optimizer, results, best_accuracy):
    """
    Creates a checkpoint for the current epoch

    :param pathlib.Path checkpoints_dir: the path of the directory to store the checkpoints in
    :param int epoch: the current epoch (0-indexed)
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :param dict results: the results
    :param float best_accuracy: the best accuracy thus far
    """
    print('Creating checkpoint...', end=' ')
    epoch += 1
    checkpoint_path = checkpoints_dir / (f'{model.__class__.__name__}_{optimizer.__class__.__name__}_checkpoint_{epoch}_.pt')
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results,
            'best_accuracy': best_accuracy
        },
        checkpoint_path
    )
    print('Done!')


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads a checkpoint

    :param pathlib.Path checkpoint_path: the path of the checkpoint
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :returns: tuple of epoch, model, optimizer, results and best_accuracy of the checkpoint
    :rtype: tuple(int, nn.Module, optim.Optimizer, dict, float)
    """
    print('Loading the checkpoint...', end=' ')
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    results = checkpoint['results']
    best_accuracy = checkpoint['best_accuracy']
    print('Done!')

    return epoch, results, best_accuracy


def load_latest_checkpoint(checkpoints_dir, model, optimizer):
    """
    Loads the latest available checkpoint for the model and optimizer in question

    :param pathlib.Path checkpoints_dir: the path of the directory to load the checkpoints from
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :returns: tuple of epoch, results and best_accuracy of the checkpoint
    :rtype: tuple(int, dict, float)
    """
    print('Loading the latest checkpoint (if any exist)...', end=' ')
    checkpoints = list(checkpoints_dir.glob(
        f'{model.__class__.__name__}_{optimizer.__class__.__name__}_checkpoint_*.pt'))
    if len(checkpoints) > 0:  # there exist checkpoints for this model and optimizer!
        # determine the latest checkpoint
        checkpoints.sort(key=getctime)
        latest_checkpoint_path = checkpoints[-1]

        # load the latest checkpoint
        checkpoint = torch.load(latest_checkpoint_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        results = checkpoint['results']
        best_accuracy = checkpoint['best_accuracy']
    else:  # no checkpoints for this model and optimizer exist yet
        # initialize the epoch, results and best_accuracy
        epoch = 0
        results = {
            'epoch': [],
            'train_accuracy': [],
            'train_loss': [],
            'dev_accuracy': [],
            'dev_loss': [],
            'test_accuracy': None,
            'test_loss': None
        }
        best_accuracy = 0.0
    print('Done!')

    return epoch, results, best_accuracy


def save_model(models_dir, model):
    """
    Saves the model in the specified directory

    :param pathlib.Path models_dir: the path of the directory to save the models in
    :param nn.Module model: the model
    """
    print('Saving the model...', end=' ')
    model_path = models_dir / f'{model.__class__.__name__}_model.pt'
    torch.save(model.state_dict(), model_path)
    print('Done!')


def save_results(results_dir, results, model):
    """
    Saves the results in the specified directory

    :param pathlib.Path results_dir: the path of the directory to save the results in
    :param dict results: the results
    :param nn.Module model: the model
    """
    print('Saving the results...', end=' ')
    results_df = df.from_dict(results)
    results_path = results_dir / f'{model.__class__.__name__}_results.csv'
    results_df.to_csv(results_path, sep=';', encoding='utf-8', index=False)
    print('Done!')


def plot_results(results_dir, results, model):
    """
    Plots the results in the specified directory

    :param pathlib.Path results_dir: the path of the directory to save the plots in
    :param dict results: the results
    :param nn.Module model: the model
    """
    results_df = df.from_dict(results)

    fig, ax1 = plt.subplots()
    ax1.plot(results_df['epoch'], results_df['val_accuracy'], color='tab:orange', marker='o', label='Val accuracy')
    ax1.plot(results_df['epoch'], results_df['train_accuracy'], color='tab:red', marker='o', label='Train accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params('y')
    ax1.set_ylim([0, 1])

    ax2 = ax1.twinx()
    ax2.plot(results_df['epoch'], results_df['val_loss'], color='tab:blue', marker='o', label='Val loss')
    ax2.plot(results_df['epoch'], results_df['train_loss'], color='tab:green', marker='o', label='Train loss')
    ax2.set_ylabel('Loss')
    ax2.tick_params('y')
    # ax2.set_ylim([0, 5])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=0)
    plt.tight_layout()
    plot_path = results_dir / f'{model.__class__.__name__}_accuracy_loss_curves.png'
    plt.savefig(plot_path)
    plt.show()


def create_directories(*args):
    for dir in args:
        dir.mkdir(parents=True, exist_ok=True)