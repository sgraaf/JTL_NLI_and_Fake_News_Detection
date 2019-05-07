from os import getctime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pandas import DataFrame as df
from pandas import read_csv

plt.style.use('seaborn-white')


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


def matrix_matmul(seq, weight, bias=None):
    features = []
    for feature in seq:
        feature = torch.mm(feature, weight)
        if bias:
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        features.append(feature)

    return torch.cat(feature_list, 0).squeeze()


def attention_mul(rnn_output, attention_weights):
    features = []
    for feature_1, feature_2 in zip(input1, input2):
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
    print('Loading checkpoint...', end=' ')
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
    checkpoints = list(checkpoints_dir.glob(f'{model.__class__.__name__}_{optimizer.__class__.__name__}_checkpoint_*.pt'))
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
