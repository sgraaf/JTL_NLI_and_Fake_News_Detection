#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import word_tokenize

from torchtext.data import Field, TabularDataset
from torchtext.datasets import SNLI
from torchtext.vocab import GloVe


def get_SNLI(text_field, label_field, percentage=None):
    """
    Returns the SNLI dataset in splits

    :param torchtext.data.Field text_field: the field that will be used for premise and hypothesis data
    :param torchtext.data.Field label_field: the field that will be used for label data
    :param float percentage: the percentage of the data to use
    :returns: the SNLI dataset in splits
    :rtype: tuple
    """
    train, dev, test = SNLI.splits(text_field, label_field)

    if percentage:
        train.examples = train.examples[:np.int(np.ceil(len(train) * percentage))]
        dev.examples = dev.examples[:np.int(np.ceil(len(dev) * percentage))]
        test.examples = test.examples[:np.int(np.ceil(len(test) * percentage))]

    return train, dev, test


def load_data(data_dir, percentage=None, only_fn=False):
    """
    Load all relevant data (GloVe vectors, SNLI & FakeNewsNet datasets) for our experiments

    :param Path data_dir: the directory in which the data is stored
    :param float percentage: the percentage of the data to use
    :returns: the data (SNLI dateset, FNN dataset, TEXT field and LABEL field)
    :rtype: tuple(Dataset, Dataset, Field, Field)
    """
    # get the GloVe vectors
    print('Loading the GloVe vectors...', end=' ')
    GloVe_vectors = GloVe()
    print('Done!')

    # set the dataset fields
    TEXT = Field(
        sequential=True,
        use_vocab=True,
        #eos_token='<eos>',
        lower=True,
        tokenize=word_tokenize,
        include_lengths=True
        #dtype=torch.cuda.FloatTensor
    )

    LABEL = Field(
        sequential=False,
        use_vocab=True,
        pad_token=None,
        unk_token=None,
        is_target=True
        #dtype=torch.cuda.float
    )

    # get the SNLI dataset in splits
    if not only_fn:
        print('Loading the SNLI dataset...', end=' ')
        SNLI = {}
        SNLI['train'], SNLI['dev'], SNLI['test'] = get_SNLI(TEXT, LABEL, percentage)
        print('Done')

    # get the FakeNewsNet dataset in splits
    print('Loading the FakeNewsNet dataset...', end=' ')
    FNN_Fields = [
        ('title', TEXT),
        ('text', TEXT),
        ('label', LABEL)
    ]
    FNN = {}
    FNN['train'], FNN['val'], FNN['test'] = TabularDataset.splits(
        path=data_dir,
        format='csv',
        fields=FNN_Fields,
        skip_header=True,
        train='train.csv', validation='val.csv', test='test.csv',
        csv_reader_params={'delimiter': ';'}
    )
    if percentage:
        FNN['train'] = FNN['train'][:np.int(np.ceil(len(FNN['train']) * percentage))]
        FNN['val'] = FNN['val'][:np.int(np.ceil(len(FNN['val']) * percentage))]
        FNN['test'] = FNN['test'][:np.int(np.ceil(len(FNN['test']) * percentage))] 
    print('Done!')

    # build the text_field vocabulary from all data splits
    # build the label_field vocabulary from the train split
    print('Building the vocabularies...', end=' ')
    
    if only_fn:
        TEXT.build_vocab(FNN['train'], FNN['val'], FNN['test'], vectors=GloVe_vectors)
        LABEL.build_vocab(FNN['train'])
        print('Done!')
        return FNN, TEXT, LABEL
    else:
        TEXT.build_vocab(SNLI['train'], SNLI['dev'], SNLI['test'], FNN['train'], FNN['val'], FNN['test'], vectors=GloVe_vectors)
        LABEL.build_vocab(SNLI['train'], FNN['train'])
        print('Done!')
        return SNLI, FNN, TEXT, LABEL

        
