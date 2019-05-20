#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMPORTANT:
    
This script should be run once before all experiments, 
it just dumps the collection of JSON files into train/val/test splits

"""

import jsonlines
import re
from pathlib import Path

from nltk import tokenize
import pickle as pkl


MAX_DOC_LEN = 100
MAX_SENT_LEN = 40

SNLI_DIR = Path.cwd().parent / 'snli_1.0'
DATA_DIR = Path.cwd()

JSON_FILES = sorted(SNLI_DIR.rglob('*.jsonl'))


import copy
dataset = {
    'sentence1': [],
    'sentence2': [],
    'label': []
}

final_snli = {'val': copy.deepcopy(dataset),
              'test': copy.deepcopy(dataset),
              'train': copy.deepcopy(dataset)}
splits = ['val', 'test', 'train']
n = 0
f = 0
iter = 0
for json_file in JSON_FILES:
    #temp_dataset = final_snli[splits[iter]]#.copy()
    with jsonlines.open(json_file) as jf:
        for line in jf:
            n += 1
            #print(line)
            premise = [tokenize.word_tokenize(line["sentence1"].strip().lower())]
            hypothesis = [tokenize.word_tokenize(line["sentence2"].strip().lower())]
            if line["gold_label"] == 'contradiction':
                label = 0
            elif line["gold_label"] == 'neutral':
                label = 1
            elif line["gold_label"] == 'entailment':
                label = 2
            else:
                f += 1
                continue
            #print(line)
            if len(sentence1[0]) <= MAX_SENT_LEN and len(sentence2[0]) <= MAX_SENT_LEN:
                final_snli[splits[iter]]['premise'].append(sentence1)
                final_snli[splits[iter]]['hypothesis'].append(sentence2)
                final_snli[splits[iter]]['label'].append(label)
    #final_snli[splits[iter]] = temp_dataset.copy()
    iter += 1 
     

# pickle the dataset
pkl.dump(final_snli["train"], open(DATA_DIR / 'SNLI_train.pkl', 'wb'))
pkl.dump(final_snli["test"], open(DATA_DIR / 'SNLI_test.pkl', 'wb'))
pkl.dump(final_snli["val"], open(DATA_DIR / 'SNLI_val.pkl', 'wb')) 
