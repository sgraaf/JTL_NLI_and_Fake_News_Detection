#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle as pkl

DATA_DIR = Path.cwd().parent / 'data'
FNN_path = DATA_DIR / 'FNN.pkl'


def create_splits(df, ratio):
    train, test = train_test_split(df, test_size=ratio, random_state=1)
    ratio = len(test) / len(train)
    train, val = train_test_split(train, test_size=ratio, random_state=1)

    return train, test, val


# read the pickle and create DF
with open(FNN_path, 'rb') as f:
    x = pkl.load(f)
df = pd.DataFrame.from_dict(x)

# create the splits
train, test, val = create_splits(df, ratio=0.1)    
train, test, val = train.to_dict('list'), test.to_dict('list'), val.to_dict('list') #convert them back to dict

#pickle the split
pkl.dump(train, open(DATA_DIR / 'FNN_train.pkl', 'wb'))
pkl.dump(test, open(DATA_DIR / 'FNN_test.pkl', 'wb'))
pkl.dump(val, open(DATA_DIR / 'FNN_val.pkl', 'wb')) 

