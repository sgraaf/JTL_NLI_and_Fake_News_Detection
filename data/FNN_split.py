import csv
from pathlib import Path

from pandas import read_csv
from sklearn.model_selection import train_test_split

DATA_DIR = Path.cwd().parent / 'data'
FNN_path = DATA_DIR / 'FNN_clipped.csv'


def create_splits(df, ratio):
    train, test = train_test_split(df, test_size=ratio, random_state=1)
    ratio = len(test) / len(train)
    train, val = train_test_split(train, test_size=ratio, random_state=1)

    return train, test, val


# read the DF
df = read_csv(FNN_path, sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

# create the splits
train, test, val = create_splits(df, ratio=0.1)

# save the splits
train.to_csv(DATA_DIR / 'train.csv', sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
test.to_csv(DATA_DIR / 'test.csv', sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
val.to_csv(DATA_DIR / 'val.csv', sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
