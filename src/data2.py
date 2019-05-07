"""
IMPORTANT:
    
This script should be run once before all experiments, 
it just dumps the collection of JSON files into train/val/test splits

"""

import csv
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from pandas import DataFrame as df


def create_splits(df, ratio):
    train, test = train_test_split(df, test_size=ratio, random_state=1)
    ratio = len(test) / len(train)
    train, val = train_test_split(train, test_size=ratio, random_state=1)
    
    return train, test, val


DATA_DIR = Path.cwd().parent / 'fakenewsnet_dataset'
JSON_FILES = sorted(DATA_DIR.rglob('*.json'))

news_content = {
    'title': [],
    'text': [],
    'label': []
}

for json_file in JSON_FILES:
    if '/real/' in str(json_file):
        label = 'real'
    else:
        label = 'fake' 

    with open(json_file) as jf:
        json_dict = json.load(jf)
        title = json_dict['title'].strip()
        text = json_dict['text'].strip().replace('\n', ' ')

    if len(title) > 0 and len(text) > 0:
        news_content['title'].append(title)
        news_content['text'].append(text)
        news_content['label'].append(label)

df = df.from_dict(news_content)
df_path = DATA_DIR / 'news_content.csv'
df.to_csv(df_path, sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)

train, test, val = create_splits(df, ratio=0.1)
train.to_csv(DATA_DIR / 'train.csv', sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
test.to_csv(DATA_DIR / 'test.csv', sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
val.to_csv(DATA_DIR / 'val.csv', sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
