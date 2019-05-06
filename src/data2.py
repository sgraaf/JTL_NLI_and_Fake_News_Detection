import csv
import json
from pathlib import Path

from pandas import DataFrame as df

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
