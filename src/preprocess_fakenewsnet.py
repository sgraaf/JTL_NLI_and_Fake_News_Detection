"""
IMPORTANT:
    
This script should be run once before all experiments, 
it just dumps the collection of JSON files into train/val/test splits

"""

import csv
import json
import re
from pathlib import Path

from pandas import DataFrame as df
from sklearn.model_selection import train_test_split


def create_splits(df, ratio):
    train, test = train_test_split(df, test_size=ratio, random_state=1)
    ratio = len(test) / len(train)
    train, val = train_test_split(train, test_size=ratio, random_state=1)
    
    return train, test, val


def is_valid_article(title, text, ILLEGAL_PATTERNS):
    # check the title
    for illegal_pattern in ILLEGAL_PATTERNS['title']:
        if bool(re.search(illegal_pattern, title)):
            return False
    for illegal_pattern in ILLEGAL_PATTERNS['text']:
        if bool(re.search(illegal_pattern, text)):
            return False

    return True

DATA_DIR = Path.cwd().parent / 'fakenewsnet_dataset'
JSON_FILES = sorted(DATA_DIR.rglob('*.json'))
ILLEGAL_PATTERNS = {
    'title': [
        'X17 Online',
        'Robot Check',
        'Entertainment, News, and Lifestyle for Moms',
        'NPR Choice page',
        'Consent Form',
        'LexisNexis(R) Publisher',
        'FindArticles.com',
        'CQ.com',
        'Notice: Data not available: U.S. Bureau of Labor Statistics',
        'Redirect Notice',
        'Resource Not Available',
        'Embed  Copy and paste this code into your blog to embed it on your site',
        'The Colbert Report'
    ],
    'text': [
        'Once registered, you can',
        'U.S. CA U.K. AU Asia DE FR  E! Is Everywhere This content is available customized for our international audience',
        'IMDb.com, Inc. takes no responsibility for the content or accuracy of the above',
        'About Trendolizer™  Trendolizer™ (patent pending) automatically scans the internet for trending content',
        'Please enable cookies on your web browser in order to continue.',
        'About Your Privacy on this Site  Welcome!',
        'version=""1.0"" encoding=""utf-8""?',
        'We and our partners use cookies on this site to improve our service',
        'This website uses profiling (non technical) cookies',
        'Screen Rant – Privacy Policy  We respect your privacy and we are committed to safeguarding your privacy while online at our site',
        'We use cookies to ensure that we give you the best experience on our website',
        'Cookies help us deliver our Services',
        'Netflix uses cookies for personalisation, to customise its online advertisements, and for other purposes',
        'Wir verwenden Cookies, um Inhalte zu personalisieren',
        'H&M använder cookies för att ge dig den bästa',
        'We respect your privacy and we are committed to safeguarding your privacy while online at our site. The following discloses the information gathering and dissemination practices for this Web site.  This Privacy Policy',
        'You must set your browser to accept cookies',
        'We use cookies to enhance your experience, for analytics and',
        'You are using an older browser version',
        'The interactive transcript could not be loaded',
        'Support the kind of journalism done by the NewsHour',
        'Watch Queue Queue  Watch Queue Queue Remove',
        'Interest Successfully Added  ',
        "We're sorry  We are unable to locate information at"
    ]
}

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
        if is_valid_article(title, text, ILLEGAL_PATTERNS):
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
