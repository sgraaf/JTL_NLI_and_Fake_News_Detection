"""
IMPORTANT:
    
This script should be run once before all experiments, 
it just dumps the collection of JSON files into train/val/test splits

"""

import json
import re
from pathlib import Path

from nltk import tokenize
import pickle as pkl

def is_valid_article(title, text, ILLEGAL_PATTERNS):
    # check the title
    for illegal_pattern in ILLEGAL_PATTERNS['title']:
        if bool(re.search(illegal_pattern, title)):
            return False
    # check the body
    for illegal_pattern in ILLEGAL_PATTERNS['text']:
        if bool(re.search(illegal_pattern, text)):
            return False

    return True

ILLEGAL_PATTERNS = {
    'title': [
        "CQ.com",
        "Consent Form",
        "Embed  Copy and paste this code into your blog to embed it on your site",
        "Entertainment, News, and Lifestyle for Moms",
        "FindArticles.com",
        "LexisNexis(R) Publisher",
        "NPR Choice page",
        "Notice: Data not available: U.S. Bureau of Labor Statistics",
        "Redirect Notice",
        "Resource Not Available",
        "Robot Check",
        "The Colbert Report",
        "X17 Online"
],
    'text': [
        "Once registered, you can",
        "U.S. CA U.K. AU Asia DE FR  E! Is Everywhere This content is available customized for our international audience",
        "IMDb.com, Inc. takes no responsibility for the content or accuracy of the above",
        "About Trendolizer™  Trendolizer™ (patent pending) automatically scans the internet for trending content",
        "Please enable cookies on your web browser in order to continue.",
        "About Your Privacy on this Site  Welcome!",
        "version=\"\"1.0\"\" encoding=\"\"utf-8\"\"?",
        "We and our partners use cookies on this site to improve our service",
        "This website uses profiling (non technical) cookies",
        "Screen Rant – Privacy Policy  We respect your privacy and we are committed to safeguarding your privacy while online at our site",
        "We use cookies to ensure that we give you the best experience on our website",
        "Cookies help us deliver our Services",
        "Netflix uses cookies for personalisation, to customise its online advertisements, and for other purposes",
        "Wir verwenden Cookies, um Inhalte zu personalisieren",
        "H&M använder cookies för att ge dig den bästa",
        "We respect your privacy and we are committed to safeguarding your privacy while online at our site. The following discloses the information gathering and dissemination practices for this Web site.  This Privacy Policy",
        "You must set your browser to accept cookies",
        "We use cookies to enhance your experience, for analytics and",
        "You are using an older browser version",
        "The interactive transcript could not be loaded",
        "Support the kind of journalism done by the NewsHour",
        "Watch Queue Queue  Watch Queue Queue Remove",
        "Interest Successfully Added  ",
        "We're sorry  We are unable to locate information at"
    ]
}

MAX_DOC_LEN = 100
MAX_SENT_LEN = 40

FNN_DIR = Path(__file__).resolve().parent.parent / 'fakenewsnet_dataset'
DATA_DIR = Path(__file__).resolve().parent
JSON_FILES = sorted(FNN_DIR.rglob('*.json'))

news_content = {
    'title': [],
    'text': [],
    'label': []
}

dataset = {
    'articles': [],
    'labels': []
}

for json_file in JSON_FILES:
    if '/real/' in str(json_file):
        label = 0
    else:
        label = 1 

    with open(json_file) as jf:
        json_dict = json.load(jf)
        title = json_dict['title'].strip()
        text = json_dict['text'].strip()

    if len(title) > 0 and len(text) > 0:  # No empty article titles or bodies
        if is_valid_article(title, text, ILLEGAL_PATTERNS):  # No illegal patterns in article titles or bodies
            if len(tokenize.word_tokenize(text)) > 1:  # No single-word article bodies
                # tokenize the article
                article_sentences = tokenize.sent_tokenize(title) + tokenize.sent_tokenize(text)

                if len(article_sentences) <= MAX_DOC_LEN:  # No articles with more than 100 sentences
                    article_sentences_words = [tokenize.word_tokenize(sentence) for sentence in article_sentences if len(tokenize.word_tokenize(sentence)) <= MAX_SENT_LEN]

                    # append the article and label
                    dataset['articles'].append(article_sentences_words)
                    dataset['labels'].append(label)

# pickle the dataset
dataset_path = DATA_DIR / 'FNN.pkl'
pkl.dump(dataset, open(dataset_path, 'wb'))
