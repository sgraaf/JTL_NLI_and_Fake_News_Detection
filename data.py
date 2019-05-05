import json
import os

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def list_json_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if 'json' in name: 
                r.append(os.path.join(root, name))
    return r

def get_news_content(chunk=None):
    datadir = os.getcwd() + '/fakenewsnet_dataset/'
    all_files = list_files(datadir)
    all_jsons = list_json_files(datadir)

    fake_news = []
    real_news = []
    if chunk is None:
        end = len(all_jsons)
    else:
        end = chunk
    for js in all_jsons[0:end]:
        with open(js) as data:
            news = json.load(data)
            if '/real/' in js:
                real_news.append([news['title'], news['text']])
            elif '/fake/' in js:
                fake_news.append([news['title'], news['text']])

    print(len(real_news))
    print(len(fake_news))

get_news_content(1000)
