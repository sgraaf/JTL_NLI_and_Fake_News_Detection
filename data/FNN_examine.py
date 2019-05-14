import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk import tokenize
from pandas import read_csv

DATA_DIR = Path(__file__).resolve().parent
FNN_paths = sorted(DATA_DIR.rglob('FNN_*.csv'))
summary_path = DATA_DIR / 'FNN_summary.txt'

for FNN_path in FNN_paths:
    with open(summary_path, 'a', encoding='utf-8') as f:
        # read the DF
        df = read_csv(FNN_path, sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
        
        n_docs = df.shape[0]  # number of documents

        # compute article title and bodies sentence and word lengths
        title_sentences_lens = []
        title_words_lens = []
        text_sentences_lens = []
        text_words_lens = []

        for index, row in df.iterrows():
            title = row['title']
            title_sentences = tokenize.sent_tokenize(title)
            title_sentences_lens.append(len(title_sentences))

            title_words = tokenize.word_tokenize(title)
            title_words_lens.append(len(title_words))

            text = row['text']
            text_sentences = tokenize.sent_tokenize(text)
            text_sentences_lens.append(len(text_sentences))

            text_words = tokenize.word_tokenize(text)
            text_words_lens.append(len(text_words))

        title_sentences_lens = np.array(title_sentences_lens)
        title_words_lens = np.array(title_words_lens)
        text_sentences_lens = np.array(text_sentences_lens)
        text_words_lens = np.array(text_words_lens)

        # write stuff
        f.write(f'Filename: {FNN_path.name}' + '\n')
        f.write(f'Total number of documents: {n_docs}' + '\n')
        f.write(f'Total number of sentences: {title_sentences_lens.sum() + text_sentences_lens.sum()}' + '\n')
        f.write(f'Total number of words: {title_words_lens.sum() + text_words_lens.sum()}' + '\n')
        f.write('\n\n')
        f.write('Summary statistics for the articles titles:' + '\n')
        f.write('--------------------------------------------------------------------------------' + '\n')
        f.write(f'Min.   sentence count: {title_sentences_lens.min()}' + '\n')
        f.write(f'Median sentence count: {np.median(title_sentences_lens)}' + '\n')
        f.write(f'Mean   sentence count: {title_sentences_lens.mean()}' + '\n')
        f.write(f'Max.   sentence count: {title_sentences_lens.max()}' + '\n')
        f.write('\n')
        f.write(f'Min.   word count: {title_words_lens.min()}' + '\n')
        f.write(f'Median word count: {np.median(title_words_lens)}' + '\n')
        f.write(f'Mean   word count: {title_words_lens.mean()}' + '\n')
        f.write(f'Max.   word count: {title_words_lens.max()}' + '\n')
        f.write('\n\n')
        f.write('Summary statistics for the articles bodies:' + '\n')
        f.write('--------------------------------------------------------------------------------' + '\n')
        f.write(f'Min.   sentence count: {text_sentences_lens.min()}' + '\n')
        f.write(f'Median sentence count: {np.median(text_sentences_lens)}' + '\n')
        f.write(f'Mean   sentence count: {text_sentences_lens.mean()}' + '\n')
        f.write(f'Max.   sentence count: {text_sentences_lens.max()}' + '\n')
        f.write('\n')
        f.write(f'Min.   word count: {text_words_lens.min()}' + '\n')
        f.write(f'Median word count: {np.median(text_words_lens)}' + '\n')
        f.write(f'Mean   word count: {text_words_lens.mean()}' + '\n')
        f.write(f'Max.   word count: {text_words_lens.max()}' + '\n')

    # plot a histogram of the article body sentence counts
    hist, bins = np.histogram(text_sentences_lens, bins=100)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('Document length (in number of sentences)')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.tight_layout()
    plot_path = DATA_DIR / (FNN_path.stem + '_plot.png')
    plt.savefig(plot_path)
    plt.close()