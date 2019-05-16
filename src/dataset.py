#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl

import torch
import torch.nn.functional as F
import torch.utils.data as data

# MAX_DOC_LEN = 100
# MAX_SENT_LEN = 40
EMBEDDING_DIM = 300

class FNNDataset(data.Dataset):

    def __init__(self, file_path, GloVe_vectors):
        super(FNNDataset, self).__init__()
        self.GloVe_vectors = GloVe_vectors
        self.articles, self.labels = self.load_data(file_path)

    def __getitem__(self, idx):
        article = self.articles[idx]
        label = self.labels[idx]
        
        MAX_SENT_LEN = max([len(sentence) for sentence in article])

        article_embed = []
        for sentence in article:
            # pad the sentence
            sentence_pad = sentence +['<pad>'] * (MAX_SENT_LEN - len(sentence))
            
            # embed the sentence
            sentence_embed = torch.stack([self.GloVe_vectors[word] if word in self.GloVe_vectors.stoi else self.GloVe_vectors[word.lower()] for word in sentence_pad])
            
            article_embed.append(sentence_embed)
        
        # pad the article
        # article_embed += [torch.zeros(MAX_SENT_LEN, 300)] * (MAX_DOC_LEN - len(article_embed))

        # create article embedding of shape (MAX_DOC_LEN, MAX_SENT_LEN, embedding_dim)
        article_embed = torch.stack(article_embed)
        
        return article_embed, label
    
    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_data(file_path):
        dataset = pkl.load(open(file_path, 'rb'))
        return dataset['articles'], dataset['labels']

    @property
    def size(self):
        return len(self.labels)


def sort_pad_batch(batch):
    articles, labels = list(zip(*batch))
    
    # convert articles and labels to list
    articles = list(articles)
    labels = list(labels)

    # sort the articles in reverse order of document length and sentence length
    articles.sort(reverse=True, key=lambda article: article.shape[1])
    articles.sort(reverse=True, key=lambda article: article.shape[0])

    # pad the articles with zeroes
    max_doc_len = max([article.shape[0] for article in articles])
    max_sent_len = max([article.shape[1] for article in articles])
    articles_padded = [F.pad(article, (0, 0, 0, max_sent_len - article.shape[1], 0, max_doc_len - article.shape[0])) for article in articles]


class SortPadBatch(object):
    def __call__(self, batch):
		# batch is a tuple (articles, labels).
        articles, labels = list(zip(*batch))
#        print(articles)
#        print(type(articles))
#        print(labels)
#        print(type(labels))
        
        articles = list(articles)
        labels = list(labels)

        # sort the articles in reverse order of document length and sentence length
        articles.sort(reverse=True, key=lambda article: article.shape[1])
        articles.sort(reverse=True, key=lambda article: article.shape[0])

        # determine article_dims
        article_dims = [tuple(article.shape[:2]) for article in articles]

        # pad the articles with zeroes
        max_doc_len = max([article.shape[0] for article in articles])
        max_sent_len = max([article.shape[1] for article in articles])
        articles_padded = torch.stack([F.pad(article, (0, 0, 0, max_sent_len - article.shape[1], 0, max_doc_len - article.shape[0])) for article in articles])

        # convert the labels to torch.LongTensor
        labels_tensor = torch.LongTensor(labels)
        
        return articles_padded, article_dims, labels_tensor
