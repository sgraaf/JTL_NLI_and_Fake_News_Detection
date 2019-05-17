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
        
        sent_lens = [len(sentence) for sentence in article]
        MAX_SENT_LEN = max(sent_lens)

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
        
        return article_embed, sent_lens, label
    
    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_data(file_path):
        dataset = pkl.load(open(file_path, 'rb'))
        return dataset['articles'], dataset['labels']

    @property
    def size(self):
        return len(self.labels)


class PadSortBatch(object):
    def __call__(self, batch):
		# sort the batch
        batch_sorted_sent = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
        batch_sorted_doc = sorted(batch_sorted_sent, key=lambda x: x[0].shape[0], reverse=True)
        
        # unpack the batch
        articles_sorted, article_dims_sorted, labels_sorted = map(list, zip(*batch_sorted_doc))       

        # pad the articles with zeroes
        max_doc_len = max([article.shape[0] for article in articles_sorted])
        max_sent_len = max([article.shape[1] for article in articles_sorted])
        articles_padded = [F.pad(article, (0, 0, 0, max_sent_len - article.shape[1], 0, max_doc_len - article.shape[0])) for article in articles_sorted]

        # convert articles and labels to tensors
        articles_tensor = torch.stack(articles_padded)
        labels_tensor = torch.LongTensor(labels_sorted)
        
        return articles_tensor, article_dims_sorted, labels_tensor
