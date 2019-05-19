#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as Pack
from torch.nn.utils.rnn import pad_packed_sequence as Pad


class EmbedAttention(nn.Module):

    def __init__(self, attention_dim):
        super(EmbedAttention, self).__init__()
        self.attend_weights = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_embed, sent_lens):
        # compute the attention
        attention = self.attend_weights(hidden_embed).squeeze(-1)
        
        # perform masked softmax
        out = self._masked_softmax(attention, sent_lens).unsqueeze(-1)
        return out

    def _masked_softmax(self, attention, sent_lens):
        # create the mask
        mask = torch.arange(attention.shape[1])[None, :] < sent_lens[:, None]
        
        # mask the attention
        attention[~mask] = -float('inf')
        
        # perform softmax
        masked_softmax = torch.softmax(attention, dim=1)

        return masked_softmax


class SentAttentionRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(SentAttentionRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # initialize the GRU cell
        self.LSTM_cell = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # initialize the attention parameters
        self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.attend_weights = nn.Linear(self.hidden_dim * 2, 1, bias=False)

    def forward(self, batch, batch_dims):
        doc_lens = [len(dim) for dim in batch_dims]
        max_doc_len = max(doc_lens)
        sents_lens = batch_dims
        sent_embeds = []

        # print(f'doc lens: {doc_lens}')
        # print(f'sents lens: {sents_lens}')

        for i in range(batch.shape[0]):
            article_len = doc_lens[i]
            sent_lens = torch.LongTensor(sents_lens[i])

            # remove pad sentences
            article = batch[i][:article_len]
            
            # sort
            sent_lens_sorted, sort_idxs = torch.sort(sent_lens, dim=0, descending=True)
            article_sorted = article[sort_idxs, :]

            # print(f'unpadded article shape: {article.shape}')

            # pack the article (batch)
            article_packed = Pack(article_sorted, sent_lens_sorted, batch_first=True)

            # run the packed article (batch) through the LSTM cell
            article_encoded_packed, _ = self.LSTM_cell(article_packed)

            # unpack the article (batch)
            article_encoded, article_encoded_lens = Pad(article_encoded_packed, batch_first=True)

            # print(f'encoded article shape: {article_encoded.shape}')

            # compute the attention
            hidden_embed = torch.tanh(self.linear(article_encoded))
            attention = self.attend_weights(hidden_embed).squeeze(-1)
            mask = torch.arange(attention.shape[1])[None, :] < article_encoded_lens[:, None]  # create the mask
            attention[~mask] = -float('inf')  # mask the attention
            masked_softmax = torch.softmax(attention, dim=1).unsqueeze(-1)  # perform softmax
            attended = masked_softmax * article_encoded
            
            # sum to obtain sentence representations
            attended_sum = attended.sum(1, keepdim=True).squeeze(1)
            
            # pad to stack
            attended_pad = F.pad(attended_sum, (0, 0, 0, max_doc_len - attended_sum.shape[0]))

            sent_embeds.append(attended_pad)
            
            # print(f'attended encoded article shape: {sent_embeds[-1].shape}')
        
        return torch.stack(sent_embeds)
    
    
class DocAttentionRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(DocAttentionRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # initialize the GRU cell
        self.LSTM_cell = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # initialize the attention parameters
        self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.attend_weights = nn.Linear(self.hidden_dim * 2, 1, bias=False)

    def forward(self, batch, batch_dims):
        doc_lens = torch.LongTensor([len(dim) for dim in batch_dims])
        
        # sort
        doc_lens_sorted, sort_idxs = torch.sort(doc_lens, dim=0, descending=True)
        batch_sorted = batch[sort_idxs, :]

        # pack the batch
        batch_packed = Pack(batch_sorted, doc_lens_sorted, batch_first=True)

        # run the packed batch through the LSTM cell
        batch_encoded_packed, _ = self.LSTM_cell(batch_packed)

        # unpack the batch
        batch_encoded, batch_encoded_lens = Pad(batch_encoded_packed, batch_first=True)


        # compute the attention
        hidden_embed = torch.tanh(self.linear(batch_encoded))
        attention = self.attend_weights(hidden_embed).squeeze(-1)
        mask = torch.arange(attention.shape[1])[None, :] < batch_encoded_lens[:, None]  # create the mask
        attention[~mask] = -float('inf')  # mask the attention
        masked_softmax = torch.softmax(attention, dim=1).unsqueeze(-1)  # perform softmax
        attended = masked_softmax * batch_encoded
        
        # sum to obtain doc representations
        attended_sum = attended.sum(1, keepdim=True).squeeze(1)
        
        return attended_sum
