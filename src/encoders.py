#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
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


class WordAttentionRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(WordAttentionRNN, self).__init__()
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
        self.embed_attend = EmbedAttention(self.hidden_dim * 2)

    def forward(self, batch, batch_dims):
        doc_lens = [len(dim) for dim in batch_dims]
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

            # pack the article
            article_packed = Pack(article_sorted, sent_lens_sorted, batch_first=True)

            # run the packed batch through the LSTM cell
            article_encoded_packed, _ = self.LSTM_cell(article_packed)

            # unpack the batch
            article_encoded, article_encodes_lens = Pad(article_encoded_packed, batch_first=True)

            # print(f'encoded article shape: {article_encoded.shape}')

            # compute the attention
            hidden_embed = torch.tanh(self.linear(article_encoded))
            attended = self.embed_attend(hidden_embed, article_encodes_lens) * article_encoded

            sent_embeds.append(attended.sum(0, keepdim=True).squeeze(0))
            
            # print(f'attended encoded article shape: {sent_embeds[-1].shape}')
        
        return sent_embeds
    
    
class SentenceAttentionRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SentenceAttentionRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # initialize the GRU cell
        self.GRU_cell = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True
        )

        # initialize the linear module
        self.linear = nn.Linear(
            in_features=2*self.hidden_dim,
            out_features=self.num_classes
        )
        if torch.cuda.is_available():
            self.to('cuda')


        # initialize the attention parameters
        mu = 0.0
        sigma = 0.05

        self.sentence_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim, 2 * self.hidden_dim)))
        self.sentence_bias = nn.Parameter(torch.zeros((1, 2 * self.hidden_dim)))
        self.context_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim, 1)))

        # initialize the Softmax activation function
        self.softmax = nn.Softmax()

    def forward(self, sentence, sentence_hidden=None):
        # run the input through the GRU cell
        sentence_output, sentence_hidden = self.GRU_cell(sentence, sentence_hidden)

        # compute the attention
        sentence_squish = matrix_matmul(sentence_output, self.sentence_weight, self.sentence_bias)
        sentence_attention = matrix_matmul(sentence_squish.unsqueeze(1), self.context_weight)
        sentence_attention_norm = self.softmax(sentence_attention)
        sentence_attn_vecs = attention_mul(sentence_output, sentence_attention_norm.unsqueeze(0).transpose(1, 0))

        # compute the final output
        output = self.linear(sentence_attn_vecs.squeeze(0))

        return output, sentence_hidden, sentence_attention_norm
