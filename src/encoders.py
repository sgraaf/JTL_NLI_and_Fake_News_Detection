#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn.pack_padded_sequence as Pack
import torch.nn.utils.rnn.pad_packed_sequence as Pad


class EmbedAttention(nn.Module):

    def __init__(self, attention_dim):
        super(EmbedAttention, self).__init__()
        self.attend_weights = nn.Linear(attention_dim, 1, bias=False)
        self.eps = 1e-7

    def forward(self, hidden_embed, sent_lens):
        attention = self.attend_weights(hidden_embed).squeeze(-1)
        out = self._masked_softmax(attention, sent_lens).unsqueeze(-1)
        return out
        
    
    def _masked_softmax(self, attention, sent_lens):
        sent_lens = sent_lens.type_as(attention.data)#.long()
        idxs = torch.arange(0, int(sent_lens[0]), out=attention.data.new(int(sent_lens[0])).long()).unsqueeze(1)
        mask = (idxs.float() < sent_lens.unsqueeze(0)).float()

        exp = torch.exp(attention) * mask
        sum_exp = exp.sum(0, keepdim=True) + self.eps
     
        return exp / sum_exp.expand_as(exp)


class BiAttentionRNN(nn.Module):

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
        doc_lens = [dim[0] for dim in batch_dims]
        sent_lens = [dim[1] for dim in batch_dims]
        sentence_embeds = []
        
        for i in range(batch.shape[0]):
            article = batch[i]
            article_len = doc_lens[i]
            
            # remove pad sentences
            article = article[:article_len, :, :]
            
            # pack the article
            article_packed = Pack(article, sent_lens, batch_first=True)
            
            # run the packed batch through the LSTM cell
            article_output_packed, _ = self.LSTM_cell(article_packed)
            
            # unpack the batch
            article_encoded, _ = Pad(article_output_packed)
            
            # compute the attention
            hidden_embed = F.tanh(self.linear(article_encoded))
            attended = self.embed_attend(hidden_embed, sent_lens) * article_encoded
            
            sentence_embeds.append(attended.sum(0, keepdim=True).squeeze(0))
            
        return torch.stack(sentence_embeds)
