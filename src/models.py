#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from WordAttentionRNN import WordAttentionRNN
from SentenceAttentionRNN import SentenceAttentionRNN

# just for sanity checks, baseline encoder from practical 1
class BaselineEncoder(nn.Module):
    def __init__(self):
        super(BaselineEncoder, self).__init__()

    def forward(self, sent, sent_l):
        output = torch.div(torch.sum(sent, dim=0), sent_l.view(-1, 1).to(torch.float))

        return output

# model combining word and sentence level attention encoders
class HierarchicalAttentionNet(nn.Module):
	# initialize the model
    def __init__(self, word_input_dim, 
                 word_hidden_dim, sent_hidden_dim,
                 batch_size, num_classes, embedding):
                 #max_sent_len, max_word_len):
        super(HierarchicalAttentionNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_dim = word_hidden_dim 
        self.sent_hidden_dim = sent_hidden_dim # 
        self.word_input_dim = word_input_dim # word embedding dimension
        self.sent_input_dim = 2 * self.word_hidden_dim # sentence input is 2*word_hidden dim (concat of 2 LSTM hidden states)
        #self.max_sent_len = max_sent_len
        #self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.embedding = embedding
        if torch.cuda.is_available():
            self.to('cuda')
            self.embedding = self.embedding.to('cuda')
        self.word_att = WordAttentionRNN(self.word_input_dim, self.word_hidden_dim)
        self.sent_att = SentenceAttentionRNN(self.sent_input_dim, self.sent_hidden_dim, self.num_classes)
        self._init_hidden_state()


    def _init_hidden_state(self):
        self.word_hidden_state = torch.zeros(2, self.batch_size, self.word_hidden_dim)
        self.sent_hidden_state = torch.zeros(2, self.batch_size, self.sent_hidden_dim)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, document):

        output_list = []
        #print(sentence.size())
        #sentence = sentence.permute(1, 0, 2)
        for sentence in document:
            #packed_sents = nn.utils.rnn.pack_padded_sequence(sentence, lens,batch_first=True)
            sentence = self.embedding(sentence)
            output, self.word_hidden_state, _ = self.word_att(sentence, self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state, _ = self.sent_att(output, self.sent_hidden_state)
        #print(output)
        return output

"""
# slightly adapted classifier layer for SNLI from practical 1
class InferenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, classes, encoder, embedding):
        super(InferenceClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = len(classes)
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.encoder = WordAttentionRNN(self.input_size, 
                                        self.hidden_size, 
                                        self.embedding)
        self.clf = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size), #Input layer
                nn.Linear(self.hidden_size, self.classes), #Softmax layer
                )

    def forward(self, premise_batch, hypothesis_batch):
        pre_s = premise_batch[0]
        pre_l = premise_batch[1]
        hyp_s = hypothesis_batch[0]
        hyp_l = hypothesis_batch[1]
        u_embed = self.embedding(pre_s)
        v_embed = self.embedding(hyp_s)
        u_encode = self.encode(u_embed, pre_l)
        v_encode = self.encode(v_embed, hyp_l)
        features = self.concat_embed(u_encode, v_encode)
        out = self.clf(features)
        return out

    def concat_embed(self, u,v):
        concat = torch.cat((u, v, (u-v).abs(), u*v), dim=1)
        return concat

    def encode(self, s, sl):
        emb = self.encoder(s, sl)
        return emb
"""