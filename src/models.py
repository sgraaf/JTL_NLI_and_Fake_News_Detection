#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import SentAttentionRNN, DocAttentionRNN


class HierarchicalAttentionNet(nn.Module):
        # initialize the model
    def __init__(self, input_dim, hidden_dim, num_classes_task_fn, embedding, 
                 num_classes_task_nli=None, dropout=0):
        super(HierarchicalAttentionNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes_task_fn = num_classes_task_fn
        self.embedding = embedding
        self.dropout = dropout

        self.sent_attend = SentAttentionRNN(self.input_dim, self.hidden_dim)
        self.doc_attend = DocAttentionRNN(self.hidden_dim * 2, self.hidden_dim)
        self.fnn_classifier = nn.Linear(self.hidden_dim * 2, self.num_classes_task_fn)
        
        if num_classes_task_nli is not None:
            self.num_classes_task_nli = num_classes_task_nli
            self.snli_classifier = nn.Linear(self.hidden_dim * 2 * 4, 
                                             self.num_classes_task_nli)


    def forward(self, batch, batch_dims, task='FN', batch_hyp=None, batch_hyp_dims=None):
        # print(f'batch shape: {batch.shape}')
        # print(f'batch dims: {batch_dims}')
        
        # apply dropout to the batch
        batch_dropout = F.dropout(batch, p=self.dropout, training=self.training)

        if task == 'FN':
            # get the sentence embeddings
            sent_embeds = self.sent_attend(batch_dropout, batch_dims, task='FN')
            # get the document embeddings
            doc_embeds = self.doc_attend(sent_embeds, batch_dims)
        
            # get the classification
            out = self.fnn_classifier(doc_embeds)
        elif task == 'NLI':
            # get the sentence embeddings
            sent_embeds = self.sent_attend(batch_dropout, batch_dims, task='NLI')
            
            # squeeze the premises
            sent_embeds = torch.squeeze(sent_embeds)
                    
            # get the hypothesis embeddings
            batch_dropout_hyp = F.dropout(batch_hyp, p=self.dropout, training=self.training)
            sent_hyp_embeds = torch.squeeze(self.sent_attend(batch_dropout_hyp, batch_hyp_dims))
            
            # get embedding for sentence of pairs
            sent_pair_embeds = self.concat_embed(sent_embeds, sent_hyp_embeds)
            
            # get the classification
            out = self.snli_classifier(sent_pair_embeds)
        return out
    
    def concat_embed(self, u, v):
        concat = torch.cat((u, v, (u-v).abs(), u*v), dim=1)
        return concat
        
        
        
        
        # output_list = []
        # # print(sentence.size())
        # #sentence = sentence.permute(1, 0, 2)
        # for sentence in document:
        #     #packed_sents = nn.utils.rnn.pack_padded_sequence(sentence, lens,batch_first=True)
        #     sentence = self.embedding(sentence)
        #     output, self.word_hidden_state, _ = self.word_att(
        #         sentence, self.word_hidden_state)
        #     output_list.append(output)
        # output = torch.cat(output_list, 0)
        # output, self.sent_hidden_state, _ = self.sent_att(
        #     output, self.sent_hidden_state)
        # # print(output)
        # return output


# # slightly adapted classifier layer for SNLI from practical 1
# class InferenceClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, classes, encoder, embedding):
#         super(InferenceClassifier, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.classes = len(classes)
#         self.embedding = nn.Embedding.from_pretrained(embedding)
#         self.encoder = WordAttentionRNN(self.input_size, 
#                                         self.hidden_size, 
#                                         self.embedding)
#         self.clf = nn.Sequential(
#                 nn.Linear(self.input_size, self.hidden_size), #Input layer
#                 nn.Linear(self.hidden_size, self.classes), #Softmax layer
#                 )

#     def forward(self, premise_batch, hypothesis_batch):
#         pre_s = premise_batch[0]
#         pre_l = premise_batch[1]
#         hyp_s = hypothesis_batch[0]
#         hyp_l = hypothesis_batch[1]
#         u_embed = self.embedding(pre_s)
#         v_embed = self.embedding(hyp_s)
#         u_encode = self.encode(u_embed, pre_l)
#         v_encode = self.encode(v_embed, hyp_l)
#         features = self.concat_embed(u_encode, v_encode)
#         out = self.clf(features)
#         return out

#     def concat_embed(self, u,v):
#         concat = torch.cat((u, v, (u-v).abs(), u*v), dim=1)
#         return concat

#     def encode(self, s, sl):
#         emb = self.encoder(s, sl)
#         return emb
