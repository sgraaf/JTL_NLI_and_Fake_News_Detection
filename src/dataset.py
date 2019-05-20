#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle as pkl

from allennlp.modules.elmo import batch_to_ids
import torch
import torch.nn.functional as F
import torch.utils.data as data

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
class FNNDataset(data.Dataset):

    def __init__(self, file_path, GloVe_vectors, ELMo):
        super(FNNDataset, self).__init__()
        self.GloVe = GloVe_vectors
        self.ELMo = ELMo
        self.articles, self.labels = self.load_data(file_path)

    def __getitem__(self, idx):
        article = self.articles[idx]
        #print(type(article))
        label = self.labels[idx]
        
        sent_lens = [len(sentence) for sentence in article]
        MAX_SENT_LEN = max(sent_lens)
        
        # get the GloVe embeddings
        GloVe_embed = torch.stack([torch.stack([self.GloVe[word] if word in self.GloVe.stoi else self.GloVe[word.lower()] for word in sent + ['<pad>'] * (MAX_SENT_LEN - len(sent))]) for sent in article])
        # print(GloVe_embeddings.shape)
        
        # get the ELMo embeddings
        ELMo_character_ids = batch_to_ids(article).to(DEVICE)
        ELMo_embed = self.ELMo(ELMo_character_ids)['elmo_representations'][0]
        # print(ELMo_embeddings.shape)
        
        # concat the GloVe and ELMo embeddings
        article_embed = torch.cat([GloVe_embed, ELMo_embed], dim=2)
        
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
    

class SNLIDataset(data.Dataset):

    def __init__(self, file_path, GloVe_vectors, ELMo):
        super(FNNDataset, self).__init__()
        self.GloVe = GloVe_vectors
        self.ELMo = ELMo
        self.premises, self.hypotheses, self.labels = self.load_data(file_path)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        label = self.labels[idx]
        
        # premise embedding
        # get the GloVe embeddings
        GloVe_embed = torch.stack([self.GloVe[word] if word in self.GloVe.stoi else self.GloVe[word.lower()] for word in premise])        
        # get the ELMo embeddings
        ELMo_character_ids = batch_to_ids(premise)
        ELMo_embed = self.ELMo(ELMo_character_ids)['elmo_representations'][0]      
        # concat the GloVe and ELMo embeddings
        premise_embed = torch.cat([GloVe_embed, ELMo_embed], dim=2)
        
        # hypothesis embedding
        # get the GloVe embeddings
        GloVe_embed = torch.stack([self.GloVe[word] if word in self.GloVe.stoi else self.GloVe[word.lower()] for word in hypothesis])        
        # get the ELMo embeddings
        ELMo_character_ids = batch_to_ids(hypothesis)
        ELMo_embed = self.ELMo(ELMo_character_ids)['elmo_representations'][0]      
        # concat the GloVe and ELMo embeddings
        hypothesis_embed = torch.cat([GloVe_embed, ELMo_embed], dim=2)
        
        return premise_embed, len(premise), hypothesis_embed, len(hypothesis), label
    
    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_data(file_path):
        labels_dict = {
            'neutral': 0,
            'contradiction': 1,
            'entailment': 2
        }
        
        premises = []
        hypotheses = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                json_ = json.loads(line)
                
                if json_['gold_label'] != '-':
                    labels.append(labels_dict[json_['gold_label']])
                else:
                    labels.append(labels_dict[json_['annotator_labels'][0]])

                premises.append(json_['sentence1'])
                hypotheses.append(json_['sentence2'])
                
                
        return premises, hypotheses, labels

    @property
    def size(self):
        return len(self.labels)

class SNLIDataset(data.Dataset):

    def __init__(self, file_path, GloVe_vectors, ELMo):
        super(SNLIDataset, self).__init__()
        self.GloVe = GloVe_vectors
        self.ELMo = ELMo
        self.premise, self.hypothesis, self.labels = self.load_data(file_path)

    def __getitem__(self, idx):
        premise = self.premise[idx]
        #print(premise)
        hypothesis = self.hypothesis[idx]
        #print(hypothesis)
        label = self.labels[idx]
        
        premise_lens = [len(sentence) for sentence in premise]
        hypothesis_lens = [len(sentence) for sentence in hypothesis]
        #MAX_SENT_LEN = max([max(premise_lens), max(hypothesis_lens)])
        
        # get the GloVe embeddings
        premise_GloVe_embed = torch.stack([torch.stack([self.GloVe[word] if word in self.GloVe.stoi else self.GloVe[word.lower()] for word in sent]) for sent in premise])
        hypothesis_GloVe_embed = torch.stack([torch.stack([self.GloVe[word] if word in self.GloVe.stoi else self.GloVe[word.lower()] for word in sent]) for sent in hypothesis])
        
        
        # print(GloVe_embeddings.shape)
        
        # get the ELMo embeddings
        premise_ELMo_character_ids = batch_to_ids(premise).to(DEVICE)
        premise_ELMo_embed = self.ELMo(premise_ELMo_character_ids)['elmo_representations'][0]

        hypothesis_ELMo_character_ids = batch_to_ids(hypothesis)
        hypothesis_ELMo_embed = self.ELMo(hypothesis_ELMo_character_ids)['elmo_representations'][0]
        # print(ELMo_embeddings.shape)
        
        # concat the GloVe and ELMo embeddings
        premise_embed = torch.cat([premise_GloVe_embed, premise_ELMo_embed], dim=2)
        hypothesis_embed = torch.cat([hypothesis_GloVe_embed, hypothesis_ELMo_embed], dim=2)
        
        return premise_embed, hypothesis_embed, premise_lens, hypothesis_lens, label
    
    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_data(file_path):
        dataset = pkl.load(open(file_path, 'rb'))
        return dataset['sentence1'], dataset['sentence2'], dataset['label']

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

class PadSortBatchSNLI(object):
    def __call__(self, batch):
		# sort the batch
        batch_sorted_premise = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
        
        # unpack the batch
        premise_sorted, hypothesis_sorted, premise_dims_sorted, hypothesis_dims_sorted, labels_sorted = map(list, zip(*batch_sorted_premise))       

        # pad the articles with zeroes
        max_sent_len = max([article.shape[1] for article in premise_sorted])
        premise_padded = [F.pad(article, (0, 0, 0, max_sent_len - article.shape[1], 0, 0)) for article in premise_sorted]
        hypothesis_padded = [F.pad(article, (0, 0, 0, max_sent_len - article.shape[1], 0, 0)) for article in hypothesis_sorted]

        # convert articles and labels to tensors
        premise_tensor = torch.stack(premise_padded)
        hypothesis_tensor = torch.stack(hypothesis_padded)
        
        labels_tensor = torch.LongTensor(labels_sorted)
        
        return premise_tensor, hypothesis_tensor, premise_dims_sorted, hypothesis_dims_sorted, labels_tensor 
