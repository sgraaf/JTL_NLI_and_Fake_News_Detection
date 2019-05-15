import pickle as pkl

import torch
import torch.utils.data as data

MAX_DOC_LEN = 100
MAX_SENT_LEN = 40
EMBEDDING_DIM = 300

class FNNDataset(data.Dataset):

    def __init__(self, file_path, GloVe_vectors):
        super(FNNDataset, self).__init__()
        self.GloVe_vectors = GloVe_vectors
        self.articles, self.labels = self.load_data(file_path)

    def __getitem__(self, idx):
        article = self.articles[idx]
        label = self.labels[idx]
        
        article_embed = []
        for sentence in article:
            # pad the sentence
            sentence_pad = sentence +['<pad>'] * (MAX_SENT_LEN - len(sentence))
            
            # embed the sentence
            sentence_embed = torch.stack([self.GloVe_vectors[word] if word in self.GloVe_vectors.stoi else self.GloVe_vectors[word.lower()] for word in sentence_pad])
            
            article_embed.append(sentence_embed)
        
        # pad the article
        article_embed += [torch.zeros(MAX_SENT_LEN, 300)] * (MAX_DOC_LEN - len(article_embed))

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
    