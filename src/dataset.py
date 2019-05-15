import pickle as pkl

import torch
import torch.utils.data as data


class FNNDataset(data.Dataset):

    def __init__(self, file_path, GloVe_vectors):
        super(FNNDataset, self).__init__()
        self.GloVe_vectors = GloVe_vectors
        self.articles, self.labels = self.load_data(file_path)

    def __getitem__(self, idx):
        article = self.articles[idx]
        label = self.labels[idx]
        
        # determine the longest sentence in the article
        max_sent_len = max([len(sentence) for sentence in article])
        
        article_embed = []
        for sentence in article:
            # pad the sentence
            sentence_pad = sentence +['<pad>'] * (max_sent_len - len(sentence))
            
            # embed the sentence
            sentence_embed = torch.stack([self.GloVe_vectors[word] if word in self.GloVe_vectors.stoi else self.GloVe_vectors[word.lower()] for word in sentence_pad])
            
            article_embed.append(sentence_embed)
        
        # create article embedding of shape (document length, max_sent_len, embedding_dim)
        article_embed = torch.stack(article_embed)
        
        return article_embed, label

    def load_data(self, file_path):
        dataset = pkl.load(open(file_path, 'rb'))
        return dataset['articles'], dataset['labels']

    @property
    def size(self):
        return len(self.labels)
