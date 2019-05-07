import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import attention_mul, matrix_matmul

# just for sanity checks, baseline encoder from practical 1
class BaselineEncoder(nn.Module):
    def __init__(self):
        super(BaselineEncoder, self).__init__()

    def forward(self, sent, sent_l):
        output = torch.div(torch.sum(sent, dim=0), sent_l.view(-1, 1).to(torch.float))

        return output

class WordAttentionRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding):
        super(WordAttentionRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding

        # initialize the GRU cell
        self.GRU_cell = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True
        )

        # initialize the attention parameters
        mu = 0.0
        sigma = 0.05

        self.word_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim, 2 * self.hidden_dim)))
        self.word_bias = nn.Parameter(torch.zeros((1, 2 * self.hidden_dim)))
        self.context_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim, 1)))

        # initialize the Softmax activation function
        self.softmax = nn.Softmax()

    def forward(self, input, word_hidden=None):
        # embed the input
        input_emb = self.embedding(input)

        # run the embedded input through the GRU cell
        word_output, word_hidden = self.GRU_cell(input_emb, word_hidden)

        # compute the attention
        word_squish = matrix_matmul(word_output, self.word_weight, self.word_bias)
        word_attention = matrix_matmul(word_squish, self.context_weight).transpose(1, 0)
        word_attention_norm = self.softmax(word_attention)
        word_attn_vecs = attention_mul(word_output, word_attention_norm.transpose(1, 0))

        return word_attn_vecs, word_hidden, word_attention_norm

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
