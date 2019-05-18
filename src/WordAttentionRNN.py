import torch
import torch.nn as nn


from utils import attention_mul, matrix_matmul


class WordAttentionRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes=None):
        super(WordAttentionRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #self.embedding = embedding
        if torch.cuda.is_available():
            self.to('cuda')
            #self.embedding = self.embedding.to('cuda')

        # initialize the GRU cell
        self.GRU_cell = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True
        )
        if num_classes is not None:
            self.num_classes = num_classes
            self.linear = nn.Linear(
                in_features=2*self.hidden_dim,
                out_features=self.num_classes
            )
        
        # initialize the attention parameters
        mu = 0.0
        sigma = 0.05

        self.word_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim, 2 * self.hidden_dim)))
        self.word_bias = nn.Parameter(torch.zeros((1, 2 * self.hidden_dim)))
        self.context_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim, 1)))
        
        # initialize the Softmax activation function
        self.softmax = nn.Softmax()

    def forward(self, word, word_hidden=None):
        # embed the input
        #input_emb = self.embedding(word)
        input_emb = word
        # run the embedded input through the GRU cell
        if input_emb.size() != 3:
            input_emb = input_emb.unsqueeze(1)
        word_output, word_hidden = self.GRU_cell(input_emb, word_hidden)
        # compute the attention
        word_squish = matrix_matmul(word_output, self.word_weight, self.word_bias)
        word_attention = matrix_matmul(word_squish.unsqueeze(1), self.context_weight)
        word_attention_norm = self.softmax(word_attention)
        word_attn_vecs = attention_mul(word_output, word_attention_norm.unsqueeze(0).transpose(1, 0))

        return word_attn_vecs, word_hidden, word_attention_norm
    
    def predict(self, word_vecs):
        output = self.linear(word_vecs.squeeze(0))
        return output
