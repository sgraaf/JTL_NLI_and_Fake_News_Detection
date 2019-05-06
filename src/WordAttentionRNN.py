import torch
import torch.nn as nn

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
        self.context_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim,1)))

    def forward(self, input):
        # more stuff here