import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import attention_mul, matrix_matmul


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

        # initialize the attention parameters
        mu = 0.0
        sigma = 0.05

        self.sentence_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim, 2 * self.hidden_dim)))
        self.sentence_bias = nn.Parameter(torch.zeros((1, 2 * self.hidden_dim)))
        self.context_weight = nn.Parameter(mu + sigma * torch.randn((2 * self.hidden_dim, 1)))

        # initialize the Softmax activation function
        self.softmax = nn.Softmax()

    def forward(self, input, sentence_hidden=None):
        # run the input through the GRU cell
        sentence_output, sentence_hidden = self.GRU_cell(input, sentence_hidden)

        # compute the attention
        sentence_squish = matrix_matmul(sentence_output, self.sentence_weight, self.sentence_bias)
        sentence_attention = matrix_matmul(sentence_squish, self.context_weight).transpose(1, 0)
        sentence_attention_norm = self.softmax(sentence_attention)
        sentence_attn_vecs = attention_mul(sentence_output, sentence_attention_norm.transpose(1, 0))

        # compute the final output
        output = self.linear(sentence_attn_vecs.squeeze(0))

        return output, sentence_hidden, sentence_attention_norm
