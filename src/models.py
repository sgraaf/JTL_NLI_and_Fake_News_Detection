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
class HierAttNet(nn.Module):
    def __init__(self, word_input_dim, sent_input_dim, 
                 word_hidden_dim, sent_hidden_dim,
                 batch_size, num_classes, embedding,
                 max_sent_len, max_word_len):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_dim = word_hidden_dim
        self.sent_hidden_dim = sent_hidden_dim
        self.word_input_dim = word_input_dim
        self.sent_input_dim = sent_input_dim
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.word_att = WordAttentionRNN(self.input_dim, self.word_hidden_dim, self.embedding)
        self.sent_att = SentenceAttentionRNN(self.input_dim, self.sent_hidden_dim, self.num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self):
        self.word_hidden_state = torch.zeros(2, self.batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, self.batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):

        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att(output, self.sent_hidden_state)

        return output
