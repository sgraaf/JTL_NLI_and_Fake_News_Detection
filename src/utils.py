import torch
import torch.nn as nn

def matrix_matmul(seq, weight, bias=None):
    features = []
    for feature in seq:
        feature = torch.mm(feature, weight)
        if bias:
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        features.append(feature)

    return torch.cat(feature_list, 0).squeeze()


def attention_mul(rnn_output, attention_weights):
    features = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        features.append(feature.unsqueeze(0))
    output = torch.cat(features, 0)
    
    return torch.sum(output, 0).unsqueeze(0)