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