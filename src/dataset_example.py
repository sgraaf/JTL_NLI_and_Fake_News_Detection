#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import torch
import torch.utils.data as data
from torchtext.vocab import GloVe

from dataset import FNNDataset, PadSortBatch
from models import HierarchicalAttentionNet

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')

FNN_path = Path().cwd().parent / 'data' / 'FNN.pkl'

# load the GloVe vectors
GloVe_vectors = GloVe()

# load the FNN dataset
FNN = FNNDataset(FNN_path, GloVe_vectors)

# initialize the model
model = HierarchicalAttentionNet(
        input_dim=300,
        hidden_dim=100,
        num_classes=2,
        embedding=None
).to(DEVICE)

FNN_DL = data.DataLoader(
    dataset=FNN,
    batch_size=5,
    num_workers=1,
    shuffle=True,
    drop_last=True,
    collate_fn=PadSortBatch()
)

for step, batch in enumerate(FNN_DL):
    articles, article_dims, labels = batch
    
    sent_embeds = model(articles, article_dims)
    
    if step == 10:
        break
