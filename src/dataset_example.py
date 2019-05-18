#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

from allennlp.modules.elmo import Elmo
import torch
import torch.utils.data as data
from torchtext.vocab import GloVe

from dataset import FNNDataset, PadSortBatch
from models import HierarchicalAttentionNet

ELMO_DIR = Path().cwd().parent / 'data' / 'elmo'
ELMO_OPTIONS_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
ELMO_WEIGHT_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')

FNN_path = Path().cwd().parent / 'data' / 'FNN.pkl'

# load the GloVe vectors
GloVe_vectors = GloVe()

# initiate ELMo
ELMo = Elmo(
    options_file=ELMO_OPTIONS_FILE, 
    weight_file=ELMO_WEIGHT_FILE,
    num_output_representations=1, 
    requires_grad=False,
    dropout=0
)

# load the FNN dataset
FNN = FNNDataset(FNN_path, GloVe_vectors, ELMo)

# initialize the model
model = HierarchicalAttentionNet(
        input_dim=300+1024,
        hidden_dim=100,
        num_classes=2,
        embedding=None
).to(DEVICE)

FNN_DL = data.DataLoader(
    dataset=FNN,
    batch_size=3,
    num_workers=1,
    shuffle=True,
    drop_last=True,
    collate_fn=PadSortBatch()
)

for step, batch in enumerate(FNN_DL):
    articles, article_dims, labels = batch
    print(articles.shape)
    
    out = model(articles, article_dims)
    
    print(f'accuracy: {(out.argmax(dim=1) == labels).float().mean()}')
    
    if step == 10:
        break
