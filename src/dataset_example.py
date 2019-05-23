#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

from allennlp.modules.elmo import Elmo
import torch
import torch.utils.data as data
from torchtext.vocab import GloVe

from dataset import FNNDataset, SNLIDataset, PadSortBatch
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
SNLI_path = Path().cwd().parent / 'data' / 'SNLI_val.pkl'
# load the GloVe vectors
GloVe_vectors = GloVe()

# initiate ELMo
ELMo = Elmo(
    options_file=ELMO_OPTIONS_FILE, 
    weight_file=ELMO_WEIGHT_FILE,
    num_output_representations=1, 
    requires_grad=False,
    dropout=0
).to(DEVICE)

# load the FNN dataset
FNN = FNNDataset(FNN_path, GloVe_vectors, ELMo)

FNN_DL = data.DataLoader(
    dataset=FNN,
    batch_size=5,
    num_workers=0,
    shuffle=True,
    drop_last=True,
    collate_fn=PadSortBatch()
)
from dataset import FNNDataset, SNLIDataset, PadSortBatch, PadSortBatchSNLI 
SNLI = SNLIDataset(SNLI_path, GloVe_vectors, ELMo)

SNLI_DL = data.DataLoader(
    dataset=SNLI,
    batch_size=5,
    num_workers=0,
    shuffle=True,
    drop_last=True,
    collate_fn=PadSortBatchSNLI()
)

# initialize the model
model = HierarchicalAttentionNet(
        input_dim=300+1024,
        hidden_dim=100,
        num_classes_task_fn=2,
        num_classes_task_nli=3,
        embedding=None
).to(DEVICE)

for step, batch in enumerate(FNN_DL):
    articles, article_dims, labels = batch
    print(articles.shape)
    
    out = model(batch=articles, batch_dims=article_dims, task='FN')
    
    print(f'accuracy: {(out.argmax(dim=1).to(DEVICE)== labels.to(DEVICE)).float().mean()}')
    
    if step == 10:
        break


for step, batch in enumerate(SNLI_DL):
    premises, hypotheses, pre_dims, hyp_dims, labels = batch
    out = model(batch=premises, batch_dims=pre_dims, task='NLI',
                batch_hyp=hypotheses, batch_hyp_dims=hyp_dims)
    
    print(f'accuracy: {(out.argmax(dim=1).to(DEVICE) == labels.to(DEVICE)).float().mean()}')
    
    if step == 10:
        break
    
    
