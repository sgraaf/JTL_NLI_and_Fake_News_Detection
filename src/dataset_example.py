#!/usr/bin/env python3
# -*- coding: utf-8 -*-from torchtext.vocab import GloVe
from pathlib import Path
from dataset import FNNDataset, SortPadBatch
import torch.utils.data as data

FNN_path = Path().cwd().parent / 'data' / 'FNN.pkl'

# load the GloVe vectors
GloVe_vectors = GloVe()

FNN = FNNDataset(FNN_path, GloVe_vectors)


FNN_DL = data.DataLoader(
    dataset=FNN,
    batch_size=5,
    num_workers=1,
    shuffle=True,
    drop_last=True,
    collate_fn=SortPadBatch()
)

for step, batch in enumerate(FNN_DL):
    articles, article_dims, labels = batch
    for i in range(articles.shape[0]):
        article = articles[i]
        print(article.shape)
        print(article_dims[i])
        unpack = article[:article_dims[i][0], :article_dims[i][1], :]
        print(unpack.shape)
        print(unpack[:, :, 1])

    if step == 1:
            break