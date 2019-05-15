from torchtext.vocab import GloVe
from pathlib import Path
from dataset import FNNDataset
import torch.utils.data as data

FNN_path = Path().cwd().parent / 'data' / 'FNN.pkl'

# load the GloVe vectors
GloVe_vectors = GloVe()

FNN = FNNDataset(FNN_path, GloVe_vectors)


FNN_FL = data.DataLoader(
    dataset=FNN,
    batch_size=64,
    num_workers=1,
    shuffle=True,
    drop_last=True
)

for step, batch in enumerate(FNN_FL):
    articles, labels = batch
    print(articles.shape)
    print(len(labels))
    if step == 10:
            break