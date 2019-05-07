from torchtext import data
import nltk 
from pathlib import Path
from torchtext.data import TabularDataset

def preprocess_data(data_dir):
    """
    Preprocess data using Torchtext Fields.
    
    Args:
        data_dir: directory with train/val/test splits of data
    
    Returns:
        all_data: TabularDataset with the data splits
        x_field: Field for input text
        y_field: Field for output label
    """
    x_field = data.Field(sequential=True,
                         lower=True,
                         tokenize=nltk.word_tokenize,
                         include_lengths=True)
    y_field = data.Field(sequential=False,
                         pad_token=None,
                         unk_token=None,
                         is_target=True)
    datafields = [("title", x_field), 
                  ("text", x_field), 
                  ("label", y_field)]
    all_data = {}
    all_data["train"], all_data["val"], all_data["test"] = TabularDataset.splits(
                   path=data_dir, # the root directory where the data lies
                   train='train.csv', validation='val.csv', test='test.csv',
                   format='csv',
                   skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                   fields=datafields,
                   csv_reader_params={'delimiter': ';'})    
    
    return all_data, x_field, y_field

nltk.download('punkt') #For tokenizing words in the data
DATA_DIR = Path.cwd().parent / 'fakenewsnet_dataset'
all_data, texts, labels = preprocess_data(DATA_DIR)

# TODO: vectorize vocab through pre-trained embeddings
#texts.build_vocab(all_data["val"], vectors=word_embeddings)
#labels.build_vocab(all_data["val"])
    