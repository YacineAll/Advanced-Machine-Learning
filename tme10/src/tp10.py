import logging
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from einops import rearrange, reduce, repeat


Batch = namedtuple("Batch", ["text", "labels"])


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return torch.LongTensor(self.tokenizer(s if isinstance(s, str) else s.read_text())), self.filelabels[ix] 

    @staticmethod
    def collate(batch):
        data, labels  = zip(*batch)
        return Batch(torch.nn.utils.rnn.pad_sequence(data, batch_first=True), torch.LongTensor(labels)) 

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    word2id, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    word2id = dict(zip(word2id, np.arange(0,len(word2id) )))

    OOVID = len(word2id)
    word2id["__OOV__"] = OOVID
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)





class MeanEmbedding(nn.Module):
    """Some Information about MeanEmbedding"""
    def __init__(self, vocab_size, embedding_dim, n_class, static_embeding_weight=None):
        super(MeanEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if static_embeding_weight is None else nn.Embedding.from_pretrained(torch.FloatTensor(static_embeding_weight))

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x,dim=1)
        x = self.mlp(x)
        return x



#  TODO:  Q1: Modèle d'attention propre




#  TODO:  Q2: Modèle avec des embeddings de position


#  TODO:  Q3: Utilisation du token CLS
