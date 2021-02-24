import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
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

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)



class Model_Q1(nn.Module):
    """Some Information about Model_Q1"""
    def __init__(self, vocab_size, embedding_dim, n_class, static_embeding_weight=None):
        super(Model_Q1, self).__init__()
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



class Model_Q2(nn.Module):
    """Some Information about Model_Q1"""
    def __init__(self, vocab_size, embedding_dim, n_class, static_embeding_weight=None):
        super(Model_Q2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if static_embeding_weight is None else nn.Embedding.from_pretrained(torch.FloatTensor(static_embeding_weight)) 

        self.q = nn.Parameter(torch.rand(1, embedding_dim, requires_grad=True))
        self.cnt = torch.zeros(1)+1

        self.mlp = nn.Sequential(nn.Linear(embedding_dim, n_class))

    def forward(self, x):
        x = self.embedding(x)
        b, seqlen, dim = x.shape

        attn = torch.softmax((self.cnt+torch.mm(x.view(-1, dim), self.q.T)).view(b, seqlen, 1), dim=1)

        x = x * attn
        x = x.sum(1)

        return self.mlp(x)

class Model_Q3(nn.Module):
    """Some Information about Model_Q1"""
    def __init__(self, vocab_size, embedding_dim, n_class, static_embeding_weight=None):
        super(Model_Q3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if static_embeding_weight is None else nn.Embedding.from_pretrained(torch.FloatTensor(static_embeding_weight)) 


        self.attn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.mlp = nn.Sequential(nn.Linear(embedding_dim, n_class))

    def forward(self, x):
        _, seqlen = x.shape
        x = self.embedding(x)

        q = x.mean(1)

        q = self.attn(q)
        q.unsqueeze_(-1)

        w = torch.bmm(x, q)

        attn = F.softmax(w, dim=1)

        x = x * attn
        x = x.sum(1)

        x = self.mlp(x)
        return x