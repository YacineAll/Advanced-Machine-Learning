import csv
import numpy as np
import logging
import time
import string
from itertools import chain


import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from textloader import *
from generate import *
import logging
logging.basicConfig(level=logging.INFO)


def maskedCrossEntropy(output, y, m):
    logits = output.view(-1, output.shape[-1])
    target = y.view(-1)
    return (F.cross_entropy(logits, target, reduction='none').view(len(y), -1)*m).sum()/m.sum()

class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    def forward(self, logit, target, mask):
        loss = F.nll_loss(logit, target, reduction='none')
        loss *= mask
        return loss.sum() / mask.sum()

class rnn(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
       
        self.hidden = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.output = nn.Linear(embedding_dim + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        x = self.embeddings(x).squeeze(1)
        input_combined = torch.cat((x, h), 1)
        hidden = self.hidden(input_combined)
        output = self.output(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, n):
        return torch.zeros(n, self.hidden_size)


class RNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        
        self.x = nn.Linear(embedding_dim, self.hidden_size)
        self.h = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h):
        x = self.embeddings(x)
        x = self.x(x)
        h = self.h(h)
        return h,

    def decode(self, ht):
        return self.softmax(self.out(ht))

    def initHidden(self, n):
        return torch.zeros(n, self.hidden_size),


class LSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        
        self.f = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.i = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.c = nn.Linear(embedding_dim + hidden_size, hidden_size)

        self.o = nn.Linear(embedding_dim + hidden_size, hidden_size)
    
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size), nn.LogSoftmax(dim=1))

    def forward(self, x, h, c):
        x = self.embeddings(x).squeeze(1)
        input_combined = torch.cat((x, h), 1)

        ft = torch.sigmoid(self.f(input_combined))
        it = torch.sigmoid(self.i(input_combined))
        ct = ft*c + it*torch.tanh(self.c(input_combined))
        ot = torch.sigmoid(self.o(input_combined))

        ht = ot*torch.tanh(ct)
        return ht, ct

    def decode(self, ht):
        return self.out(ht)

    def initHidden(self, n):
        return torch.zeros(n, self.hidden_size), torch.zeros(n, self.hidden_size) 


class GRU(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size):
            super(GRU, self).__init__()
            self.hidden_size = hidden_size
            self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

            self.z = nn.Linear(embedding_dim + hidden_size, hidden_size)
            self.r = nn.Linear(embedding_dim + hidden_size, hidden_size)
            self.h = nn.Linear(embedding_dim + hidden_size, hidden_size)


            self.out = nn.Sequential(nn.Linear(hidden_size, output_size), nn.LogSoftmax(dim=1))

        def forward(self, x, h):
            x = self.embeddings(x).squeeze(1)
            input_combined = torch.cat((x, h), 1)
            zt = torch.sigmoid(self.z(input_combined))
            rt = torch.sigmoid(self.r(input_combined))
            input_combined = torch.cat((rt*h, x), 1)
            ht = (1-zt)*h + zt * torch.tanh(self.h( input_combined))
            
            return ht, 

        def decode(self, ht):
            return self.out(ht)

        def initHidden(self, n):
            return torch.zeros(n, self.hidden_size), 

#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
