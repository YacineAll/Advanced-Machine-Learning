import csv
import numpy as np
import logging
import time
import string
from itertools import chain
from abc import ABC, abstractmethod
from tqdm import tqdm

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maskedCrossEntropy(output, y, m):
    logits = output.view(-1, output.shape[-1])
    target = y.view(-1)
    return ((F.cross_entropy(logits, target, reduction='none').view(len(y), -1)*m).sum(0)/m.sum(0)).mean()



class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    def forward(self, logit, target, mask):
        loss = F.nll_loss(logit, target, reduction='none')
        loss *= mask
        return loss.sum() / mask.sum()

class RNN(nn.Module, ABC):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.out = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Softmax(dim=-1))


    def forward(self, x, h):
        results=[]
        for xi in x:
            h = self.one_step(xi, *h)
            results.append(h[0])
        return torch.stack(results)
    
    def decode(self, h):
        return self.out(h)

    def one_step(self, x, h):
        return self.embeddings(x).squeeze(1)


    @abstractmethod
    def initHidden(self):
        pass


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
        return torch.zeros(n, self.hidden_size).to(device), torch.zeros(n, self.hidden_size).to(device)
        
        
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
            return torch.zeros(n, self.hidden_size).to(device),

def train_step(state,model, criterion, x, y, m):
    x.unsqueeze_(-1)
    h = state.model.initHidden(x.shape[1])
    loss = 0
    for i, x in enumerate(x):
        h = state.model(x, *h)
        logits = state.model.decode(h[0])
        loss += criterion(logits, y[i], m[i])
    return loss
    

def train(train, model, criterion, optimizer, scheduler, n_epochs, log_dir, checkpoint_path):
    losses = []
    writer = SummaryWriter(log_dir=log_dir)
    pbar = tqdm(range(n_epochs), total=n_epochs, file=sys.stdout)
    state = load_state(checkpoint_path, model, optimizer)

    for i in pbar:
          l = []
          for x in train:
              x, y, m = x[:-1].to(device), x[1:].to(device), (x[1:]!=PAD_IX ).to(device)

              loss = train_step(state,model, criterion, x, y, m)
              
              state.optimizer.zero_grad()
              loss.backward()            
              state.optimizer.step()
              l.append(loss.item()/len(x))
              state.iteration += 1
          
          state.epoch +=1
          save_state(checkpoint_path, state)

          #scheduler.step()
          lo = np.mean(l)
          losses.append(lo)

          writer.add_scalar('Loss/train', lo, i)

          pbar.set_description(f'Train: Loss: {np.round(lo, 4)}') # \tTest: Loss: {np.round(test_lo, 4)}
          pbar.update()

    return losses
    
    
class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.epoch, self.iteration = 0, 0


def save_state(checkpoint_path, state):
    savepath = Path(f"{checkpoint_path}")
    with savepath.open("wb") as f:
        torch.save(state, f)
def load_state(checkpoint_path, model, optimizer):
    savepath = Path(f"{checkpoint_path}")
    if savepath.is_file():
        with savepath.open("rb") as f:
            state = torch.load(f)
            return state
    return State(model, optimizer)