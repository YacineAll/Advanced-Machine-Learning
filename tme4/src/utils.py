
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pathlib import Path

from abc import ABC, abstractmethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


def fill_na(mat):
    ix, iy = np.where(np.isnan(mat))
    for i, j in zip(ix, iy):
        if np.isnan(mat[i+1, j]):
            mat[i, j] = mat[i-1, j]
        else:
            mat[i, j] = (mat[i-1, j]+mat[i+1, j])/2.
    return mat


def read_temps(path):
    """Lit le fichier de températures"""
    data = []
    with open(path, "rt") as fp:
        reader = csv.reader(fp, delimiter=',')
        next(reader)
        for row in reader:
            data.append([float(x) if x != "" else float('nan')
                         for x in row[1:]])
    return torch.tensor(fill_na(np.array(data)), dtype=torch.float)


class RNN(nn.Module, ABC):
    def __init__(self, input_size, latent_size):
        super(RNN, self).__init__()
        self.latent_size = latent_size

        self.x = nn.Linear(input_size, latent_size)
        self.h = nn.Linear(latent_size, latent_size)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        l = x.shape[0]
        results = []
        for i in range(l):
            h = self.one_step(x[i], h)
            results.append(h)
        return torch.stack(results)

    def one_step(self, x, h):
        h = self.h(h)
        x = self.x(x)
        return self.tanh(x+h)

    @abstractmethod
    def decode(self, h):
        pass

    def initHidden(self, n):
        return torch.zeros(n, self.latent_size).to(device)


class SequencesDatasetWithSameLength(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        super(SequencesDatasetWithSameLength, self).__init__()
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return np.expand_dims(np.array(self.sequences[idx]), -1), self.labels[idx]


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


# class SequencesDataset(torch.utils.data.Dataset):
#     def __init__(self, X, n_labels, mini=7, maxi=14):
#         super(SequencesDataset, self).__init__()
#         self.X = X
#         self.n_labels = n_labels
#         self.i = 0
#         self.mini, self.maxi = mini, maxi
#         self.length = np.random.randint(mini, maxi)

#     def __len__(self):
#         return len(self.X)

#     def get_element(self, idx):
#         col = np.random.randint(self.n_labels)
#         if (idx + self.length) > len(self.X):
#             idx -= self.length
#         r = self.X[idx:idx+self.length, col]
#         return r, col

#     def __getitem__(self, idx):
#         self.i += 1
#         x, y = self.get_element(idx)
#         return x.reshape(1,-1), y

#     def update(self):
#         self.length = np.random.randint(self.mini, self.maxi)
