import torch
from torch.utils.tensorboard import SummaryWriter
# Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm

from torch.autograd import Function
import torch.nn as nn


import numpy as np
from sklearn.preprocessing import StandardScaler


class MSE:
    """Début d'implementation de la fonction MSE"""

    def forward(self, yhat, y):
        return torch.mean((yhat-y)**2)
    def __call__(self, yhat, y):
        return self.forward(yhat, y)


class Linear:
    """Début d'implementation de la fonction Linear"""
    def forward(self, X, W, b):
        return torch.matmul(X, W) + b

    def __call__(self, X, W, b):
        return self.forward(X, W, b)


class Housing_data(torch.utils.data.Dataset):
    def __init__(self, transform=True):
        data = np.loadtxt('../data/housing.data')
        self.X = data[:, :-1]
        self.y = data[:, -1]
        if transform:
            self.X = StandardScaler().fit_transform(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]


class Net(nn.Module):
    def __init__(self, shape_l1):
        super().__init__()
        input_size_1, output_size_1 = shape_l1

        self.resultat = nn.Sequential(
            nn.Linear(input_size_1, output_size_1),
            nn.Tanh(),
            nn.Linear(output_size_1, input_size_1)
        )
    def forward(self, x_in):
        return self.resultat(x_in)
