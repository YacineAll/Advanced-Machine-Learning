import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click

import numpy as np
from datamaestro import prepare_dataset





class MnistDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        return self.X[i].astype(np.float32), self.y[i].astype(np.int64)

    def __len__(self, ):
        return len(self.X)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=100),
            nn.ReLU,
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU,
            nn.Linear(in_features=100, out_features=100)
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=100, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var





ds = prepare_dataset("com.lecun.mnist")
train_image, train_labels = ds.train.images.data(), ds.train.labels.data()
test_image, test_labels   = ds.test.images.data(), ds.test.labels.data()
TRAIN_RATIO = 0.05
proportions = [int(len(train_image)*TRAIN_RATIO), int(len(train_image)*(1-TRAIN_RATIO))]
train, _ = random_split(MnistDataset(train_image, train_labels), proportions, generator=torch.Generator().manual_seed(42))
test = MnistDataset(test_image, test_labels)
trainloader = DataLoader(dataset=train, batch_size=300, shuffle=True)
testloader = DataLoader(dataset=test, batch_size=len(test), shuffle=False)


def train_classifier(train, test, model, criterion, optimizer, device, checkpoint_path, writer, n_iter=int(1e2)):
    state = load_state(checkpoint_path, model, optimizer)
    
    state.model = state.model.to(device)
    
    pbar = tqdm(range(n_iter), total=n_iter)
    for epoch in pbar:
        state.model.train()
        epoch_loss = []

        for x, y in train:

            x, y = x.to(device), y.to(device)
            state.optimizer.zero_grad()
            y_hat = state.model(x)
            loss = criterion(y_hat, y)
            epoch_loss.append(loss.item())
            loss.backward()
            state.optimizer.step()

            state.iteration += 1 
        

        
        state.epoch += 1 
        save_state(checkpoint_path, state)

        train_loss = eval_classifier(state.model, train, criterion, device)
        test_loss = eval_classifier(state.model, test, criterion, device)
        train_acc = accuracy(train, model)
        test_acc = accuracy(test, model)


        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/test', test_acc, epoch)

        pbar.set_description(f'Train Loss: {train_loss} Acc: {train_acc}\tTest Loss: {test_loss} Acc: {test_acc}')
        

#  TODO:  Implémenter
