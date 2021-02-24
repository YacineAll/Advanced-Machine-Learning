from abc import abstractmethod, ABC

import pandas as pd
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.regression import MeanSquaredError

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from typing import Union, Tuple


class SequenceDataset(torch.utils.data.Dataset):
    """Some Information about SequenceDataset"""

    def __init__(self, x: np.ndarray, y: np.ndarray, forcasting:bool=False):
        super(SequenceDataset, self).__init__()
        self.x = x
        self.y = y
        self.forcasting=forcasting

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.forcasting:
            return x.astype(np.float32), y.astype(np.float32)
        return x.astype(np.float32), y.astype(np.int64)

    def __len__(self):
        return len(self.x)


class Temperature(pl.LightningDataModule):
    """Some Information about MyModule"""
    def __init__(self,
                seq_length: Union[int, Tuple[int, int]] = 10,
                batch_size: int = 32,
                data_dir: str = '.',
                columns: Union[list, str] = "all",
                task: int = 0,
                normalize:bool=False,
                multi:bool=False,
                **kwargs,
    ):

        super(Temperature, self).__init__()
        temperatures_train = pd.read_csv(
            f'{data_dir}/tempAMAL_train.csv', low_memory=False)
        temperatures_test = pd.read_csv(
            f'{data_dir}/tempAMAL_test.csv', low_memory=False, header=None)
        train = temperatures_train.iloc[:11115, :].dropna()
        train = pd.concat(
            [train, temperatures_train.iloc[11116:-1, :].dropna()], axis=0)
        train = train.set_index('datetime')
        test = temperatures_test.iloc[1:, :].dropna()
        test = test.set_index(0)

        train = StandardScaler().fit_transform(train.values)
        test  = StandardScaler().fit_transform(test.values)

        if columns == "all":
            _, l = train
            columns = np.arange(l)

        self.batch_size = batch_size
        self.train = train
        self.test = test
        self.seq_length = seq_length
        self.columns = columns
        self.task = task
        self.multi = multi

    def create_sequences_multi(self, data, length, columns, t=1):
        def rolling_window(a, window):
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            array = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
            return array
        data = data[:, columns]
        arrays = list(map(lambda x: rolling_window(x, length+t), data.T))
        X = np.stack(arrays, axis=1)

        x = np.expand_dims(X[:,:,:length], axis=-1)
        y = X[:,:,length:]

        return x, y

    def create_sequences(self, data: np.ndarray, length: int, columns: list, task: int = 0):
        def rolling_window(a, window, label=None):
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            array = np.lib.stride_tricks.as_strided(
                a, shape=shape, strides=strides)
            if label is not None:
                return np.hstack([array, np.ones((len(array), 1))*int(label)])
            return array
        data = data[:, columns]
        if task > 0:
            arrays = np.vstack(
                list(map(lambda v: rolling_window(v, length+task), data.T)))
            x = np.expand_dims(arrays[:, :-task], axis=-1)
            y = np.expand_dims(arrays[:, -task:], axis=-1)
            return x, y

        labels = np.array(columns).astype(str)
        arrays = np.vstack(
            list(map(lambda v, c: rolling_window(v, length, c), data.T, labels)))
        x = np.expand_dims(arrays[:, :-1], axis=-1)
        y = arrays[:, -1]
        return x, y

    def setup(self, stage=None):
        if self.multi:
            X, y = self.create_sequences_multi(
                self.train, self.seq_length, self.columns, self.task)
            self.trainset = SequenceDataset(X, y, forcasting=True)
            X, y = self.create_sequences_multi(
                self.test, self.seq_length, self.columns, self.task)
            self.validset = SequenceDataset(X, y, forcasting=True)
        else:
            X, y = self.create_sequences(
                self.train, self.seq_length, self.columns, self.task)
            self.trainset = SequenceDataset(X, y, forcasting=self.task>0)
            X, y = self.create_sequences(
                self.test, self.seq_length, self.columns, self.task)
            self.validset = SequenceDataset(X, y, forcasting=self.task>0)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size)




class RNN(nn.Module, ABC):
    def __init__(self, input_size: int, latent_size: int):
        super(RNN, self).__init__()
        self.latent_size = latent_size

        self.x = nn.Linear(input_size, latent_size)
        self.h = nn.Linear(latent_size, latent_size)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        l = x.shape[1]
        results = []
        for i in range(l):
            h = self.one_step(x[:,i,:], h)
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
        return torch.zeros(n, self.latent_size, dtype=torch.float32)


class LitModel(pl.LightningModule):
    def __init__(self, backbone:RNN, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.backbone = backbone
        if self.hparams.forcasting:
            self.criterion = nn.MSELoss()
            self.metric = MeanSquaredError()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.metric = Accuracy()

    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.backbone(x, self.backbone.initHidden(len(x)))
        logits = self.backbone.decode(out[-1])
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.backbone(x, self.backbone.initHidden(len(x)))
        logits = self.backbone.decode(out[-1])
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)

        if self.hparams.forcasting:
            metric_value = self.metric(logits, y)
            self.log('val_mse',  metric_value, prog_bar=True)
        else:
            preds = torch.argmax(logits, dim=1)
            metric_value = self.metric(preds, y)
            self.log('val_acc',  metric_value, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
