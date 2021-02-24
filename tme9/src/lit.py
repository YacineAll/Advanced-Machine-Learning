import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class MyLitDataModule(pl.LightningDataModule):
    """Some Information about MyLitDataModule"""
    def __init__(self,
                collate_fn,
                trainset:Dataset,
                validset:Dataset,
                testset:Dataset,
                batch_size:int=128,
                data_dir: str ='.'
        ):
        super(MyLitDataModule, self).__init__()
        self.collate_fn=collate_fn
        self.batch_size = batch_size
        self.data = (trainset, validset, testset)



    def setup(self, stage=None):
        trainset, validset, testset = self.data
        self.trainset = trainset
        self.validset = validset
        self.testset = testset



    def train_dataloader(self):
        return DataLoader(dataset=self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn)



    def val_dataloader(self):
        return DataLoader(dataset=self.validset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.collate_fn)



    def test_dataloader(self):
        return DataLoader(dataset=self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.collate_fn)



class LitModel(pl.LightningModule):

    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone



    def forward(self, x):
        x = self.backbone(x)
        return x



    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


