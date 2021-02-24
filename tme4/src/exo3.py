
import sys
import argparse
import os
from datetime import datetime

from utils import RNN

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


fixed_length_train = 18
fixed_length_test = 14


class Rnn_forecasting(RNN):
    def __init__(self, input_size, latent_size, pas_de_temp, **kwargs):
        super(Rnn_forecasting, self).__init__(input_size, latent_size)
        self.out = nn.Linear(latent_size, 1)
        self.decision = nn.Sigmoid()
        self.pas_de_temp = pas_de_temp

    def decode(self, last_h):
        result = []
        for _ in range(self.pas_de_temp):
            x = self.out(last_h)
            x = self.decision(x).squeeze(1)
            result.append(x)

        return torch.stack(result, dim=-1).unsqueeze(-1)

class MultipleRNN(RNN):
    def __init__(self, n_villes, input_size, latent_size, pas_de_temp, **kwargs):
        super(MultipleRNN, self).__init__(input_size, latent_size)
        self.m = torch.nn.ModuleList([Rnn_forecasting(
            input_size=input_size, latent_size=latent_size, pas_de_temp=pas_de_temp) for _ in range(n_villes)])

    def forward(self, X, h=None):
        result = [list(range(len(self.m)))]
        l = []
        for i, model in enumerate(self.m):
            x = X[:, i, :, :]            
            out = model(x, model.initHidden(x.shape[0]))
            l.append(out[-1])
        result.append(l)
        return result
    
    def decode(self, outs):
        result = []
        for i, model in enumerate(self.m):
            pred = model.decode(outs[i])
            result.append(pred)
        return torch.cat(result, dim=-1).permute(0, 2, 1)

###################################################### Rnn_forecasting MC ######################################################

def train(model, criterion, optimizer, dataloader, test_loader, n_epochs, log_dir, checkpoint_path):
    writer = SummaryWriter(log_dir=log_dir)
    model = model.train()
    pbar = tqdm(range(n_epochs), total=n_epochs,
                position=0, leave=True, file=sys.stdout)
    losses = []

    state = load_state(checkpoint_path, model, optimizer)

    for i in pbar:
        l = []
        for x, y in dataloader:
            batch_size, seq_len, input_dim = x.shape
            x = x.permute(1, 0, -1)
            
            x, y = x.to(device), y.to(device)
            out = state.model(x, state.model.initHidden(batch_size))
            pred = state.model.decode(out[-1]).view(batch_size,-1)
            
            loss = criterion(pred, y)
            loss.backward()
            state.optimizer.step()
            state.optimizer.zero_grad()
            l.append(loss.item())

            state.iteration += 1

        test_lo = test(state.model, criterion, test_loader)
        lo = np.mean(l)
        losses.append(lo)
        pbar.set_description(
            f'Train: Loss: {np.round(lo, 4)}\tTest: Loss: {np.round(test_lo, 4)}')
        writer.add_scalar('Loss/train', lo, i)
        writer.add_scalar('Loss/test', test_lo, i)

        state.epoch += 1
        save_state(checkpoint_path, state)

    return losses


def test(model, criterion, dataloader):
    model = model.eval()
    l = []
    for x, y in dataloader:
        
        batch_size, seq_len, input_dim = x.shape
        x = x.permute(1, 0, -1)
            

        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x, model.initHidden(batch_size))
            pred = model.decode(out[-1]).view(batch_size,-1)

        loss = criterion(pred, y)
        l.append(loss.item())
    lo = np.mean(l)
    return lo


def training_examples(X, n_labels, length, pas_de_temps):
    results = []
    labels = []
    idx = 0
    while idx < len(X):
        length = length
        x = X[idx:idx+length, :n_labels]
        y = X[idx+length:idx+length+pas_de_temps, :n_labels]
        if len(y) == pas_de_temps and len(x) == length:
            results.append(x.reshape(-1))
            labels.append(y.reshape(-1))
        idx += length
    return np.array(results).astype(np.float32), np.array(labels).astype(np.float32)




def one_RNN(X_train, X_test, model, criterion, optimizer, batch_size, pas_de_temps, n_epochs, log_dir, checkpoint_path):
    sequences_train, labels_train = training_examples(
        X_train, N_LABELS, length=fixed_length_train, pas_de_temps=pas_de_temps)
    sequences_test, labels_test = training_examples(
        X_test, N_LABELS, length=fixed_length_test, pas_de_temps=pas_de_temps)

    traindataset = SequencesDatasetWithSameLength(
        sequences_train, labels=labels_train)
    testdataset = SequencesDatasetWithSameLength(
        sequences_test, labels=labels_test)

    trainloader = torch.utils.data.DataLoader(
        traindataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=batch_size, shuffle=True)

    losses = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=trainloader,
        test_loader=testloader,
        n_epochs=n_epochs,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path
    )

    return losses

###################################################### Rnn_forecasting MC ######################################################

###################################################### Multi RNN ######################################################



def trainMR(model, criterion, optimizer, dataloader, test_loader, n_epochs, log_dir, checkpoint_path):

    model = model.train()
    pbar = tqdm(range(n_epochs), total=n_epochs, leave=False,
                dynamic_ncols=True, file=sys.stdout)
    losses = []

    writer = SummaryWriter(log_dir=log_dir)
    state = load_state(checkpoint_path, model, optimizer)

    for i in pbar:
        l = []
        for x, y in dataloader:
            batch_size = x.shape[0]

            x = x.view(batch_size, -1, len(model.m))
            x = x.permute(1, 0, -1)

            x, y = x.to(device), y.to(device)
            pred = state.model(x).view(batch_size, -1)
            loss = criterion(pred, y)
            loss.backward()
            state.optimizer.step()
            state.optimizer.zero_grad()
            l.append(loss.item())

            state.iteration += 1

        state.epoch += 1
        save_state(checkpoint_path, state)

        test_lo = testMR(state.model, criterion, test_loader)
        lo = np.mean(l)
        losses.append(lo)
        pbar.set_description(
            f'Train: Loss: {np.round(lo, 4)}\tTest: Loss: {np.round(test_lo, 4)}')

        writer.add_scalar('Loss/train', lo, i)
        writer.add_scalar('Loss/test', test_lo, i)

    return losses


def testMR(model, criterion, dataloader):
    model = model.eval()
    l = []
    for x, y in dataloader:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, len(model.m))
        x = x.permute(1, 0, -1)

        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x).view(batch_size, -1)

        loss = criterion(pred, y)
        l.append(loss.item())
    lo = np.mean(l)
    return lo


def multi_RNN(X_train, X_test, model, criterion, optimizer, batch_size, pas_de_temps, n_epochs, log_dir, checkpoint_path):

    sequences_train, labels_train = training_examples(
        X_train, N_LABELS, length=fixed_length_train, pas_de_temps=pas_de_temps)
    sequences_test, labels_test = training_examples(
        X_test, N_LABELS, length=fixed_length_test, pas_de_temps=pas_de_temps)

    traindataset = SequencesDatasetWithSameLength(
        sequences_train, labels=labels_train)
    testdataset = SequencesDatasetWithSameLength(
        sequences_test, labels=labels_test)

    trainloader = torch.utils.data.DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    losses = trainMR(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=trainloader,
        test_loader=testloader,
        n_epochs=n_epochs,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path
    )
    return losses


###################################################### Multi RNN ######################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("train_data", type=str,
                        help="Path data, it must be a text file!!")
    parser.add_argument("test_data", type=str,
                        help="Path data, it must be a text file!!")
    parser.add_argument("log_dir", type=str, help="tensorboard log result!!")
    parser.add_argument("-s", "--save", type=str,
                        help="checkpoints dir", default='./checkpoints')
    parser.add_argument("-NL", "--n_labels", type=int,
                        help="Define number of labels to predict", default=10)
    parser.add_argument("-p", "--p_temps", type=int,
                        help="i le pas de temps t+i", default=1)
    parser.add_argument("-mr", "--multi_rnn",
                        help="using multi rnn (rnn for each label)", action="store_true")
    parser.add_argument("-LS", "--latent_size", type=int,
                        help="Latent size", default=10)

    parser.add_argument("-LR", "--lr", type=float,
                        help="Learning rate", default=1e-3)

    parser.add_argument("-BS", "--batch_size", type=int,
                        help="Define the batch size", default=16)

    parser.add_argument("-NE", "--n_epochs", type=int,
                        help="Define number of epochs on training", default=10)

    args = parser.parse_args()

    checkpoint_dir = f"{args.save}/exo3"
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = f"{args.log_dir}/exo3"
    os.makedirs(log_dir, exist_ok=True)

    # OUTPUTS = 1
    N_LABELS = args.n_labels

    tempiratures_train = pd.read_csv(args.train_data, low_memory=False)
    tempiratures_test = pd.read_csv(
        args.test_data, low_memory=False, header=None)

    X_train = tempiratures_train.iloc[:11115, 1:].dropna()
    X_train = pd.concat(
        [X_train, tempiratures_train.iloc[11116:-1, 1:].dropna()], axis=0)
    X_test = tempiratures_test.iloc[:, 1:].dropna()

    X_train = StandardScaler().fit_transform(X_train).astype(np.float32)
    X_test = StandardScaler().fit_transform(X_test).astype(np.float32)

    criterion = torch.nn.MSELoss()

    if args.multi_rnn:
        checkpoint_path = f'{checkpoint_dir}/checkpoint_m_rnn_' + \
            datetime.now().strftime('%d_%m_%Y_%H:%M:%S')

        m_rnn = MultipleRNN(
            n_model=N_LABELS,
            input_size=1,
            latent_size=args.latent_size,
            pas_de_temp=args.p_temps
        )

        optimizer = torch.optim.Adam(m_rnn.parameters(), lr=args.lr)
        m_rnn.to(device)
        print(f"For multi_RNN")
        multi_RNN(
            X_train=X_train,
            X_test=X_test,
            model=m_rnn,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=args.batch_size,
            pas_de_temps=args.p_temps,
            n_epochs=args.n_epochs,
            log_dir=log_dir,
            checkpoint_path=checkpoint_path
        )

    else:
        checkpoint_path = f'{checkpoint_dir}/checkpoint_one_rnn_' + \
            datetime.now().strftime('%d_%m_%Y_%H:%M:%S')
        model = Rnn_forecasting(
            input_size=1,
            latent_size=args.latent_size,
            output=1,
            pas_de_temp=args.p_temps
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.to(device)
        print(f"One RNN")
        one_RNN(
            X_train=X_train,
            X_test=X_test,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=args.batch_size,
            pas_de_temps=args.p_temps,
            n_epochs=args.n_epochs,
            log_dir=log_dir,
            checkpoint_path=checkpoint_path
        )


#  TODO:  Question 3 : Prédiction de séries temporelles
