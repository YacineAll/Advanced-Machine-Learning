import argparse
import os
from datetime import datetime


from utils import RNN, device, SequencesDatasetWithSameLength, State, save_state, load_state
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score


class Rnn_classifier(RNN):
    def __init__(self, input_size, latent_size, output):
        super(Rnn_classifier, self).__init__(input_size, latent_size)
        self.out = nn.Linear(latent_size, output)
        self.decision = nn.Sigmoid() if output == 2 else nn.Softmax(dim=1)

    def decode(self, h):
        x = self.out(h)
        return self.decision(x)


def train(
        model,
        criterion,
        optimizer,
        dataloader,
        test_loader,
        n_epochs,
        n_label,
        device,
        log_dir,
        checkpoint_path):

    writer = SummaryWriter(log_dir=log_dir)

    model = model.train()
    pbar = tqdm(range(n_epochs), total=n_epochs)
    losses = []
    accuracies = []

    state = load_state(checkpoint_path, model, optimizer)

    for i in pbar:
        l = []
        y_hat = []
        y_true = []
        for x, y in dataloader:
            batch_size, seq_len, _ = x.shape
            x = x.permute(1, 0, -1)

            x, y = x.to(device), y.to(device)
            out = state.model(x, state.model.initHidden(batch_size))
            pred = state.model.decode(out[-1])
            y_hat += list(np.argmax(pred.detach().numpy(), axis=1))
            y_true += list(y.detach().numpy())
            loss = criterion(pred.reshape(-1) if n_label == 2 else pred, y)
            loss.backward()
            state.optimizer.step()
            state.optimizer.zero_grad()
            l.append(loss.item())

            state.iteration += 1

        lo = np.mean(l)
        acc = accuracy_score(y_true, y_hat)*100
        test_lo, test_acc = test(
            state.model, criterion, test_loader, n_label, device)
        pbar.set_description(
            f'Train: Loss: {np.round(lo, 4)} Acc: {np.round(acc,2)}%\tTest: Loss: {np.round(test_lo, 4)} Acc: {np.round(test_acc,2)}%')

        state.epoch += 1
        save_state(checkpoint_path, state)

        writer.add_scalar('Loss/train', lo, i)
        writer.add_scalar('Loss/test', test_lo, i)
        writer.add_scalar('Acc/train', acc, i)
        writer.add_scalar('Acc/test', test_acc, i)

        accuracies.append(acc)
        losses.append(lo)
    return losses, accuracies


def test(model, criterion, dataloader, n_label, device):
    model = model.eval()
    l = []
    y_hat = []
    y_true = []
    for x, y in dataloader:
        batch_size, seq_len, _ = x.shape
        x = x.permute(1, 0, -1)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x, model.initHidden(batch_size))
            pred = model.decode(out[-1])
        y_hat += list(np.argmax(pred.detach().numpy(), axis=1))
        y_true += list(y.detach().numpy())
        loss = criterion(pred.reshape(-1) if n_label == 2 else pred, y)
        l.append(loss.item())
    lo = np.mean(l)
    acc = accuracy_score(y_true, y_hat)*100
    return lo, acc


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)/float(labels.size)

###################################################### With Same Length ######################################################


def training_examples(X, n_labels, length):
    results = []
    labels = []
    idx = 0
    while idx < len(X):
        col = np.random.randint(n_labels)
        length = length
        value = X[idx:idx+length, col]
        if len(value) == length:
            results.append(value)
            labels.append(col)
        idx += length
    return np.array(results).astype(np.float32), np.array(labels).astype(np.float32) if n_labels == 2 else np.array(labels).astype(np.int64)


def collate_fn(batch):
    x, y = zip(*batch)
    x, y = torch.from_numpy(np.array(x).transpose(
        1, 0, 2)), torch.from_numpy(np.array(y))
    return x, y


def same_sequence_length(log_dir, checkpoint_path, n_epochs_same, batch_size, fixed_length_train=18, fixed_length_test=14):

    sequences_train_same, labels_train_same = training_examples(
        X_train, N_LABELS, length=fixed_length_train)
    sequences_test, labels_test = training_examples(
        X_test, N_LABELS, length=fixed_length_test)

    traindataset_same = SequencesDatasetWithSameLength(
        sequences_train_same, labels=labels_train_same)
    testdataset_same = SequencesDatasetWithSameLength(
        sequences_test, labels=labels_test)

    trainloader_same = torch.utils.data.DataLoader(
        traindataset_same, batch_size=batch_size, shuffle=True)
    testloader_same = torch.utils.data.DataLoader(
        testdataset_same, batch_size=batch_size, shuffle=True)

    losses, accuracies = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=trainloader_same,
        test_loader=testloader_same,
        n_epochs=n_epochs_same,
        n_label=N_LABELS,
        device=device,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path
    )

    return losses, accuracies

########################################################## With Same Length ######################################################
########################################################  With diffrent length ######################################################


def func(dataset):
    def collate_fn(batch):
        x, y = zip(*batch)
        x, y = torch.from_numpy(np.array(x).transpose(
            2, 0, 1)), torch.from_numpy(np.array(y))
        dataset.update()
        return x, y
    return collate_fn


def With_diffrent_length(log_dir, checkpoint_path, n_epochs, batch_size):
    train_datset_diff = SequencesDataset(X_train, n_labels=N_LABELS)
    test_datset_diff = SequencesDataset(X_test, n_labels=N_LABELS)

    trainloader_diff = torch.utils.data.DataLoader(
        train_datset_diff, batch_size=batch_size, shuffle=True, collate_fn=func(train_datset_diff))
    testloader_diff = torch.utils.data.DataLoader(
        test_datset_diff, batch_size=batch_size, shuffle=True, collate_fn=func(test_datset_diff))

    losses, accuracies = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=trainloader_diff,
        test_loader=testloader_diff,
        n_epochs=n_epochs,
        n_label=N_LABELS,
        device=device,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path
    )

    return losses, accuracies
###################################################### With diffrent length ######################################################


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

    parser.add_argument("-BS", "--batch_size", type=int,
                        help="Define the batch size", default=16)

    parser.add_argument("-NE", "--n_epochs", type=int,
                        help="Define number of epochs on training", default=10)

    parser.add_argument("-sl", "--same_length",
                        help="Train model with the sequences with the same length", action="store_true")
    parser.add_argument("-LS", "--latent_size", type=int,
                        help="Latent size", default=10)
    parser.add_argument("-LR", "--lr", type=float,
                        help="Learning rate", default=1e-3)

    args = parser.parse_args()

    checkpoint_dir = f"{args.save}/exo2"
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = f"{args.log_dir}/exo2"
    os.makedirs(log_dir, exist_ok=True)

    N_LABELS = args.n_labels

    tempiratures_train = pd.read_csv(args.train_data, low_memory=False)
    tempiratures_test = pd.read_csv(
        args.test_data, low_memory=False, header=None)

    X_train = tempiratures_train.iloc[:11115, 1:].dropna()
    X_train = pd.concat(
        [X_train, tempiratures_train.iloc[11116:-1, 1:].dropna()], axis=0)
    X_test = tempiratures_test.iloc[:, 1:].dropna()

    X_train = MinMaxScaler().fit_transform(X_train).astype(np.float32)
    X_test = MinMaxScaler().fit_transform(X_test).astype(np.float32)

    model = Rnn_classifier(1, args.latent_size,
                           1 if N_LABELS == 2 else N_LABELS)
    criterion = torch.nn.BCELoss() if N_LABELS == 2 else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)

    if args.same_length:
        checkpoint_path = f'{checkpoint_dir}/checkpoint_same_l_' + \
            datetime.now().strftime('%d_%m_%Y_%H:%M:%S')
        print(f"With same sequences length")
        same_sequence_length(
            log_dir,
            checkpoint_path,
            n_epochs_same=args.n_epochs,
            batch_size=args.batch_size
        )
    else:
        checkpoint_path = f'{checkpoint_dir}/checkpoint_diff_l_' + \
            datetime.now().strftime('%d_%m_%Y_%H:%M:%S')

        print(f"With diffrent sequences length")
        With_diffrent_length(
            log_dir,
            checkpoint_path,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size
        )
