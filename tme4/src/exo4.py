import sys
import argparse
import os
from datetime import datetime

import string
import unicodedata
import unidecode
import re

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from utils import RNN, device, State, save_state, load_state


# LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
# id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
# id2lettre[0]='' ##NULL CHARACTER
# lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

# def normalize(s):
#     return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

# def string2code(s):
#     return torch.tensor([lettre2id[c] for c in normalize(s)])

# def code2string(t):
#     if type(t) !=list:
#         t = t.tolist()
#     return ''.join(id2lettre[i] for i in t)


def string2code(s, lettre2id):
    return np.array([lettre2id[c] for c in s])


def code2string(t, lettre2id, id2lettre):
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


def embedding_function(data, n_lettres):
    one_hot = np.zeros((data.size, n_lettres))
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    return one_hot


def training_examples(X, length, n_lettres):
    results = []
    labels = []
    idx = 0
    while idx < len(X):
        length = length

        tmp = X[idx:idx+length]
        x = embedding_function(tmp[:-1].astype(np.int32), n_lettres)
        y = tmp[1:]
        if len(x) == length-1:
            results.append(x)
            labels.append(y)
        idx += length
    return np.array(results).astype(np.float32), np.array(labels).astype(np.int64)


class Rnn_Generator(RNN):
    def __init__(self, input_size, latent_size, output):
        super(Rnn_Generator, self).__init__(input_size, latent_size)
        self.out = nn.Linear(latent_size, output)
        self.decision = nn.Softmax(dim=1)

    def decode(self, outputs):
        result = []
        for out in outputs:
            y = self.out(out)
            y = self.decision(y)
            result.append(y)

        return torch.stack(result)


def train(train, model, criterion, optimizer, n_lettres, n_epochs, log_dir, checkpoint_path):
    losses = []
    writer = SummaryWriter(log_dir=log_dir)

    pbar = tqdm(range(n_epochs), total=n_epochs, file=sys.stdout)

    state = load_state(checkpoint_path, model, optimizer)

    for i in pbar:
        l = []
        for x, y in train:

            x = x.squeeze(-1).permute(1, 0, -1).to(device)
            seq_len, batch_size, embeding = x.shape

            y = y.view(seq_len*batch_size).to(device)

            o = state.model(x, state.model.initHidden(batch_size).to(device))
            d = state.model.decode(o).view(seq_len*batch_size, embeding)

            loss = criterion(d, y)
            loss.backward()

            state.optimizer.step()
            state.optimizer.zero_grad()

            l.append(loss.item())

            state.iteration += 1

        state.epoch += 1
        save_state(checkpoint_path, state)

        lo = np.mean(l)
        losses.append(lo)
        # \tTest: Loss: {np.round(test_lo, 4)}
        pbar.set_description(f'Train: Loss: {np.round(lo, 4)}')

        writer.add_scalar('Loss/train', lo, i)

    return losses


def generate_sequence(model, start, n_lettres, length, temperature, lettre2id, id2lettre):
    with torch.no_grad():
        l = [start]
        start = torch.from_numpy(
            embedding_function(np.array([lettre2id[start]]), n_lettres).astype(np.float32))
        h = model.initHidden(1).to(device)
        for _ in range(length):
            h = model.one_step(start, h)
            out = model.decode(h.unsqueeze(0))

            output_dist = out.data.view(-1).div(temperature).exp()
            nex = torch.distributions.categorical.Categorical(
                output_dist).sample().item()

            emb_nex = embedding_function(
                np.array([nex]), n_lettres).astype(np.float32)

            start = torch.from_numpy(emb_nex).unsqueeze(0)

            l.append(id2lettre[nex])
        return ''.join(l)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str,
                        help="Path data, it must be a text file!!")
    parser.add_argument("log_dir", type=str, help="tensorboard log result!!")
    parser.add_argument("-s", "--save", type=str,
                        help="checkpoints dir", default='./checkpoints')
    parser.add_argument("-BS", "--batch_size", type=int,
                        help="Batch size", default=16,)
    parser.add_argument("-NE", "--n_epochs", type=int,
                        help="number of epochs", default=10)
    parser.add_argument("-LS", "--latent_size", type=int,
                        help="Latent size", default=10)
    parser.add_argument("-LR", "--lr", type=float,
                        help="Learning rate", default=1e-3)

    args = parser.parse_args()

    checkpoint_dir = f"{args.save}/exo4"
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = f"{args.log_dir}/exo4"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = f'{checkpoint_dir}/checkpoint_' + \
        datetime.now().strftime('%d_%m_%Y_%H:%M:%S')

    data_path = args.data_path
    BATCH_SIZE = args.batch_size
    LATENT_SIZE = args.latent_size
    LR = args.lr
    N_EPOCHS = args.n_epochs

    with open(f'{data_path}') as f:
        text = f.readlines()

    text = text[0]+text[1]+text[2]+text[3]
    text = unidecode.unidecode(text)
    text = unicodedata.normalize('NFD', text)
    text = text.lower()
    text = text.translate(str.maketrans(
        "", "", re.sub('[\.|,|;]', '', string.punctuation)))
    text = text.strip()

    LETTRES = set(text)

    id2lettre = dict(zip(range(1, len(LETTRES)+1), LETTRES))
    lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))

    X = string2code(text, lettre2id).astype(np.float32)

    N_LETTRES = len(LETTRES)+1

    train_datset_diff = TextDataset(X)
    trainloader_diff = torch.utils.data.DataLoader(
        train_datset_diff, batch_size=BATCH_SIZE, shuffle=True, collate_fn=func(train_datset_diff))

    model = Rnn_Generator(
        input_size=1, latent_size=LATENT_SIZE, output=N_LETTRES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(
        train=trainloader_diff,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        n_lettres=N_LETTRES,
        n_epochs=N_EPOCHS,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    main()
