import torch
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MAINDIR = Path(os.path.dirname(
    os.path.dirname(os.path.realpath(Path(__file__)))))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model_file(name=""):
    checkpoint_dir = MAINDIR/f"checkpoints"
    os.makedirs(f"{checkpoint_dir}", exist_ok=True)
    checkpoint_path = f'{checkpoint_dir}/checkpoint_{name}'
    return checkpoint_path


def remove_model(checkpoint_path):
    try:
        os.remove(checkpoint_path)
        return True
    except FileNotFoundError:
        print("file not exists")
        return False


class State:
    def __init__(self, model, optimizer, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch, self.iteration = 0, 0


def save_state(checkpoint_path, state):
    savepath = Path(f"{checkpoint_path}")
    with savepath.open("wb") as f:
        torch.save(state, f)


def load_state(checkpoint_path, model, optimizer, lr_scheduler):
    savepath = Path(f"{checkpoint_path}")
    if savepath.is_file():
        with savepath.open("rb") as f:
            state = torch.load(f)
            return state
    return State(model, optimizer, lr_scheduler)


def train_classifier(train, test, state, criterion, checkpoint_path, n_iter=int(1e2)):
    log_dir = log_dir = f"{str(MAINDIR)}/experiments/experiments_day_" + \
        datetime.now().strftime('%d_%m_%Y_%H:%M:%S')
    os.makedirs(f"{log_dir}", exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    state.model = state.model.to(device)

    # pbar1 = tqdm(range(n_iter))
    # pbar2 = tqdm(train, desc="batchs", leave=False)

    pbar1 = tqdm(range(n_iter), total=n_iter)
    for epoch in pbar1:
        state.model.train()
        epoch_loss = []

        train_epoch(train, state, criterion, epoch_loss)


        state.epoch += 1

        train_acc, train_loss = eval_classifier(train, state.model, criterion)
        test_acc, test_loss = eval_classifier(test, state.model, criterion)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/test', test_acc, epoch)

        pbar1.set_description(
            f'Train Loss: {train_loss} Acc: {train_acc}\tTest Loss: {test_loss} Acc: {test_acc}')

    save_state(checkpoint_path, state)

def train_epoch(train, state, criterion, epoch_loss):
    for x, y in train:
        x, y = x.to(device), y.to(device)

        state.optimizer.zero_grad()
        y_hat = state.model(x)
        loss = criterion(y_hat, y)
        epoch_loss.append(loss.item())
        loss.backward()
        state.optimizer.step()

        state.iteration += 1

    state.lr_scheduler.step()


@torch.no_grad()
def eval_classifier(data, model, criterion):
    model = model.eval()
    ytrue, ypred, losses = [], [], []
    for x, y in data:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        yhat = torch.argmax(logits, dim=1)
        ytrue += y.tolist()
        ypred += yhat.tolist()
        losses.append(loss.item())
    return accuracy_score(ytrue, ypred), np.mean(losses)

