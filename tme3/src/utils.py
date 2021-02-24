import sys
import struct
from array import array
from os.path  import join
from  pathlib  import Path


import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))



class MNIST_DataSet(Dataset):
    def __init__(self, X, y, transform=True, transform_func=None):
        self.X = np.array(X, copy=True).astype(np.float32)
        self.y = np.array(y, copy=True).astype(np.int64)
        self.transform = transform
        self.transform_func = transform_func



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx].reshape(-1)
        if self.transform : 
            x = x / 255
        if self.transform_func:
            x = self.transform_func(torch.from_numpy(x)).detach().cpu().numpy()
        return x, self.y[idx]

class MNIST():
    def __init__(self, train_batch_size = 16, test_size=None, transform=True, transform_func=None, input_path=""):
        X_train, y_train, X_val, y_val = MnistDataloader(input_path).load_data()

        self.__test = None
        if test_size :
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size)
            self.__test = MNIST_DataSet(X_test, y_test, transform, transform_func)
            self.__test_loader = DataLoader(self.__test, batch_size = len(self.__test), shuffle=False, num_workers=0)

        self.__train = MNIST_DataSet(X_train, y_train, transform, transform_func)
        self.__val = MNIST_DataSet(X_val, y_val, transform, transform_func)

        self.__train_loader = DataLoader(self.__train, batch_size = train_batch_size, shuffle=True, num_workers=0)
        self.__val_loader = DataLoader(self.__test, batch_size = len(self.__test), shuffle=False, num_workers=0)


    def infos(self):
        if self.__test :
            return self.__train.X.shape, self.__val.X.shape, self.__test.X.shape 
        else:
            return self.__train.X.shape, self.__val.X.shape 


    def data(self):
        if self.__test:
            return self.__train, self.__val, self.__test
        return self.__train, self.__val

    def loader(self):
        if self.__test:
            return self.__train_loader, self.__val_loader, self.__test_loader
        return self.__train_loader, self.__val_loader



class MnistDataloader(object):
    def __init__(self, input_path):
        self.training_images_filepath = f"{input_path}/train-images-idx3-ubyte"
        self.training_labels_filepath = f"{input_path}/train-labels-idx1-ubyte"
        self.test_images_filepath = f"{input_path}/t10k-images-idx3-ubyte"
        self.test_labels_filepath = f"{input_path}/t10k-labels-idx1-ubyte"

    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels

    
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return x_train, y_train, x_test, y_test        




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
        

def train_autoencoder(train, test, model, criterion, optimizer, device,  checkpoint_path, writer, n_iter=int(1e2)):

    state = load_state(checkpoint_path, model, optimizer)
    state.model = state.model.to(device)
    
    
    pbar = tqdm(range(n_iter), total=n_iter)
    for epoch in pbar:
        state.model.train()
        epoch_loss = []

        for x, _ in train:

            x = x.to(device)
            state.optimizer.zero_grad()
            y_hat = state.model(x)
            loss = criterion(y_hat, x)
            epoch_loss.append(loss.item())
            loss.backward()
            state.optimizer.step()


            state.iteration += 1 
        

        state.epoch += 1  
        save_state(checkpoint_path, state)
        

        train_loss = eval_autoencoder(model, train, criterion, device)
        test_loss  = eval_autoencoder(model, test, criterion, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)

        pbar.set_description(f'Train/Loss: {train_loss}\tTest/Loss: {test_loss}')


def eval_classifier(model, test, criterion, device):
    if test is None:
        return

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test:
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss += criterion(output, y).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    return test_loss



def eval_autoencoder(model, test, criterion, device):
    if test is None:
        return
    model = model.to(device)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, _ in test:
            x = x.to(device)
            output = model(x)
            test_loss += criterion(output, x).item()
    return test_loss


def accuracy(dataloader, model):
    y_hat = []
    y_true = []
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            y_hat  += list(np.argmax(pred.detach().numpy(), axis=1))
            y_true += list(y.detach().numpy())

    return accuracy_score(y_true, y_hat)*100
