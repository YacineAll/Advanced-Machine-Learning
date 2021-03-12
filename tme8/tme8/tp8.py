import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import datetime

from collections import OrderedDict

from datamaestro import prepare_dataset
from sklearn.utils import shuffle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_RATIO      = 0.05    # Ratio du jeu de train à utiliser
INPUT_DIM        = 28 * 28
HIDDEN_DIM       = 100
OUTPUT_DIM       = 10
BATCH_SIZE       = 300
N_EPOCH          = 1000
FREQ_STORE_GRAD  = N_EPOCH // 20
LEARNING_RATE    = 3e-2

ds = prepare_dataset("com.lecun.mnist")

train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
train_img = train_img.reshape(-1, INPUT_DIM)
train_img, train_labels = shuffle(train_img, train_labels)

idx_train = int(len(train_img) * TRAIN_RATIO)
train_img, train_labels = train_img[:idx_train], train_labels[:idx_train]
mu, sigma = train_img.mean(0), train_img.std(0) + 1e-5
train_img = (train_img - mu)/sigma

test_img, test_labels = ds.test.images.data(), ds.test.labels.data()
test_img = train_img.reshape(-1, INPUT_DIM)
test_img = (test_img - mu)/sigma

class FlattenedImageDataset(Dataset):
    def __init__(self, img, label):
        self.img = torch.tensor(img, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.long)
    
    def __getitem__ (self, index):
        return self.img[index], self.label[index]
    
    def __len__(self):
        return len(self.img)
    
train_iter = DataLoader(FlattenedImageDataset(train_img, train_labels),
                        batch_size=BATCH_SIZE)
test_iter  = DataLoader(FlattenedImageDataset(test_img, test_labels),
                        batch_size=BATCH_SIZE)

x_val, y_val = next(iter(test_iter))

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

criterion = torch.nn.CrossEntropyLoss()


class Model():
    def __repr__(self):
        s = f"model_{self.lambda1:.3f}_{self.lambda2:.3f}"
        if self.batchnorm:
            s += "_bn"
        if self.layernorm:
            s += "_ln"
        if self.dropout:
            s += f"dropout_{self.p:.2f}"
            
        return s
    
    def __init__(self, dropout=False, p=0.5, batchnorm=False, layernorm=False,
                 lambda1=0, lambda2=0):
        
        self.dropout = dropout
        self.p = p
        self.batchnorm = batchnorm
        self.layernorm = layernorm

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        layers = OrderedDict()
    
        layers['fc1'] = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        if batchnorm:
            layers['bn1'] = nn.BatchNorm1d(HIDDEN_DIM)
        if layernorm:
            layers['ln1'] = nn.LayerNorm(HIDDEN_DIM)
        layers['relu1'] = nn.ReLU()
        
        layers['fc2'] = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        if batchnorm:
            layers['bn2'] = nn.BatchNorm1d(HIDDEN_DIM)
        if layernorm:
            layers['ln2'] = nn.LayerNorm(HIDDEN_DIM)
        layers['relu2'] = nn.ReLU()
        
        layers['fc3'] = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        if batchnorm:
            layers['bn3'] = nn.BatchNorm1d(HIDDEN_DIM)
        if layernorm:
            layers['ln3'] = nn.LayerNorm(HIDDEN_DIM)
        if dropout:
            layers['dropout'] = nn.Dropout(p=p)
        layers['relu3'] = nn.ReLU()
        
        layers['fc4'] = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
    
        model =  nn.Sequential(layers)
        
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"./runs/runs_{dt}/{self.__repr__()}")
        
        self.writer = writer


models = [Model(),
          Model(lambda1=0.1),
          Model(lambda1=1.0),
          Model(lambda1=10),
          Model(lambda2=0.1),
          Model(lambda2=1),
          Model(lambda2=10),
          Model(lambda1=0.1, lambda2=0.1),
          Model(batchnorm=True),
          Model(layernorm=True),
          Model(dropout=True, p=0.2),
          Model(dropout=True, p=0.3),
          Model(dropout=True, p=0.4),
          Model(dropout=True, p=0.5),
          Model(dropout=True, p=0.6),
          Model(dropout=True, p=0.7),
          ]


for m in models:
    print(m)
    model = m.model.to(device)
    optimizer = m.optimizer
    writer = m.writer

    for epoch in range(N_EPOCH):
        cumloss_train = 0
        model.train()
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            cumloss_train += (loss.item() / BATCH_SIZE)
            store_grad(yhat)
            loss.backward()
            optimizer.step()

        cumloss_test = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_iter:
                x, y = x.to(device), y.to(device)
                yhat = model(x)
                loss = criterion(yhat, y)
                cumloss_test += (loss.item() / BATCH_SIZE)
        
        writer.add_scalar('Loss/train', cumloss_train, epoch)
        writer.add_scalar('Loss/test', cumloss_test, epoch)
        
        if not epoch % FREQ_STORE_GRAD:
            for name, module in model.named_children():
                if name.startswith('fc'):
                    for i, p in enumerate(module.parameters()):
                        writer.add_histogram(f'Weight/{name}/{i}', p, epoch)

            out = x_val.to(device)
            grads = []
            for name, module in model.named_children():
                out = module(out)
                if name.startswith('fc'):
                    grads.append((name, store_grad(out)))
            loss = criterion(out, y_val.to(device))
            
            for name, g in grads:
                writer.add_histogram(f'Gradient/{name}', g, epoch)
