import itertools
import logging
from tqdm import tqdm
import pickle

from pathlib import Path



# from datamaestro import prepare_dataset
from conllu import parse_incr, parse
import pyconll

import nltk
from sklearn.metrics import accuracy_score
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class DS_url:
    def __init__(self,):
        data_path = lambda x: f"https://github.com/UniversalDependencies/UD_French-GSD/raw/master/fr_gsd-ud-{x}.conllu"
        
        self.train = parse(pyconll.load_from_url(data_path('train')).conll())
        self.validation = parse(pyconll.load_from_url(data_path('dev')).conll())
        self.test = parse(pyconll.load_from_url(data_path('test')).conll())
        


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



def get_embedding_layer(words, path):
    embeding_layer = torch.nn.Embedding(len(words), 300)
    #torch.save(embeding_layer.state_dict(),f"{path}/embeddings_layer.pth")
    embeding_layer = embeding_layer.eval()
    return embeding_layer


def evaluate(data, model, criterion):
    model = model.eval()
    with torch.no_grad():
        ytrue, ypred, losses = [], [], []
        for x, y in data:
            batch_size = y.data.size(1)
            x, y = x.to(device), y.to(device).view(-1)
            mask = y!=0
            
            h, c = model.init_hidden(batch_size)
            h, c = h.to(device), c.to(device)
            
            logits = model(x, h, c)
            
            loss = criterion(logits, y)
            yhat = torch.argmax(logits, dim=1)
            
            ytrue  += y[mask].tolist()
            ypred  += yhat[mask].tolist()   
            losses.append(loss.item())
        
        
        return accuracy_score(ytrue, ypred), np.mean(losses)
    
def training(train, val, model, criterion, optimizer, scheduler, n_epochs,log_dir,checkpoint_path):
    losses = []
    writer = SummaryWriter(log_dir=log_dir)
    state = load_state(checkpoint_path, model, optimizer)
    with tqdm(total=n_epochs, position=0, leave=True) as pbar:
        for i in range(n_epochs):
            model = state.model.train()
            l = []
            for x, y in train:
                batch_size = y.data.size(1)
                x, y = x.to(device), y.to(device)
                
                h, c = state.model.init_hidden(batch_size)
                h, c = h.to(device), c.to(device)
                
                logits = state.model(x, h, c)
                loss = criterion(logits, y.view(-1))
                
                state.optimizer.zero_grad()
                loss.backward()            
                state.optimizer.step()
                l.append(loss.item())
                state.iteration += 1
          
            state.epoch +=1
            save_state(checkpoint_path, state)

            #scheduler.step()
            lo = np.mean(l)
            losses.append(lo)
            
            acc_train, loss_train = evaluate(train, model, criterion)
            acc_val, loss_val = evaluate(val, model, criterion)
            writer.add_scalar('Loss/train', np.round(loss_train, 4), i)
            writer.add_scalar('Loss/test',  np.round(loss_val, 4), i)
            writer.add_scalar('Acc/train',  np.round(acc_train, 4), i)
            writer.add_scalar('Acc/test',   np.round(acc_val, 4), i)

            pbar.set_description(f'Train: Loss: {np.round(loss_train, 4)} Acc= {np.round(acc_train, 4)}\tTest: Loss: {np.round(loss_val, 4)} Acc= {np.round(acc_val, 4)}') 
            pbar.update()
    return losses

    
def predict(sentences, model, words, tags):
  tokens_list = [[ words.get(token, False) for token in nltk.word_tokenize(sentence)] for sentence in sentences] 
  seq_lengths = list(map(lambda x: len(x),tokens_list))[::1]
  tensor = pad_sequence([torch.LongTensor(sentence) for sentence in tokens_list])
  x = pack_padded_sequence(tensor, seq_lengths, batch_first=False)
  with torch.no_grad():
    x = x.to(device)
    h, c = model.init_hidden(len(sentences))
    h, c = h.to(device), c.to(device)

    _, i = model(x, h, c).max(1)
  i = i.view(len(sentences), -1)
  return [ tags.getwords(x[:seq_lengths[idx]]) for idx, x in enumerate(i)] #np.array(tags.getwords(i)).reshape(len(sentences), -1)[0]
    
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
    

class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upos"], adding) for token in s]))
            

    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        x, y = self.sentences[ix]

        return x, y , len(x)


def collate(batch):
    """Collate using pad_sequence"""
    x, y, seq_lengths = zip(*batch)

    
    x = pad_sequence([torch.LongTensor(b) for b in x])
    y = pad_sequence([torch.LongTensor(b) for b in y])
    seq_lengths = sorted(seq_lengths)[::-1]
    x = pack_padded_sequence(x, seq_lengths, batch_first=False)
    # y = pack_padded_sequence(y, seq_lengths, batch_first=False) 
    
    return x, y


def load_data(batch_size=100):
    logging.info("Loading datasets...")
    ds = DS_url()
    words = Vocabulary(True)
    tags = Vocabulary(False)
    
    train_data = TaggingDataset(ds.train, words, tags, True)
    dev_data = TaggingDataset(ds.validation, words, tags, True)
    test_data = TaggingDataset(ds.test, words, tags, False)


    logging.info("Vocabulary size: %d", len(words))
    
    embeding_layer = None
    # embeding_layer = torch.nn.Embedding(len(words), 300)

    

    train_loader = DataLoader(train_data, collate_fn=collate, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, collate_fn=collate, batch_size=batch_size)
    test_loader = DataLoader(test_data, collate_fn=collate, batch_size=batch_size)
    
    return train_loader, dev_loader, test_loader, words, tags, embeding_layer



class Net(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, number_of_tags, num_layers=1, bidirectional=False, dropout=.8, embeding_layer=None):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        if embeding_layer is None:
            self.embeding = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.embeding = embeding_layer
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)


        self.decoder = nn.Sequential(
            nn.Linear(self.num_directions*hidden_size, number_of_tags),
            nn.LogSoftmax(dim=1)
        )



    def forward(self, x, h0, c0):
        x = torch.nn.utils.rnn.PackedSequence(self.embeding(x.data), x.batch_sizes)
        x, _ = self.lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        x = x.view(-1, x.shape[2])
        x = self.decoder(x)
        return x

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)
        return h0, c0
