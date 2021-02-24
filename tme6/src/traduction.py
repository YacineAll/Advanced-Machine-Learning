import logging
import os

import torch
import unicodedata
import string
import nltk


import sentencepiece as spm

import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List

import time
import re
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(level=logging.INFO)

def create_vocab(vocab_size=1000):

    try:
        os.rmdir("./language_model")
    except :
        pass
    os.makedirs("language_model/", exist_ok=True)

    data = pd.read_csv('./data/dataset.csv', sep=";", header=None)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(data[0].values),
        model_prefix='./language_model/en',
        vocab_size =vocab_size,
        unk_id=5,
        pad_id=3,
    )
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(data[1].values),
        model_prefix='./language_model/fr',
        vocab_size =vocab_size,
        unk_id=5,
        pad_id=3,
    )

# create_vocab()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    modelorg  = spm.SentencePieceProcessor(model_file='./language_model/en.model', add_bos=True, add_eos=True)
    modeldest = spm.SentencePieceProcessor(model_file='./language_model/fr.model', add_bos=True, add_eos=True)
except OSError:
    create_vocab()
    modelorg  = spm.SentencePieceProcessor(model_file='./language_model/en.model', add_bos=True, add_eos=True)
    modeldest = spm.SentencePieceProcessor(model_file='./language_model/fr.model', add_bos=True, add_eos=True)



def normalize(s):
    return re.sub(' +', ' ', "".join(c if c in string.ascii_letters else " "
                                     for c in unicodedata.normalize('NFD', s.lower().strip())
                                     if c in string.ascii_letters+" "+string.punctuation)).strip()

class SPMTradDataset():
    def __init__(self, data, modelorg, modeldest, max_len=10):
        self.modelorg  = modelorg
        self.modeldest = modeldest
        
        
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s) < 1:
                continue
            orig, dest = map(normalize, s.split("\t")[:2])
            if len(orig) > max_len:
                continue
                
            
            encodedorg  = torch.tensor(self.modelorg.encode(orig, out_type=int))
            encodeddest = torch.tensor(self.modeldest.encode(dest, out_type=int))
            self.sentences.append((encodedorg, encodeddest))
            
    def __len__(self): return len(self.sentences)
    def __getitem__(self, i): return self.sentences[i]


def collate(batch):
    orig, dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig, padding_value=modelorg.pad_id()), o_len, pad_sequence(dest, padding_value=modeldest.pad_id()), d_len



def mode_contraint(y, decoder, criterion, enc_outputs, hidden_code, target_length, batch_size, sos):
    dec_inputs = torch.LongTensor([[sos]*batch_size]).to(device)
    hidden = hidden_code
    loss = 0
    for i in range(target_length):
        # logits, hidden = decoder(dec_inputs, hidden)
        logits, hidden = decoder(dec_inputs, enc_outputs, hidden)
        loss += criterion(logits, y[i])
        dec_inputs = y[i].unsqueeze(0)
    return loss

def mode_non_contraint(y, decoder, criterion, enc_outputs, hidden_code, target_length, batch_size, sos):
    dec_inputs = torch.LongTensor([[sos]*batch_size]).to(device)
    hidden = hidden_code
    loss = 0
    for i in range(target_length):
        # logits, hidden = decoder(dec_inputs, hidden)
        logits, hidden = decoder(dec_inputs, enc_outputs, hidden)
        loss += criterion(logits, y[i])
        dec_inputs = torch.distributions.Categorical(logits).sample().unsqueeze(0)
                
    return loss

def train(train, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, n_epochs, sos, scheduler=None, epsilon=1):
    losses = []
    with tqdm(total=n_epochs, position=0, leave=True) as pbar:
        for _ in range(n_epochs):
            encoder = encoder.train()
            decoder = decoder.train()
            
            l = []
            for x, lenx, y, leny in train:
                batch_size = x.data.size(1)
                x, y = x.to(device), y.to(device)
                
                output_code, hidden_code = encoder(x, encoder.init_hidden(batch_size))
                
                if np.random.random() < epsilon:
                    loss = mode_contraint(y, decoder, criterion, output_code, hidden_code, max(leny), batch_size, sos)
                else:
                    loss = mode_non_contraint(y, decoder, criterion, output_code, hidden_code, max(leny), batch_size, sos)
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                
                l.append(loss.item()/max(leny))
                
                epsilon *= 0.999
            
            #scheduler.step()
            lo = np.mean(l)
            losses.append(lo)
            
            pbar.set_description(f'Train: Loss: {np.round(lo, 4)}') 
            pbar.update()
    return losses

def get_translate(sentence, encoder, decoder, modeldest, modelorg, maxlen):
    x = torch.LongTensor(modelorg.encode(sentence)).to(device).view(-1,1) 

    start = torch.LongTensor([[modeldest.bos_id()]*1]).to(device)
    l = []
    with torch.no_grad():
        _, hidden = encoder(x, encoder.init_hidden(1))
        for _ in range(maxlen):
            logits, hidden = decoder(start, hidden)
            start = logits.argmax(1).unsqueeze(0)
            word = start.view(-1).item()
            l.append(start.view(-1).item())
            if word == modeldest.eos_id():
                break
    return modeldest.decode(l) 
        

def get_data(file, max_len, batch_size, train_size=.8):
    with open(file) as f:
        lines = f.readlines()
    lines = [lines[x] for x in torch.randperm(len(lines))]
    idxTrain = int(train_size*len(lines))

    datatrain = SPMTradDataset(
            data="".join(lines[:idxTrain]),
            modelorg=modelorg,
            modeldest=modeldest,
            max_len=max_len,
    )

    datatest = SPMTradDataset(
            data="".join(lines[idxTrain:]),
            modelorg=modelorg,
            modeldest=modeldest,
            max_len=max_len,
    )

    train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(datatest,  collate_fn=collate, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader 


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers=1, bidirectional=False, dropout=.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1


        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)

    def forward(self, x, h0):
        embedded = self.embedding(x)
        output = embedded
        output, hidden = self.gru(output, h0)
        return output, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(device)
        return h0


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, number_of_dest_vocab, num_layers=1, bidirectional=False, dropout=.0):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1



        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.attn = nn.Linear(hidden_size*self.num_directions, hidden_size*self.num_directions )

        self.classifier = nn.Sequential(
            nn.Linear(self.num_directions*hidden_size*2, number_of_dest_vocab),
            nn.LogSoftmax(dim=1)
        ) 


    def forward(self, dec_inputs, enc_outputs, h0):
        
        dec_output, dec_hidden = self.gru(self.relu(self.embedding(dec_inputs)), h0)
        
        context = self.get_vector_context(dec_output, enc_outputs)
        
        out = self.classifier(torch.cat((dec_output.squeeze(0), context), 1))

        return out, dec_hidden

    def get_vector_context(self, dec_output, enc_outputs):
        """

        Args:
            dec_output  : [n_step(=1), batch_size(=N), num_directions(=1) * n_hidden]
            enc_outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        """
        n_step = len(enc_outputs)
        attn_scores = []
        
        for i in range(n_step):
            score = self.attn(enc_outputs[i])
            alpha = torch.bmm(dec_output.permute(1,0,-1), score.unsqueeze(-1))
            attn_scores.append(alpha) 

        scores = torch.softmax(torch.cat(attn_scores, dim=1).squeeze(-1), dim=1)
        out = torch.bmm(scores.unsqueeze(-1).permute(0,-1,1), enc_outputs.permute(1,0, -1))

        return out.squeeze(1) 

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(device)
        return h0


class AttnModule(nn.Module):
    def __init__(self, n_hidden):
        super(AttnModule, self).__init__()
        self.attn = nn.Linear(n_hidden, n_hidden)

    def forward(self, dec_output, enc_outputs):
        """

        Args:
            dec_output  : [n_step(=1), batch_size(=N), num_directions(=1) * n_hidden]
            enc_outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        """
        n_step = len(enc_outputs)
        attn_scores = []
        
        for i in range(n_step):
            score = self.attn(enc_outputs[i])
            alpha = torch.bmm(dec_output.permute(1,0,-1), score.unsqueeze(-1))
            attn_scores.append(alpha) 

        scores = torch.softmax(torch.cat(attn_scores, dim=1).squeeze(-1), dim=1)
        out = torch.bmm(scores.unsqueeze(-1).permute(0,-1,1), enc_outputs.permute(1,0, -1))

        return out.squeeze(1) 



def main():
    FILE = "data/en-fra.txt"

    writer = SummaryWriter("runs/tag-"+time.asctime())