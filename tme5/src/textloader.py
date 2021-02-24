import sys
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
from typing import List
import torch

import nltk
import re

PAD_IX = 0
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = ''  # NULL CHARACTER
id2lettre[EOS_IX] = '|'
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les majuscules """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)


def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    result = [lettre2id[c] for c in normalize(s)]
    return result 


def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)



class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        self.sent_text = nltk.sent_tokenize(text)

    def __len__(self):
        return len(self.sent_text)

    def __getitem__(self, i):
        sent = torch.tensor(string2code(self.sent_text[i]), dtype=torch.int64)  
        return sent, len(sent)


def collate_fn(batch):
    _, length = zip(*batch)
    maxlen = max(length)+1

    result = torch.zeros(maxlen, len(batch), dtype=torch.int64).fill_(PAD_IX)
    mask = torch.zeros(maxlen, len(batch))
    
    for i, (seq, length) in enumerate(batch):
        result[:length, i] = seq
        result[length, i] = EOS_IX
        # mask[:length+1,i] += 1
    return result #, mask==1




if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=collate_fn, batch_size=3)
    data = next(iter(loader))
    # Longueur maximum
    assert data.shape == (7, 3)
    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    # les chaînes sont identiques
    assert test == " ".join([code2string(s).replace("|", "")
                             for s in data.t()])
