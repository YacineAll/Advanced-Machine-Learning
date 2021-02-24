from textloader import code2string, string2code, id2lettre, lettre2id, EOS_IX
import torch
import numpy as np


# def generate(rnn, emb, decoder, eos, start="", maxlen=200):
# #     #    Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
#     pass

def generate(rnn, device, eos, start="", maxlen=200):
    x = torch.tensor(string2code(start)).unsqueeze(-1).to(device)
    h = [ v.to(device) for v in rnn.initHidden(1)]
    l = [lettre2id[start]]
    rnn.eval()
    with torch.no_grad():
        for _ in range(maxlen):
            h = rnn(x, *h)
            d = rnn.decode(h[0])
            probs = torch.exp(d)
            start = torch.distributions.categorical.Categorical(probs).sample()
            l.append(start.item())
            if start.item() == eos:
                break
            start = start.unsqueeze(-1).to(device)
    return code2string(l)


# def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
#     pass

def generate_beam(model, eos, device, k, start="", maxlen=200):
    def beam_search_decoder(predictions, top_k = 3):
        output_sequences = [([], 0)]
        for probs_idx in predictions:
            new_sequences = []
            for old_seq, old_score in output_sequences:
                for p, char_index in zip(*probs_idx):
                    new_seq = old_seq + [char_index.item()]
                    new_score = old_score + np.log(p.item())
                    new_sequences.append((new_seq, new_score))
            output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
            output_sequences = output_sequences[:top_k]
        return output_sequences

    x = torch.tensor(string2code(start)).unsqueeze(-1).to(device)
    h = [ v.to(device) for v in model.initHidden(1)]
    l = [lettre2id[start]]
    model.eval()
    predictions = []
    p = p_nucleus(model.decode, k)
    with torch.no_grad():
        for _ in range(maxlen):
            h = model(x, *h)
            probs = p(h[0])
            predictions.append(probs)
            start = beam_search_decoder(predictions, top_k = 1)[0][0][-1]
            l.append(start)
            if start == eos:
                break
            start = torch.tensor(start).unsqueeze(-1).to(device)
    return code2string(l)

# p_nucleus
def p_nucleus(decoder, k: int):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        decoder: renvoie les logits étant donné l'état du RNN
        k (int): top k element
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
            h (torch.Tensor): L'état à décoder
        """
        logits  = torch.exp(decoder(h)).squeeze(0)
        probs, idx = torch.topk(logits, k)
        probs  = torch.softmax(probs, dim=0)

        return probs, idx

    return compute
