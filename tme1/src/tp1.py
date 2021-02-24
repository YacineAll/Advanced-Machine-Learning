# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    
    def save_for_backward(self, *args):
        self._saved_tensors = args
    
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    
    @staticmethod
    def forward(ctx, yhat, y):
        loss = (1/len(y))*torch.sum(torch.square(yhat - y))
        ctx.save_for_backward(yhat, y)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        # Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y  = ctx.saved_tensors 
        q = len(y)
        return grad_output*(2/q) * (yhat-y), grad_output*-(2/q) * (yhat-y)

mse = MSE.apply

#  TODO:  Implémenter la fonction Linear(X, W, b)

class Linear(Function):
    """Début d'implementation de la fonction MSE"""
    
    @staticmethod
    def forward(ctx,X, W, b):
        ctx.save_for_backward(X, W)
        return torch.add(torch.mm(X,W), b)
         
    
    @staticmethod
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        grad_x =  grad_output @ W.T 
        grad_w = X.T @ grad_output   
        grad_b = grad_output.sum(0)
        return grad_x, grad_w, grad_b


linear = Linear.apply