import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import mse, linear, Context


# Les données supervisées
x = torch.randn(50, 13, requires_grad=True)
y = torch.randn(50, 3)


# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

epsilon = 0.05


writer = SummaryWriter()

for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    y_hat = linear(x, w, b)
    loss = mse(y_hat, y)
    
    loss.backward()
    
    writer.add_scalar('Loss/train', loss, n_iter)
    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")
        
    # Calcul du backward (grad_w, grad_b)
    grad_x, grad_w, grad_b = x.grad, w.grad, b.grad
    
    with torch.no_grad():
        ## Mise à jour des paramètres du modèle
        w -=  epsilon * grad_w
        b -=  epsilon * grad_b
    
    x.grad.zero_()
    w.grad.zero_()
    b.grad.zero_()

