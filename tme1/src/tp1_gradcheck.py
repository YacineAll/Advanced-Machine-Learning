import torch
from tp1 import mse, linear

# Test du gradient de MSE
yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))



x = torch.randn(50, 13, requires_grad=True, dtype=torch.float64)
y = torch.randn(50, 3)

w = torch.randn(13, 3, requires_grad=True, dtype=torch.float64)
b = torch.randn(3,     requires_grad=True, dtype=torch.float64)



torch.autograd.gradcheck(linear, (x, w, b))
