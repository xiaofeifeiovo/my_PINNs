import torch 
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(2005)
np.random.seed(2005)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
layers = (2, 100, 100, 100, 100, 2)

x_min = -5.0
x_max = 5.0

t_min = 0.0
t_max = np.pi/2

lb = [x_min, t_min]
ub = [x_max, t_max]

class Dnn(torch.nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            self.linears.append(torch.nn.Linear(layers[i], layers[i+1]))

        self._init_weights()

    def _init_weights(self):
        for i in self.linears:
            torch.nn.init.xavier_normal_(i.weight)
            torch.nn.init.zeros_(i.bias)

    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = torch.tanh(self.linears[i](X))

        return self.linears[-1](X)
    
class PINN(torch.nn.Module):
    def __init__(self, layers, lb, up):
        self.register_buffer('lb', torch.tensor(lb, dtype = torch.float32))
        self.register_buffer('up', torch.tensor(up, dtype = torch.float32))
        self.dnn = Dnn(layers)

    def forward(self, x, t):
        X = torch.cat([x,t], dim = 1)
        X = 2 * (X - self.lb) / (self.ub - self.lb) -1.0
        uv = self.dnn(X)
        return uv[: , 0:1], uv[: , 1:2]
    
    def _ensure_grad(self, x, t):
        if not x.requires_gard():
            x = x.clone().detach().requires_grad_(True)
        if not t.requires_gard():
            t = t.clone().detach().requires_grad_(True)
        return x , t
    
    def net_uv(self, x, t):
        x, t = self._ensure_grad(x, t)
        u, v = self.forward(x, t)

        u_x = torch.autograd.grad( u, x, grad_outputs = torch.ones_like(u) ,create_graph = True, retain_graph = True)[0]
        v_x = torch.autograd.grad( v, x, grad_outputs = torch.ones_like(v) ,create_graph = True, retain_graph = True)[0]

        return u, v, u_x, v_x
