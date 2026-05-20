import torch 
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import scipy


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
    def __init__(self, layers, lb, ub):
        self.register_buffer('lb', torch.tensor(lb, dtype = torch.float32))
        self.register_buffer('ub', torch.tensor(ub, dtype = torch.float32))
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

    def net_f_uv(self, x, t):

        x, t = self._ensure_grad()

        u, v, u_x, v_x = self.net_uv(x, t)
        u_t = torch.autograd.grad( u, t, grad_outputs = torch.ones_like(u) ,create_graph = True, retain_graph = True)[0]
        u_xx = torch.autograd.grad( u_x, x, grad_outputs = torch.ones_like(u_x) ,create_graph = True, retain_graph = True)[0]
        v_t = torch.autograd.grad( v, t, grad_outputs = torch.ones_like(v) ,create_graph = True, retain_graph = True)[0]
        v_xx = torch.autograd.grad( v_x, x, grad_outputs = torch.ones_like(v_x) ,create_graph = True, retain_graph = True)[0]

        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

        return f_u, f_v
    
    def compute_loss(self, x0, t0, u0, v0, x_lb, t_lb, x_ub, t_ub, x_f, t_f):

        u0_pred, v0_pred = self.forward(x0, t0)
        loss_ic = torch.mean((u0_pred - u0)**2) + torch.mean((v0_pred - v0)**2)

        # periodic conditions
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.net_uv(x_lb, t_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.net_uv(x_ub, t_ub)

        loss_bc = (
                    torch.mean((u_lb_pred - u_ub_pred)**2) +
                    torch.mean((v_lb_pred - v_ub_pred)**2) +
                    torch.mean((u_x_lb_pred - u_x_ub_pred)**2) +
                    torch.mean((v_x_lb_pred - v_x_ub_pred)**2)
                    )
        
        f_u_pred, f_v_pred = self.net_f_uv(x_f, t_f)
        loss_pde = torch.mean(f_u_pred**2) + torch.mean(f_v_pred**2)

        return loss_pde + loss_bc + loss_ic

    def predict(self, X_star):
        X_star_tensor = torch.tensor(X_star, dtype = torch.float32, device = device)
        x = X_star_tensor[:,0:1]
        t = X_star_tensor[:,1:2]

        self.eval()
        u_pred, v_pred = self.forward(x, t)
        u_f_pred, v_f_pred = self.net_f_uv(x, t)


        return (
            u_pred.cpu().detach().numpy(),
            v_pred.cpu().detach().numpy(),
            u_f_pred.cpu().detach().numpy(),
            v_f_pred.cpu().detach().numpy()
        )
def to_tensor(array):
    return torch.tensor(array, dtype = torch.float32, device = device)


if __name__ == '__main__':
    # seed
    torch.manual_seed(2005)
    np.random.seed(2005)
    # Adam
    AIter = 50000
    Adam_lr = 1e-3


    # L-BFGS
    Max_iter = 50000
    Max_eval = 50000
    LBFGS_lr = 1

    output_frequenry = 2000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layers = (2, 100, 100, 100, 100, 2)

    x_min = -5.0
    x_max = 5.0

    t_min = 0.0
    t_max = np.pi/2

    lb = [x_min, t_min]
    ub = [x_max, t_max]

    lb = np.array(lb)
    ub = np.array(ub)

    N0 = 50
    N_b = 50
    N_f = 20000

    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / 'main'/ 'Data'/ 'NLS.mat'
    data = scipy.io.loadmat(data_path)

    t = data['tt'].reshape(-1, 1)
    x = data['x'].reshape(-1, 1)
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x, t)
    X_star = np.concatenate([X.reshape(-1, 1), T.reshape(-1, 1)], axis = 1)

    u_star = Exact_u.T.reshape(-1, 1)
    v_star = Exact_v.T.reshape(-1, 1)
    h_star = Exact_h.T.reshape(-1, 1)

    idx_x = np.random.choice(x.shape[0], N0, replace = False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace = False)
    tb = t[idx_t, :]

    X_f = lb + (ub - lb) * np.random.rand(N_f, 2)
    
    X0 = np.concatenate([x0, 0*x0], 1)
    X_lb = np.concatenate([np.ones_like(tb) * lb[0], tb], 1)
    X_ub = np.concatenate([np.ones_like(tb) * ub[0], tb], 1)

    x0_pt = to_tensor(X0[:, 0:1])
    t0_pt = to_tensor(X0[:, 1:2])
    u0_pt = to_tensor(u0)
    v0_pt = to_tensor(v0)

    xlb_pt = to_tensor(X_lb[:, 0:1])
    tlb_pt = to_tensor(X_lb[:, 1:2])

    xub_pt = to_tensor(X_ub[:, 0:1])
    tub_pt = to_tensor(X_ub[:, 1:2])
    
    x_f_pt = to_tensor(X_f[:, 0:1])
    t_f_pt = to_tensor(X_f[:, 1:2])

    model = PINN(layers, lb, ub).to(device)
    loss_fn = lambda : model.compute_loss(x0_pt, t0_pt, u0_pt, v0_pt, xlb_pt, tlb_pt, xub_pt, tub_pt, x_f_pt, t_f_pt)

    print('Adam:')
    optimizer_adam = torch.optim.Adam(model.parameters(), lr = Adam_lr)




    print('finished')

    

