import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from pathlib import Path
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        super().__init__()
        self.register_buffer('lb', torch.tensor(lb, dtype = torch.float32))
        self.register_buffer('ub', torch.tensor(ub, dtype = torch.float32))
        self.dnn = Dnn(layers)

    def forward(self, x, t):
        X = torch.cat([x,t], dim = 1)
        X = 2 * (X - self.lb) / (self.ub - self.lb) -1.0
        uv = self.dnn(X)
        return uv[: , 0:1], uv[: , 1:2]
    
    def _ensure_grad(self, x, t):
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
        return x , t
    
    def net_uv(self, x, t):
        x, t = self._ensure_grad(x, t)
        u, v = self.forward(x, t)

        u_x = torch.autograd.grad( u, x, grad_outputs = torch.ones_like(u) ,create_graph = True, retain_graph = True)[0]
        v_x = torch.autograd.grad( v, x, grad_outputs = torch.ones_like(v) ,create_graph = True, retain_graph = True)[0]

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):

        x, t = self._ensure_grad(x, t)

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

    output_frequency = 2000

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

    start_time = time.time()
    for i in range(AIter):
        optimizer_adam.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer_adam.step()

        if i % output_frequency ==0:
            elapsed = time.time() - start_time
            print(f'Iteration:{i},loss = {loss.item()} consumed time:{elapsed} \n')
            start_time = time.time()


    print('L-BFGS')
    optimizer_LBFGS = torch.optim.LBFGS(
        model.parameters(),
        lr = LBFGS_lr,
        max_iter = Max_iter,
        max_eval = Max_eval,
        line_search_fn = 'strong_wolfe',
        tolerance_change = np.finfo(float).eps
    )

    def closure():
        optimizer_LBFGS.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    
    optimizer_LBFGS.step(closure)
    elapsed = time.time() - start_time
    print(f'Train time: f{elapsed}')

    print('End training')

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)

    print(f'Error u:{error_u:e}')
    print(f'Error v:{error_v:e}')
    print(f'Error h:{error_h:e}')

    U_pred = u_pred.reshape(X.shape)
    V_pred = v_pred.reshape(X.shape)
    H_pred = h_pred.reshape(X.shape)
    FU_pred = f_u_pred.reshape(X.shape)
    FV_pred = f_v_pred.reshape(X.shape)

    X_u_train = np.concatenate([X0, X_lb, X_ub], 0)

    
    # 创建图像
    fig = plt.figure(figsize=(7.0, 4.0))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # ---- 上半部分：|h(t,x)| 的全景热力图 ----
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    # imshow 画二维热力图，横轴是 t，纵轴是 x
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],  # 坐标轴范围
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    # 在热力图上叠加训练数据点（黑色 ×）
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx',
            label=f'Data ({X_u_train.shape[0]} points)',
            markersize=4, clip_on=False)
    
    # 标注 t=0.75, 1.00, 1.25 的截面线
    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[75] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[100] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[125] * np.ones((2, 1)), line, 'k--', linewidth=1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title('$|h(t,x)|$', fontsize=10)
    
    # ---- 下半部分：三个时间截面的预测 vs 精确解对比 ----
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    for i, idx in enumerate([75, 100, 125]):
        ax = plt.subplot(gs1[0, i])
        ax.plot(x, Exact_h[:, idx], 'b-', linewidth=2, label='Exact')
        ax.plot(x, H_pred[idx, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')
        ax.set_title(f'$t = {t[idx, 0]:.2f}$', fontsize=10)
        ax.axis('square')
        ax.set_xlim([-5.1, 5.1])
        ax.set_ylim([-0.1, 5.1])
        if i == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8),
                      ncol=5, frameon=False)
    
    output_path = script_dir / 'schrodinger_result.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved to {output_path}')


