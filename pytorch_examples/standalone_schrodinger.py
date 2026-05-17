"""
================================================================================
 Physics-Informed Neural Network (PINN) 求解非线性薛定谔方程
================================================================================
 
 【什么是 PINN？】
 传统神经网络通过大量标注数据学习输入→输出的映射。但物理问题中，我们往往
 没有那么多数据。PINN 的核心思想是：把物理定律（偏微分方程）直接编码到损失
 函数中。网络不仅要拟合已有的观测数据，还必须"遵守"物理方程。
 
 损失函数 = 数据误差 + 初始条件误差 + 边界条件误差 + PDE残差误差
           (本例无数据)   (t=0 时刻)     (x=±5 边界)     (方程本身)
 
 【本例求解的方程：非线性薛定谔方程 (Nonlinear Schrödinger Equation)】
 
 复数形式：  i·h_t + 0.5·h_xx + |h|²·h = 0
 其中 h(t,x) = u(t,x) + i·v(t,x) 是复函数，i 是虚数单位
 
 分离实部和虚部后得到两个实方程：
   f_u:  u_t + 0.5·v_xx + (u²+v²)·v = 0
   f_v:  v_t - 0.5·u_xx - (u²+v²)·u = 0
 
 初始条件 (t=0)：   u(0,x) = 2·sech(x),   v(0,x) = 0
 周期边界 (x=±5)：  u(t,-5)=u(t,5),  v(t,-5)=v(t,5),
                    u_x(t,-5)=u_x(t,5),  v_x(t,-5)=v_x(t,5)
 定义域：x ∈ [-5, 5],  t ∈ [0, π/2]

 【PyTorch vs NumPy 核心区别】
  NumPy 数组：只存储数值，不具备梯度追踪能力
  PyTorch 张量 (Tensor)：可以追踪计算历史，自动求导 (autograd)
  类比：NumPy 是计算器，PyTorch 是带"录像"功能的计算器——
       它能记住每一步运算，然后反向计算出每个参数的梯度。
 
 【参考文献】
  Raissi et al., "Physics-informed neural networks: A deep learning framework
  for solving forward and inverse problems involving nonlinear PDEs", JCP 2019
================================================================================
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 导入依赖库
# ---------------------------------------------------------------------------
import torch                # PyTorch 主库：张量运算、自动求导
import torch.nn as nn       # 神经网络模块：各种层、损失函数
import numpy as np           # 数值计算库（你已熟悉）
import scipy.io              # 读取 .mat 格式数据文件（MATLAB 格式）
from scipy.interpolate import griddata  # 将散点数据插值到规则网格
from pyDOE import lhs        # 拉丁超立方采样：在空间中均匀采样（类似 np.random 但分布更均匀）
import time                  # 计时
import matplotlib
matplotlib.use('Agg')        # 使用非交互式后端，不弹窗，直接保存图片
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable  # 给 imshow 图加 colorbar

# 固定随机种子——保证每次运行结果一致，便于调试和复现
torch.manual_seed(1234)
np.random.seed(1234)

# 自动选择设备：有 GPU 用 GPU，没有用 CPU
# PyTorch 中，张量和模型都必须放在同一设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================================================
# 第1部分：定义全连接神经网络 (DNN)
# ===========================================================================

class DNN(nn.Module):
    """
    全连接深度神经网络（多层感知机 / MLP）
    
    【PyTorch 知识点：nn.Module】
      nn.Module 是所有神经网络模块的基类。继承它之后：
      - __init__ 中定义网络层（参数）
      - forward  中定义前向传播（计算图）
      - 参数自动注册，可以被 optimizer 追踪和更新
      
    【网络结构】
      输入层 (2个神经元: x, t)
        ↓  Linear + tanh
      隐藏层 (100个神经元)
        ↓  Linear + tanh
      隐藏层 (100个神经元)
        ↓  Linear + tanh
      隐藏层 (100个神经元)
        ↓  Linear + tanh
      隐藏层 (100个神经元)
        ↓  Linear（无激活函数，因为输出可以取任意实数）
      输出层 (2个神经元: u, v)
      
    【为什么最后一层不用激活函数？】
      tanh 输出范围是 (-1, 1)，但 u 和 v 可能是任意值。
      输出层不加激活函数，让网络自由输出任意实数。
    """
    
    def __init__(self, layers):
        """
        参数:
          layers: list，例如 [2, 100, 100, 100, 100, 2]
                  每个数字代表一层神经元的个数
                  第一个是输入维度(x,t两个变量)，最后一个是输出维度(u,v两个函数)
        """
        super().__init__()  # 必须调用父类 __init__
        
        # nn.ModuleList: PyTorch 的模块列表，内部模块会被自动追踪
        # 类比：普通的 Python list，但里面的 nn.Linear 会被自动注册
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            # nn.Linear(in, out): 全连接层
            #   内部包含权重矩阵 W (shape: [in, out]) 和偏置向量 b (shape: [out])
            #   计算: y = x @ W + b
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))
        
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化网络权重——这对训练收敛至关重要
        
        Xavier 初始化 (也叫 Glorot 初始化)：
          根据输入和输出维度自适应地缩放初始权重，
          使得信号在前向和反向传播中保持合适的方差。
          公式: std = sqrt(2 / (in_dim + out_dim))
        
        偏置初始化为 0。
        """
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)  # Xavier 正态分布初始化权重
            nn.init.zeros_(m.bias)             # 偏置置零
    
    def forward(self, X):
        """
        前向传播：输入 X，经过每一层计算得到输出。
        
        参数:
          X: PyTorch 张量，shape = (N, 2)，N 是样本数，2 是 (x, t)
        返回:
          输出张量，shape = (N, 2)，即每个样本的 [u, v] 预测值
        
        【PyTorch 知识点：forward 方法】
          不需要手动调用 model.forward(x)，而是直接 model(x)。
          PyTorch 的 nn.Module 会自动在 __call__ 中调用 forward，
          并在前后做一些 hook 处理和梯度追踪。
        """
        for i in range(len(self.linears) - 1):
            # 隐藏层: 线性变换 → tanh 激活
            X = torch.tanh(self.linears[i](X))
        # 输出层: 线性变换（无激活函数）
        return self.linears[-1](X)


# ===========================================================================
# 第2部分：物理信息神经网络 (PINN) 主体
# ===========================================================================

class PhysicsInformedNN(nn.Module):
    """
    PINN 的核心：把神经网络和物理方程绑定在一起。
    
    继承 nn.Module 的好处：
      - 网络的参数被自动追踪
      - 可以方便地 .to(device) 迁移设备
      - 可以 .train() / .eval() 切换模式
      - 可以直接传给 PyTorch 的优化器
    """
    
    def __init__(self, layers, lb, ub):
        """
        参数:
          layers: 网络结构，如 [2, 100, 100, 100, 100, 2]
          lb:     定义域下界 [x_min, t_min] = [-5.0, 0.0]
          ub:     定义域上界 [x_max, t_max] = [5.0, π/2]
        """
        super().__init__()
        
        # register_buffer: 把张量注册为模型的持久状态
        #   - 会随模型一起 .to(device) 迁移
        #   - 会随模型一起保存/加载 (state_dict)
        #   - 但不是可训练参数（requires_grad=False）
        self.register_buffer('lb', torch.tensor(lb, dtype=torch.float32))
        self.register_buffer('ub', torch.tensor(ub, dtype=torch.float32))
        self.dnn = DNN(layers)
    
    def forward(self, x, t):
        """
        基础前向传播：输入 (x, t)，输出 (u, v)。
        
        输入归一化：
          X_norm = 2*(X - lb)/(ub - lb) - 1
          作用：将输入从原始范围 [lb, ub] 映射到 [-1, 1]
          这是神经网络训练的常见技巧，能加速收敛、提高精度。
          
        参数:
          x: shape (N, 1) 空间坐标
          t: shape (N, 1) 时间坐标
        返回:
          u: shape (N, 1) 解的实部
          v: shape (N, 1) 解的虚部
        """
        # 按列拼接: (N,1) + (N,1) → (N,2)
        X = torch.cat([x, t], dim=1)
        # 归一化到 [-1, 1]
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        uv = self.dnn(X)
        # 切片: 第0列是 u，第1列是 v。保留第二维使其 shape 为 (N,1) 而非 (N,)
        return uv[:, 0:1], uv[:, 1:2]
    
    def _ensure_grad(self, x, t):
        """
        确保张量开启了梯度追踪。
        
        【PyTorch 核心概念：requires_grad】
          PyTorch 中每个张量都有一个 requires_grad 属性：
          - requires_grad=True: 对此张量的所有操作都会被记录，
            之后可以调用 .backward() 自动计算梯度。
          - requires_grad=False: 不记录操作，节省内存。
          
          clone().detach(): 创建一个"断开"计算图的新张量副本
          requires_grad_(True): 开启该张量的梯度追踪
        
        这样设计的原因：从外部传入的 x, t 可能没有开启 requires_grad，
        但我们又需要对其求导，所以创建一个"可导"的副本。
        """
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
        return x, t
    
    def net_uv(self, x, t):
        """
        计算 u, v 以及它们对空间的一阶导数 u_x, v_x。
        
        【PyTorch 核心概念：torch.autograd.grad —— 自动微分】
          这是 PINN 实现中最关键的 PyTorch 功能。
          
          torch.autograd.grad(output, input, ...):
            - output: 要对其求导的输出张量
            - input:  对哪个输入求导
            - grad_outputs: 梯度的"权重"，通常用全1张量（即直接求和）
            - create_graph=True:   为梯度再建计算图（以便求二阶导！）
            - retain_graph=True:   保留原计算图（以便后续再求其他导数）
            - allow_unused=True:   允许某些输入没有被使用
            
          返回: d(output)/d(input)，即 output 对 input 的偏导数
        
        【为什么需要 create_graph=True？】
          对 u 求 u_x 是一次求导。之后还需要对 u_x 求 u_xx（二阶导）。
          如果 create_graph=False，u_x 就只是一个数值，不能再对其求导。
          设为 True 意味着 u_x 的计算图被保留，可以继续求高阶导数。
        """
        x, t = self._ensure_grad(x, t)
        u, v = self.forward(x, t)
        
        # ∂u/∂x: u 对空间坐标 x 的偏导数
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),  # 等价于 sum(u)，对每个分量权重为 1
            create_graph=True,                 # 保留计算图，为二阶导做准备
            retain_graph=True                  # 保留计算图，后续还要用
        )[0]  # grad 返回的是元组 (grad,)，取第一个元素
        
        # ∂v/∂x
        v_x = torch.autograd.grad(v, x,
            torch.ones_like(v),
            create_graph=True, retain_graph=True)[0]
        
        return u, v, u_x, v_x
    
    def net_f_uv(self, x, t):
        """
        计算 PDE 残差（方程左边的值，理想情况应为 0）。
        
        薛定谔方程拆分后：
          f_u = u_t + 0.5·v_xx + (u²+v²)·v = 0
          f_v = v_t - 0.5·u_xx - (u²+v²)·u = 0
        
        如果网络预测完美，则 f_u 和 f_v 应该处处为 0。
        训练目标就是让 f_u 和 f_v 尽可能接近 0。
        
        这里需要计算：
          u_t, v_t:  一阶时间导数
          u_xx, v_xx: 二阶空间导数（对 u_x 再求一次导）
        """
        x, t = self._ensure_grad(x, t)
        u, v, u_x, v_x = self.net_uv(x, t)
        
        # 一阶时间导数: ∂u/∂t
        u_t = torch.autograd.grad(u, t,
            torch.ones_like(u),
            create_graph=True, retain_graph=True)[0]
        
        # 二阶空间导数: ∂²u/∂x²，即对 ∂u/∂x 再求一次 ∂/∂x
        u_xx = torch.autograd.grad(u_x, x,
            torch.ones_like(u_x),
            create_graph=True, retain_graph=True)[0]
        
        # ∂v/∂t
        v_t = torch.autograd.grad(v, t,
            torch.ones_like(v),
            create_graph=True, retain_graph=True)[0]
        
        # ∂²v/∂x²
        v_xx = torch.autograd.grad(v_x, x,
            torch.ones_like(v_x),
            create_graph=True, retain_graph=True)[0]
        
        # 计算 PDE 残差（如果网络完美预测，这些值都为 0）
        # u**2: 逐元素平方，等价于 u²
        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u
        
        return f_u, f_v
    
    def compute_loss(self, x0, t0, u0, v0, x_lb, t_lb, x_ub, t_ub, x_f, t_f):
        """
        计算总损失函数——这是 PINN 训练的核心。
        
        损失由三部分组成（对本例而言没有观测数据项）：
        
        【1. 初始条件损失 (Initial Condition Loss)】
          在 t=0 时刻，网络的预测值必须匹配给定的初始条件：
            u(0,x) = 2·sech(x),  v(0,x) = 0
          损失 = mean((u_pred - u_true)²) + mean((v_pred - v_true)²)
        
        【2. 边界条件损失 (Boundary Condition Loss)】
          周期边界：x=-5 和 x=5 处，函数值和导数值必须相等：
            u(t,-5)=u(t,5),  v(t,-5)=v(t,5)
            u_x(t,-5)=u_x(t,5),  v_x(t,-5)=v_x(t,5)
          损失 = mean((u_left - u_right)²) + 类似的 v, u_x, v_x 项
        
        【3. PDE 残差损失 (PDE Residual Loss)】
          在整个定义域内随机采样一些"配点"（collocation points），
          要求方程 f_u=0 和 f_v=0 在这些点上成立。
          损失 = mean(f_u²) + mean(f_v²)
        
        【参数说明】
          x0, t0: 初始条件采样点的 (x, t) 坐标（t 全是 0）
          u0, v0: 初始条件的真实值
          x_lb, t_lb: 左边界采样点 (x=-5)
          x_ub, t_ub: 右边界采样点 (x=5)
          x_f, t_f: 内部随机配点
        
        【PyTorch 知识点：torch.mean —— 对标 NumPy】
          torch.mean((a-b)**2) = 均方误差 (MSE)
          与 numpy 的 np.mean((a-b)**2) 用法相同，但返回的是带梯度的张量。
        """
        # --- 损失 1：初始条件 ---
        # net_uv 返回 (u, v, u_x, v_x)，这里只需要 u 和 v
        u0_pred, v0_pred, _, _ = self.net_uv(x0, t0)
        loss_ic = torch.mean((u0 - u0_pred) ** 2) + torch.mean((v0 - v0_pred) ** 2)
        
        # --- 损失 2：周期边界条件 ---
        # 左边界预测
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.net_uv(x_lb, t_lb)
        # 右边界预测
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.net_uv(x_ub, t_ub)
        # 周期条件要求左右相等，差值越接近 0 越好
        loss_bc = (torch.mean((u_lb_pred - u_ub_pred) ** 2) +     # u 值周期性
                   torch.mean((v_lb_pred - v_ub_pred) ** 2) +     # v 值周期性
                   torch.mean((u_x_lb_pred - u_x_ub_pred) ** 2) + # u 导数周期性
                   torch.mean((v_x_lb_pred - v_x_ub_pred) ** 2))  # v 导数周期性
        
        # --- 损失 3：PDE 残差 ---
        # 在内部随机采样点上，PDE 应成立（f_u=0, f_v=0）
        f_u_pred, f_v_pred = self.net_f_uv(x_f, t_f)
        loss_pde = torch.mean(f_u_pred ** 2) + torch.mean(f_v_pred ** 2)
        
        # 总损失 = 三部分直接相加
        # PINN 不设权重系数（某些问题中加权重效果更好，但这里不需要）
        return loss_ic + loss_bc + loss_pde
    
    def predict(self, X_star):
        """
        在给定坐标点上预测解的值。
        
        参数:
          X_star: NumPy 数组，shape (N, 2)，列分别是 [x, t]
        返回:
          u_pred, v_pred: 解的实部和虚部
          f_u_pred, f_v_pred: PDE 残差（评估预测质量）
        
        【PyTorch 知识点：.eval() 和 .detach()】
          model.eval(): 切换到评估模式。主要影响 Dropout 和 BatchNorm 等层。
                       （纯全连接网络影响不大，但这是好习惯）
          tensor.detach(): 从计算图中分离张量，不再追踪梯度。
          tensor.cpu():    将 GPU 张量搬回 CPU。
          tensor.numpy():  将 PyTorch 张量转为 NumPy 数组。
                           （注意：必须先 .detach() 再 .numpy()，否则报错）
        """
        # NumPy → PyTorch 张量（注意指定 dtype 和设备）
        X_star_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)
        x = X_star_tensor[:, 0:1]  # 第一列：空间坐标
        t = X_star_tensor[:, 1:2]  # 第二列：时间坐标
        
        self.eval()  # 评估模式
        u_pred, v_pred = self.forward(x, t)
        f_u_pred, f_v_pred = self.net_f_uv(x, t)
        
        # PyTorch 张量 → NumPy 数组
        return (u_pred.cpu().detach().numpy(),
                v_pred.cpu().detach().numpy(),
                f_u_pred.cpu().detach().numpy(),
                f_v_pred.cpu().detach().numpy())


# ===========================================================================
# 第3部分：主程序 —— 数据准备、训练、评估
# ===========================================================================

if __name__ == "__main__":
    
    # =======================================================================
    # 3.1 定义问题参数
    # =======================================================================
    
    # 定义域边界
    lb = np.array([-5.0, 0.0])       # 下界: [x_min, t_min]
    ub = np.array([5.0, np.pi / 2])  # 上界: [x_max, t_max]
    
    # 采样点数量
    N0 = 50      # 初始条件 (t=0) 上的采样点数
    N_b = 50     # 边界条件 (x=±5) 上的采样点数
    N_f = 20000  # 内部配点（collocation points）采样点数
    
    # 网络结构: [输入维, 隐藏层1, 隐藏层2, 隐藏层3, 隐藏层4, 输出维]
    layers = [2, 100, 100, 100, 100, 2]
    
    # =======================================================================
    # 3.2 加载参考数据（来自数值求解器的"精确解"，用于评估精度）
    # =======================================================================
    
    # scipy.io.loadmat: 读取 MATLAB 的 .mat 文件
    # 返回一个字典，键是变量名
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / 'main' / 'Data' / 'NLS.mat'
    data = scipy.io.loadmat(data_path)
    
    # 数据形状说明：
    #   'x':   空间网格点 (256 个点，从 -5 到 4.96)
    #   'tt':  时间网格点 (201 个点，从 0 到 π/2)
    #   'uu':  复数值解 (shape: 256×201)，来自传统数值方法（谱方法）
    t = data['tt'].flatten()[:, None]  # flatten → (201,) → [:,None] → (201,1)
    x = data['x'].flatten()[:, None]   # 同上，变为 (256,1)
    Exact = data['uu']                 # shape: (256, 201)
    Exact_u = np.real(Exact)           # 取实部
    Exact_v = np.imag(Exact)           # 取虚部
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)  # 振幅 |h|
    
    # =======================================================================
    # 3.3 构建坐标网格（用于最终评估和绘图）
    # =======================================================================
    
    # np.meshgrid: 生成二维坐标网格
    #   输入 x (256,), t (201,) → 输出两个 (256, 201) 的矩阵
    #   X[i,j] 是第 (i,j) 个点的 x 坐标，T[i,j] 是其 t 坐标
    X, T = np.meshgrid(x, t)
    
    # 将所有网格点的 (x,t) 坐标展平并堆叠
    # X.flatten() → (256*201,) → [:,None] → (51456,1)
    # hstack 后 → (51456, 2)，每行是 [x_i, t_i]
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    # 同样获取对应的精确解（展平为列向量）
    u_star = Exact_u.T.flatten()[:, None]  # (51456, 1)
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]
    
    # =======================================================================
    # 3.4 训练数据采样
    # =======================================================================
    
    # --- 初始条件采样点 (t=0) ---
    # 从空间网格中随机选 N0=50 个点
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]                           # 随机选的 x 坐标
    u0 = Exact_u[idx_x, 0:1]                   # 对应 t=0 时刻的精确 u
    v0 = Exact_v[idx_x, 0:1]                   # 对应 t=0 时刻的精确 v
    
    # --- 边界条件采样点 (x=±5) ---
    # 随机选 N_b=50 个时间点
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]                            # 随机选的时间坐标
    
    # --- 内部配点 (collocation points) ---
    # 用拉丁超立方采样 (LHS) 在定义域内均匀采样 N_f=20000 个点
    # lhs(维度, 点数): 比纯随机采样分布更均匀
    # 输出范围是 [0,1]，需要缩放到 [lb, ub]
    X_f = lb + (ub - lb) * lhs(2, N_f)
    
    # --- 构建完整的 (x,t) 坐标对 ---
    # 初始条件：(x0, t=0)
    # np.concatenate 将两个 (N,1) 数组拼成 (N,2)
    X0 = np.concatenate((x0, 0 * x0), 1)          # (50, 2): 第1列=x, 第2列=0
    
    # 左边界：(x=-5, tb)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (50, 2): x=-5, t=随机
    
    # 右边界：(x=5, tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (50, 2): x=5, t=随机
    
    # =======================================================================
    # 3.5 NumPy 数组 → PyTorch 张量
    # =======================================================================
    
    def to_tensor(arr):
        """将 NumPy 数组转为 PyTorch 张量，指定 float32 类型和设备"""
        return torch.tensor(arr, dtype=torch.float32, device=device)
    
    # 初始条件
    x0_pt = to_tensor(X0[:, 0:1])    # (50,1) x 坐标
    t0_pt = to_tensor(X0[:, 1:2])    # (50,1) t 坐标（全是 0）
    u0_pt = to_tensor(u0)            # (50,1) 精确 u
    v0_pt = to_tensor(v0)            # (50,1) 精确 v
    
    # 左边界
    x_lb_pt = to_tensor(X_lb[:, 0:1])
    t_lb_pt = to_tensor(X_lb[:, 1:2])
    
    # 右边界
    x_ub_pt = to_tensor(X_ub[:, 0:1])
    t_ub_pt = to_tensor(X_ub[:, 1:2])
    
    # 内部配点
    x_f_pt = to_tensor(X_f[:, 0:1])
    t_f_pt = to_tensor(X_f[:, 1:2])
    
    # =======================================================================
    # 3.6 创建模型
    # =======================================================================
    
    # 实例化 PINN 模型，并迁移到指定设备
    model = PhysicsInformedNN(layers, lb, ub).to(device)
    
    # 把 loss 计算封装成无参函数，方便复用
    # lambda 类似于 Python 的匿名函数，但这里捕捉了外部变量（闭包）
    loss_fn = lambda: model.compute_loss(
        x0_pt, t0_pt, u0_pt, v0_pt,        # 初始条件
        x_lb_pt, t_lb_pt,                   # 左边界
        x_ub_pt, t_ub_pt,                   # 右边界
        x_f_pt, t_f_pt                      # 内部配点
    )
    
    # =======================================================================
    # 3.7 阶段1：Adam 优化器 —— 快速找到较优区域
    # =======================================================================
    #
    # 【PyTorch 知识点：优化器 (Optimizer)】
    #   optimizer = torch.optim.XXX(model.parameters(), lr=...)
    #   - model.parameters(): 获取模型中所有可训练参数
    #   - lr (learning rate): 学习率，控制每步更新的幅度
    #
    #   训练循环三件套：
    #     optimizer.zero_grad()   # 清空之前的梯度（否则会累积！）
    #     loss.backward()         # 反向传播，计算所有参数的梯度
    #     optimizer.step()        # 根据梯度更新参数
    #
    # 【为什么 PINN 训练分两阶段？】
    #   Adam (自适应矩估计): 
    #     一阶优化器，自适应调整每个参数的学习率。
    #     优点：对初始值不敏感，收敛稳定。
    #     缺点：精度有限，后期震荡。
    #   L-BFGS (拟牛顿法):
    #     二阶优化器，利用曲率信息（近似海森矩阵）。
    #     优点：收敛精度极高。
    #     缺点：对初始值敏感，计算量大。
    #   组合策略：Adam 先大致定位 → L-BFGS 精细优化（PINN 文献的标准做法）
    
    # nIter = 50000
    nIter = 500
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Phase 1: Adam optimization")
    start_time = time.time()
    for it in range(nIter):
        optimizer_adam.zero_grad()    # 梯度清零
        loss = loss_fn()              # 前向传播，计算损失
        loss.backward()               # 反向传播，计算梯度
        optimizer_adam.step()         # 更新参数
        
        # 每 1000 步打印一次进度
        if it % 1000 == 0:
            elapsed = time.time() - start_time
            # loss.item(): 将只有一个元素的张量转为 Python 标量
            print(f'It: {it}, Loss: {loss.item():.3e}, Time: {elapsed:.2f}')
            start_time = time.time()
    
    # =======================================================================
    # 3.8 阶段2：L-BFGS 优化器 —— 精细收敛
    # =======================================================================
    
    print("Phase 2: L-BFGS optimization")
    
    # L-BFGS 需要闭包（closure）来计算损失
    # PyTorch 的 LBFGS 在内部会多次调用 closure 来做线搜索
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        # max_iter=50000,                                    # 最大迭代次数
        max_iter=500,                                    # 最大迭代次数
        # max_eval=50000,                                    # 最大函数评估次数
        max_eval=500,                                    # 最大函数评估次数
        history_size=50,                                   # 存储的历史梯度数
        line_search_fn='strong_wolfe',                     # 强 Wolfe 线搜索
        tolerance_change=1.0 * np.finfo(float).eps,        # 变化容忍度
    )
    
    def closure():
        """L-BFGS 每次评估时调用的闭包，返回损失值"""
        optimizer_lbfgs.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    
    optimizer_lbfgs.step(closure)
    
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.4f}')
    
    # =======================================================================
    # 3.9 评估：在全网格上预测并计算误差
    # =======================================================================
    
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    
    # 振幅 |h| = sqrt(u² + v²)
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    
    # 相对 L2 误差: ||pred - exact||₂ / ||exact||₂
    # np.linalg.norm(..., 2): L2 范数（欧几里得范数）
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print(f'Error u: {error_u:e}')  # :e 表示科学计数法
    print(f'Error v: {error_v:e}')
    print(f'Error h: {error_h:e}')
    
    # 将散点预测值插值到规则网格，便于绘图
    # griddata: 将不规则散点插值到规则网格
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')
    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')
    
    # =======================================================================
    # 3.10 可视化结果
    # =======================================================================
    
    # 收集用于标记训练点的坐标
    X0_plot = np.concatenate((x0, 0 * x0), 1)
    X_lb_plot = np.concatenate((0 * tb + lb[0], tb), 1)
    X_ub_plot = np.concatenate((0 * tb + ub[0], tb), 1)
    X_u_train = np.vstack([X0_plot, X_lb_plot, X_ub_plot])
    
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
    
    fig.savefig('schrodinger_result.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Plot saved to schrodinger_result.png')
