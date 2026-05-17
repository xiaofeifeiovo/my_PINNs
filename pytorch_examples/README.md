# PyTorch PINNs Examples

This folder contains examples of Physics-Informed Neural Networks (PINNs) implemented in PyTorch,
as recommended by the [original PINNs repository](https://github.com/maziarraissi/PINNs).

## Two Approaches

### 1. Standalone Pure PyTorch (`standalone_schrodinger.py`)

A self-contained implementation that only requires `torch`, `numpy`, `scipy`, `matplotlib`.
No extra framework dependencies. Direct port of the original TensorFlow v1 code to PyTorch.

```bash
pip install torch numpy scipy matplotlib pyDOE
python standalone_schrodinger.py
```

### 2. Using `pinnstorch` Framework (Recommended for production)

Uses the official [pinns-torch](https://github.com/rezaakb/pinns-torch) package
with PyTorch Lightning + Hydra. Supports GPU, CUDA Graphs, JIT compilation, with up to 9x speedup.

```bash
pip install pinnstorch
python schrodinger/train.py
python burgers/train.py
```

## Key Differences from TensorFlow v1

| TF v1 (original) | PyTorch (new) |
|---|---|
| `tf.placeholder` | `torch.Tensor` with `requires_grad_()` |
| `tf.gradients(u, x)[0]` | `torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]` |
| `tf.train.AdamOptimizer` | `torch.optim.Adam` |
| `ScipyOptimizerInterface` (L-BFGS-B) | `torch.optim.LBFGS` |
| Manual training loop | PyTorch Lightning `Trainer` |
| Hardcoded config | Hydra YAML config |
