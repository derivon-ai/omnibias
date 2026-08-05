# PINN: heat equation

`omnibias.torch.architectures.PINNHeat` is a closed-form-Laplacian
PINN reference implementation that solves the 1D heat equation with
homogeneous Dirichlet boundary conditions.

The full runnable example is at
[`packages/omnibias-torch/examples/pinn_heat.py`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/examples/pinn_heat.py).

```python
import torch
from omnibias.torch.architectures import PINNHeat

net = PINNHeat(hidden=64, base="softplus", alpha=0.1)

# Collocation points: x and t are matching (B,) tensors.
x = torch.linspace(0.0, 1.0, 64)
t = torch.linspace(0.0, 0.1, 64)

# The forward pass returns the field and the PDE residual u_t - alpha * u_xx,
# whose target is zero. Both derivatives are closed form.
u, residual = net(x, t)
loss_pde = (residual**2).mean()
```

The Laplacian is closed-form (no autograd through the activation), so
training is bit-stable and ~3x faster per iteration than an autograd
PINN of the same width.

See `docs/theory.md` for the closed-form Laplacian derivation.
