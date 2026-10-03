# omnibias

**Differentiable derivative primitives for Physics-Informed Neural Networks.**

Compute high-order activation derivatives and propagate Taylor jets through
supported neural networks without nesting spatial autodiff at every order.
The results remain differentiable with respect to network parameters, so PDE
residuals can train with ordinary optimizers.

```bash
pip install omnibias-torch
# Alternative backends: omnibias-jax or omnibias-keras
```

```python
import torch
from omnibias.torch.jet import jet_to_tower, mlp_jet

weights = torch.tensor([[0.7], [-0.4]], requires_grad=True)
bias = torch.tensor([0.1, 0.2], requires_grad=True)
readout = torch.tensor([[1.0, -0.5]], requires_grad=True)
layers = [(weights, bias, "tanh"), (readout, None, None)]
x = torch.tensor([0.3])
tower = jet_to_tower(mlp_jet(x, torch.ones_like(x), layers, order=4))
u, du, d2u, d3u, d4u = tower.unbind(0)
loss = (d4u - 24.0).square().mean()
loss.backward()
assert weights.grad is not None
```

The example computes a fourth spatial derivative and its parameter gradient.
The [PINN guide](docs/pinn.md) adds collocation points, boundary conditions and
an optimizer step; the [derivative guide](docs/derivatives.md) covers mixed
partials and JAX.

## Scope

The monorepo contains 16 reusable [primitive packages](docs/packages.md).
The primary PINN path is `core → torch/jax → fields`, with curvature and
partition primitives available when an application needs them. Keras supports
the activation/operator layer surface. Solvers and other consumer products
live in sibling repositories under `../omnibias_projects/`; the PINN consumer
is `../omnibias_projects/omnibias-pinn/`.

Closed-form formulas remove repeated spatial autodiff for supported layers.
They do not eliminate floating-point error, the number of requested mixed
partials, difficult optimization or sampling error. Read the
[guarantees](docs/guarantees.md) before making accuracy or performance claims.

## Development

```bash
uv sync --all-packages --group docs
uv run pytest packages/omnibias-core/tests packages/omnibias-torch/tests packages/omnibias-jax/tests tests -q
uv run ruff check packages tests
uv run mkdocs build --strict
```

[Contributing](CONTRIBUTING.md) · [Agent instructions](AGENTS.md) ·
[Licensing](LICENSING.md)
