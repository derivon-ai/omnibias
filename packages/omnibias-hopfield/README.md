# omnibias-hopfield

**Status: Alpha (0.1.0a1).**

Modern [Hopfield networks](https://arxiv.org/abs/2008.02217) and scaled
dot-product attention are the same operator: retrieval is
`X^T softmax(β X ξ)`, i.e. attention with `Q = ξ`, `K = V = X`.

This package implements that kernel on **PyTorch** and **JAX** with
**closed-form** log-sum-exp derivatives (no autodiff in the implementations):

| Function | Meaning |
|----------|---------|
| `logsumexp_value` | `lse(β, a) = β⁻¹ log Σᵢ exp(β aᵢ)` |
| `logsumexp_jacobian` | `∇ lse(β, a) = softmax(β a)` |
| `logsumexp_hessian` | `β (diag(p) − p pᵀ)` |

Hopfield energy includes the `½ M²` constant (`M = maxᵢ ‖Xᵢ‖`) so it
matches the standard Ramsauer et al. energy landscape.

## Quick start

```python
import torch
from omnibias.hopfield.torch import ops as hopfield

X = torch.randn(8, 64)          # 8 stored patterns, dim 64
xi = torch.randn(64)            # query / state
xi_new = hopfield.modern_hopfield_retrieve(xi, X, beta=4.0)
E = hopfield.hopfield_energy(xi, X, beta=4.0)

# Multi-query attention (modern_hopfield_retrieve is K=V, single-query)
Q = torch.randn(3, 64)
out = hopfield.attention(Q, X, X, beta=4.0)
```

```python
import jax.numpy as jnp
from omnibias.hopfield.jax import ops as hopfield

X = jnp.ones((8, 64))
a = jnp.linspace(-1.0, 1.0, 8)
p = hopfield.logsumexp_jacobian(a, beta=2.0)
H = hopfield.logsumexp_hessian(a, beta=2.0)
```

## Install

```bash
pip install omnibias-hopfield[torch]   # or [jax], [all]
```

## Public API

`omnibias.hopfield.torch.ops` and `omnibias.hopfield.jax.ops` export:

- `softmax`, `logsumexp_value`, `logsumexp_jacobian`, `logsumexp_hessian`
- `modern_hopfield_retrieve`, `hopfield_energy`, `attention`

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
