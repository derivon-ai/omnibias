# Faà di Bruno multi-layer jets

omnibias computes the exact activation derivative tower `sigma^(k)` in closed
form. For a single layer the pre-activation is affine, so higher derivatives are
trivial; the interesting case is **composition of many nonlinear layers**, which
is exactly Faà di Bruno's formula. Propagating a truncated Taylor **jet** along a
line `x(t) = x0 + t v` through a deep network gives the exact directional
derivatives `d^k/dt^k f(x0 + t v)` with no nested autodiff and no finite
differences. The runnable demo is
[`23_faa_di_bruno_jets.ipynb`](https://github.com/derivon-ai/omnibias/blob/main/notebooks/23_faa_di_bruno_jets.ipynb).

## Exact directional tower of a deep MLP

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.jax import mlp_jet, jet_to_tower
from omnibias.jax.activations import get_activation

k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
W1, b1 = 0.3 * jax.random.normal(k1, (16, 4)), jnp.zeros(16)
W2, b2 = 0.3 * jax.random.normal(k2, (16, 16)), jnp.zeros(16)
W3, b3 = 0.3 * jax.random.normal(k3, (1, 16)), jnp.zeros(1)
x0, v = jnp.zeros(4), jnp.ones(4) / 2.0    # line x(t) = x0 + t v

tanh = get_activation("tanh")
layers = [(W1, b1, tanh), (W2, b2, tanh), (W3, b3, None)]  # None = affine readout
tower = jet_to_tower(mlp_jet(x0, v, layers, order=8))       # (9, C) derivative tower
```

`tower[k]` is `d^k/dt^k f(x0 + t v)` at `t = 0`. It matches nested `jax.jacfwd`
to ~1e-12 at order 8, where central finite differences (which divide by `h^k`)
have already diverged by many orders of magnitude.

## How it works

Two equivalent kernels (the second is the production path):

- **Bell-polynomial form** (`omnibias.core.bell.faa_di_bruno_terms`): exact but
  enumerates partitions of `n`; used as a test oracle.
- **Shifted-power form**: `sigma(u(t)) = sum_k (sigma^(k)(u0)/k!) (u(t)-u0)^k`,
  built by truncated convolution in `O(n^2)`, numerically stable in float64.

The exact `sigma^(k)` come from the omnibias activation fast paths, so the whole
composition is exact. Riccati-class activations (`tanh`, `sigmoid`, `gaussian`,
`exp`, `sin`, ...) support every order; bounded activations raise a clear
`ValueError` if a too-deep jet is requested.

## Deep Hessian by polarization

A directional 2-jet gives `v^T H v`; the full Hessian of a scalar output follows
from the polarization identity, each entry one order-2 jet:

```python
def dir2(u):
    return jet_to_tower(mlp_jet(x0, u, layers, 2))[2, comp]   # u^T H u

# H[i, j] = 0.5 * (dir2(e_i + e_j) - dir2(e_i) - dir2(e_j))
```

This reconstructs `jax.hessian` to machine precision.

## Backends

`omnibias.torch.mlp_jet` is a bit-identical twin of `omnibias.jax.mlp_jet`; the
two agree to float64 precision (pinned cross-backend + golden regression tests).
