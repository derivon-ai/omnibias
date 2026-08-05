# Equivariant features from exact operators

A key inductive bias for modelling the physical world is **symmetry**. omnibias's
exact differential geometry gives equivariant feature extractors for free:
intrinsic quantities are invariant, and tangent quantities are equivariant. The
runnable demo is
[`22_equivariant_operator_features.ipynb`](https://github.com/derivon-ai/omnibias/blob/main/notebooks/22_equivariant_operator_features.ipynb).

## Intrinsic features are E(3)-invariant

Apply a random rotation `R` and translation `t` to a chart, `phi_R = R*phi + t`.
The pullback metric and scalar curvature are unchanged (to machine precision):

```python
import torch
from omnibias.geometry import ChartSpec, ManifoldSpec
from omnibias.geometry.torch import ops as geo

torch.manual_seed(0)

# Unit-sphere chart phi(u, v) = (sin u cos v, sin u sin v, cos u): R^2 -> R^3.
def base(x):
    u, v = x[..., 0], x[..., 1]
    return torch.stack([u.sin() * v.cos(), u.sin() * v.sin(), u.cos()], dim=-1)

R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]   # random rotation
t = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)          # translation
phiR = lambda x: base(x) @ R.T + t

coords = torch.tensor([[1.0, 0.4], [0.8, 1.2]], dtype=torch.float64)
g  = geo.pullback_metric(coords, ChartSpec(phiR, 2, 3))          # == g(base)
sc = geo.scalar_curvature(coords, ManifoldSpec("m", 2,
        geo.metric_spec_from_chart(ChartSpec(phiR, 2, 3))))      # == sc(base) == 2
```

## The Jacobian is equivariant

Tangent/vector features transform *with* the rotation, `J(R*phi) = R * J(phi)`:

```python
import torch.func as tf
J0 = tf.vmap(tf.jacfwd(base))(coords)
J1 = tf.vmap(tf.jacfwd(phiR))(coords)        # == einsum("ij,bjk->bik", R, J0)
```

## What omnibias already gives you — and what is deferred

- **Permutation equivariance / antisymmetry** for many-body systems (electrons,
  molecules) lives in `omnibias-ferminet`.
- **Diffeomorphism covariance** is built into `omnibias-geometry`'s exterior
  calculus and tensor operators.
- Exact `grad`, `div`, Laplacian, `laplace_beltrami`, and curvature are
  **equivariant feature extractors** to drop into geometric / graph networks:
  scalar outputs are E(3)-invariant, vector/tensor outputs equivariant.

**Deferred (different axis):** full **SE(3)/E(3) steerable** layers built from
SO(3) irreps, spherical harmonics, and Clebsch-Gordan tensor products (the `e3nn`
family). That is representation theory rather than differential calculus, and
would be a separate dependency/workstream rather than an omnibias primitive.
