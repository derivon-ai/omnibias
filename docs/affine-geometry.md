# Affine equality geometry

An equality constraint has a defining residual, a geometric solution set and a
hard membership predicate. These are different outputs. The new geometry API
preserves exact membership for rational data and supplies differentiable
Euclidean projection for numerical tensors.

For full-row-rank constraints `A @ x = b`, projection minimizes
`0.5 * ||y - x||²` over feasible `y`. Its seed derivative is the tangent
projector. The gradient of the half squared distance is `x - projected`.
An ordinary derivative of the hard membership indicator is not used in training.

## Exact rational operations

```python
from fractions import Fraction
from omnibias.core import exact_affine_geometry

plane = exact_affine_geometry([[1, 1]], [1])
point = plane.project([2, 0])
assert point == (Fraction(3, 2), Fraction(-1, 2))
assert plane.contains(point)
assert plane.squared_distance([2, 0]) == Fraction(1, 2)
```

Inputs are integers or `Fraction`, excluding booleans and floats. Consistent
redundant equations are supported. An inconsistent system has
`consistent=False`; projection and distance raise
`InconsistentAffineConstraints`. `AffineBudget` bounds construction and query
work, storage slots and rational coefficient sizes; exhausted budgets raise
`AffineBudgetExceeded`. Exact witness identities are checked in Python. This
is not a theorem-prover build or a certificate about an uncertain measured input.

Empty constraints require an explicit `dimension` and give identity projection.
The stored `anchor` is the minimum-norm feasible point; `tangent_projector`
projects onto the homogeneous kernel.

## Differentiable tensor operations

```python
import torch
from omnibias.torch import affine_project

x = torch.tensor([[2.0, 0.0]], dtype=torch.float64, requires_grad=True)
A = torch.tensor([[1.0, 1.0]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
result = affine_project(x, A, b)
assert bool(result.valid.all())
result.half_squared_distance.sum().backward()
assert torch.allclose(x.grad, x.detach() - result.projected.detach())
```

The paired JAX function has the same argument and result names:

```python
import jax
import jax.numpy as jnp
from omnibias.jax import affine_project

project = jax.jit(affine_project)
result = project(jnp.array([2.0, 0.0]), jnp.array([[1.0, 1.0]]), jnp.array([1.0]))
assert result.projected.shape == (2,)
```

`x` has shape `(..., d)`, `A` has shared shape `(m, d)` and `b` has shape `(m,)`,
with `0 <= m <= d` and positive `d`. Supported dtypes are float32 and float64.
For varying constraint matrices, use JAX `vmap` or an explicit Torch batch loop.
The result contains:

| Field | Meaning |
| --- | --- |
| `projected` | Numerical closest feasible point |
| `correction` | `projected - x` |
| `half_squared_distance` | Half the squared Euclidean correction norm |
| `valid` | Stopped-gradient numerical validity per query |
| `rcond` | Stopped-gradient singular-value ratio of the row-scaled matrix |

Reduced QR and triangular solves avoid forming and inverting `A @ A.T`.
Gradients remain connected to `x`, `A` and `b`, including second derivatives
within a regular fixed-rank region. Empty constraints give identity projection;
square full-rank constraints give a point independent of the seed.

Each row is divided by its largest absolute entry before a numerical rank
screen. The default `rank_rcond` cutoff is `max(m,d) * dtype_epsilon`; callers
can request a stricter positive cutoff below one. A small accepted ratio still
warns of sensitivity through the reported diagnostic; numerical validity is
not an accuracy bound. Rank loss and nonfinite inputs produce `valid=False`
and NaN geometric outputs. This numerical refusal does not prove infeasibility.
Test validity before consuming a candidate. Floating-point residuals are not
exact membership certificates.

## Local nonlinear corrections

`linearized_equality_step(x, residual, jacobian)` returns the minimum-norm
correction solving `jacobian @ delta = -residual`. The Jacobian has shared
shape `(m,d)` and residual has shape `(...,m)` matching `x`'s batch shape.
Its `value` is `x + delta`; `local_half_squared_distance` is
`0.5 * ||delta||²`. For nonlinear constraints this is a local linearized
quantity. It can fail to reach a root and does not generally equal distance
to the curved solution set.

The older `omnibias.fields.locus` API computes host Gauss–Newton root landing.
Its default branch mode is now `"seed"`. Requests for nearest/all/numbered
branches raise instead of silently ignoring the request. Its implicit
derivative requires a converged regular square root and is first order only.
Underdetermined solution derivatives require a selection rule: use the affine
projection API when its assumptions apply, or explicitly differentiate the
intended iteration. The legacy JAX root landing remains eager and host-only.

See the [mathematical derivation and pilot](calculus-geometry-pilot.md) for the
calculus–geometry transfers, compatibility conditions, prior art and evidence.
