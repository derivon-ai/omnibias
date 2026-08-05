# AI quickstart (vibe-coding cheat-sheet)

This page is written for an AI coding assistant (or a human in a hurry). It is
dense on purpose: an import map, a capability table, the canonical recipes, and
the gotchas that actually bite. Copy, paste, adapt.

!!! note "Conventions used everywhere in this handbook"
    - Everything is **NumPy `float64`** unless a snippet explicitly imports `jax`
      or `torch`. The discovery/geometry/exterior layer is NumPy; the
      differentiable information-theory and transport kernels live in
      `omnibias.jax` / `omnibias.torch`.
    - A **point cloud** is `X` of shape `(n, d)` — `n` sample points, `d` input
      variables.
    - A **multi-index** `α = (a_1, …, a_d)` is a plain tuple of non-negative ints;
      `|α| = sum(α)` is the derivative order and `∂^α u` is the corresponding mixed
      partial. `(0, 0)` is the value, `(1, 0)` is `∂x`, `(2, 0)` is `∂xx`, …
    - Closed form means *exact to machine precision*, not finite differences.

## Import map

| You want… | Import from | Key entry points |
| --- | --- | --- |
| Discover a 1-D ODE / activation identity | `omnibias.symbolic` | `NeuralJetDiscoverer`, `discover_activation_identity` |
| Fit a smooth 1-D field + its jet | `omnibias.symbolic` | `fit_neural_field_1d`, `extract_neural_jets` |
| Multivariate gradient / Hessian / Laplacian | `omnibias.symbolic` | `fit_neural_field_nd`, `field_gradient`, `field_laplacian` |
| Discover a PDE (heat/wave/Burgers/Laplace) | `omnibias.symbolic` | `FieldLawDiscoverer`, `discover_field_pde_law` |
| Curvature, Laplace–Beltrami, learned metric | `omnibias.symbolic` | `MetricField`, `laplace_beltrami`, `pullback_metric_field` |
| Exterior derivative / Hodge star / `d²=0` | `omnibias.symbolic` | `DifferentialForm`, `exterior_derivative`, `hodge_star` |
| Entropy / KL / JS (differentiable) | `omnibias.jax.information` or `omnibias.torch.information` | `entropy`, `kl_divergence`, `js_divergence` |
| Entropy / KL with **guaranteed bounds** | `omnibias.core.verified.information` | `entropy_enclosure`, `kl_divergence_enclosure` |
| Residual diagnostics for a fitted model | `omnibias.symbolic` | `surrogate_residual_diagnostics` |
| Wasserstein distance (differentiable) | `omnibias.jax.information` / `omnibias.torch.information` | `wasserstein1`, `sliced_wasserstein`, `sinkhorn_distance` |
| Wasserstein with **guaranteed bounds** | `omnibias.core.verified.transport` | `certified_wasserstein1_samples` |
| Fisher metric / natural gradient | `omnibias.curvature.natural_gradient` | `natural_gradient_step`, `glm_natural_gradient_step` |

Everything in `omnibias.symbolic` is re-exported from the package root, so
`from omnibias.symbolic import <name>` always works for the names in this book.

!!! important "Operators and integrals: check the operator surface first"
    omnibias exposes **six** `OperatorBlock` roles, not four:
    `identity | grad | laplacian | derivative | band | integral`. The
    `grad` / `laplacian` / `derivative` roles are closed-form `sigma^(n)`; the
    **`integral` role is a closed-form antiderivative window**
    `S(z + b_hi) - S(z + b_lo)` with `S' = sigma` (e.g. `sigmoid`'s
    antiderivative is `softplus`). "Integral" has three distinct senses --
    activation antiderivative (this), domain quadrature (`omnibias.fields`), and
    measure integral (`omnibias.measure`). The canonical capability matrix is
    [`operator-surface.md`](../operator-surface.md); ground any capability claim
    there, not in memory.

## Capability matrix

| Layer | Closed form? | Backends | Certified bounds? | Chapter |
| --- | --- | --- | --- | --- |
| 1-D neural jet & ODE discovery | ✅ exact | NumPy | — | [1](01-neural-jet-1d.md) |
| Vector calculus & PDE discovery | ✅ exact | NumPy | — | [2](02-vector-calculus-pde.md) |
| Differential geometry | ✅ field, autodiff-exact metric | NumPy | — | [3](03-differential-geometry.md) |
| Exterior calculus | ✅ exact | NumPy | — | [4](04-exterior-calculus.md) |
| Information theory | ✅ differentiable | JAX, Torch, NumPy | ✅ enclosures | [5](05-information-theory.md) |
| Optimal transport | ✅ (1-D / Gaussian exact) | JAX, Torch | ✅ enclosures | [6](06-optimal-transport.md) |
| Information geometry | ✅ exact Fisher | JAX, Torch | — | [7](07-information-geometry.md) |
| OperatorBlock roles + integrals | see [operator-surface](../operator-surface.md) | Torch, JAX, Keras | ✅ certified integral / norms | [operator-surface](../operator-surface.md) |

## The five canonical recipes

### 1. "Find the ODE behind my data"

```python
from omnibias.symbolic import discover_activation_identity

# `exp` secretly obeys y' = y; the discoverer recovers it from the closed-form jet.
result = discover_activation_identity("exp", candidate_lhs_orders=(1,))
print(result.formula())            # -> "dy = 1*y"
```

To go from *your own samples* instead of a named activation, fit a field first:
`fit_neural_field_1d(x, y)` → `extract_neural_jets(field, x)` → feed the three
`JetBundle`s to `NeuralJetDiscoverer().discover(train, val, test)`
(see [Chapter 1](01-neural-jet-1d.md)).

### 2. "Give me the exact gradient / Laplacian of a fitted field"

```python
import numpy as np
from omnibias.symbolic import (
    fit_neural_field_nd, extract_field_jet, field_gradient, field_laplacian,
)

rng = np.random.default_rng(0)
X = rng.uniform(-1.0, 1.0, size=(400, 2))
u = np.sin(X[:, 0]) * np.exp(0.3 * X[:, 1])

field = fit_neural_field_nd(X, u, hidden=512, seed=0)
jet = extract_field_jet(field, X, max_order=2)   # exact closed-form partials
g = field_gradient(jet)       # (400, 2) ∇u of the fitted field
lap = field_laplacian(jet)    # (400,)   Δu
```

### 3. "Discover which PDE my space-time data obeys"

```python
from omnibias.symbolic import make_heat_field_split, discover_field_pde_law

train, val, test, hidden = make_heat_field_split(seed=0)   # truth: u_t = 0.12*u_xx
result = discover_field_pde_law(train, val, test, lhs_index=(0, 1), time_axis=1)
print(result["equation"])                                  # -> "u_t = 0.12*u_xx"
```

### 4. "Curvature / Laplace–Beltrami on a curved space"

```python
import numpy as np
from omnibias.symbolic import warped_product_metric_field, scalar_curvature

theta = np.linspace(0.4, np.pi - 0.4, 60)
X = np.column_stack([theta, np.zeros_like(theta)])     # coordinates (theta, phi)
# warp f = sin(theta) -> the unit sphere ds^2 = dtheta^2 + sin^2(theta) dphi^2
metric = warped_product_metric_field(
    X, np.sin(theta), np.cos(theta), -np.sin(theta), var_names=("theta", "phi")
)
print(np.round(scalar_curvature(metric), 6))          # -> 2.0 everywhere (K = +1)
```

### 5. "How different are these two distributions?" (with a guarantee)

```python
from omnibias.core.verified.information import kl_divergence_enclosure

p = [0.2, 0.3, 0.5]
q = [0.1, 0.4, 0.5]
iv = kl_divergence_enclosure(p, q)   # an Interval that provably contains KL(p‖q)
print(iv.lo, iv.hi)                  # lo <= KL(p||q) <= hi, rigorously
```

## Gotchas that actually bite

!!! warning "Fit with enough features, sample where you evaluate"
    A random-feature field is only accurate inside the **support of its training
    points**. Evaluating `field_*` operators far outside that box extrapolates and
    the closed-form derivatives, while exact for the *fitted* field, will not match
    the true function. Use `n_features` ≳ a few hundred for smooth targets and keep
    evaluation points inside the training box.

- **Orders cost nothing extra per point, but you must request them.** A jet only
  carries partials up to its `max_order`/`order`. Ask for `order=2` if you need a
  Hessian, `order=3` if a curvature term needs third derivatives of a chart.
- **Multi-indices are tuples, length `d`.** `field_value(jet)` is `∂^(0,…,0)`.
  A 2-D Laplacian needs `(2,0)` and `(0,2)`; the operators do this for you.
- **`n < 0` raises `ValueError`; unimplemented orders raise `NotImplementedError`.**
  This is the derivative-tower contract — catch them, don't paper over them.
- **Probabilities must be normalised** for the differentiable info-theory kernels
  (they assume `sum(p) == 1`); the certified enclosures renormalise and widen.
- **`certified_*` returns an interval `(lo, hi)`**, never a point — that is the
  whole point. `differentiable` kernels return a scalar you can backprop through.
- **Geometry honesty note.** Field derivatives are exact closed form; *metric*
  derivatives from `analytic_metric_field` are exact **forward-mode autodiff** of
  the analytic metric. `pullback_metric_field` is fully closed form (it reads the
  chart's own jet). The docstrings label which is which.

## Determinism

Pass `random_state=<int>` to every `fit_*` / `make_*` / discoverer. Given the
same seed the fitted field, the jet, and the discovered equation are reproducible
bit-for-bit on a given platform. This is what makes the examples in this book
stable enough to assert on in the test-suite.
