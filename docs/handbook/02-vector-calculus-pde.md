# Chapter 2 — Vector calculus & PDE discovery

> Module: `omnibias.symbolic` (source: `omnibias.symbolic.field_discovery`).
> [API reference](../api/symbolic.md) has the precise signatures.

## What this chapter gives you

Chapter 1 lived in one input variable. Here we add as many as you like. A
`d`-input random-feature field still has an **affine** inner map, so the
multivariate Faà di Bruno chain rule collapses to a single surviving term and
*every mixed partial* is exact closed form:

\[
  \partial^\alpha u(x) = \sum_h c_h\,\sigma^{(|\alpha|)}(z_h)\prod_i (W_{hi}/s_i)^{\alpha_i},
  \qquad \alpha = (\alpha_1,\dots,\alpha_d).
\]

One activation-tower evaluation per *total* order \(k=|\alpha|\) yields **all**
order-\(k\) partials at once. On top of that single primitive sits the whole
vector-calculus surface (gradient, divergence, curl, Laplacian, Hessian, Itô
generator, anisotropic Laplacian, Wirtinger derivatives) and a multivariate
PDE-law discoverer.

```
 data ─fit─▶ NeuralFieldND ─extract─▶ FieldJet ─{grad,div,curl,lap,…}─▶ operators
                                          └────▶ FieldLawDiscoverer ─▶ PDE law
```

!!! note "Multi-indices"
    A partial is keyed by a tuple `α` of length `d`. `(0,0)` is the value, `(1,0)`
    is `∂x`, `(0,1)` is `∂t`, `(2,0)` is `∂xx`, `(1,1)` is `∂x∂t`. `field_*`
    operators take a `FieldJet` and do the bookkeeping for you.

---

## Fields and jets

### `NeuralFieldND`

The `d`-input field `u(x) = c·σ(W x̃ + β) + b` with per-coordinate
standardisation `x̃ = (x − x_mean)/x_scale`. Frozen dataclass; `.dim` is `d`.
Produced by `fit_neural_field_nd`; consumed by `extract_field_jet`.

### `fit_neural_field_nd`

<!-- docs-test: signature -->
```python
fit_neural_field_nd(X, y, *, hidden=256, ridge=1e-5, activation="tanh",
                    bandwidth=1.0, var_names=None, seed=0) -> NeuralFieldND
```

**What.** Fit a smooth scalar field over a point cloud `X` of shape `(n, d)`.
**When.** You have multivariate samples and want exact closed-form partials.
**Theory.** Same random-feature ridge readout as the 1-D case; `W` is scaled by
`bandwidth/√d` so the pre-activation has unit-ish variance.

```python
import numpy as np
from omnibias.symbolic import fit_neural_field_nd

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(400, 2))
u = np.sin(X[:, 0]) * np.exp(0.3 * X[:, 1])
field = fit_neural_field_nd(X, u, hidden=400, seed=0, var_names=("x", "y"))
print(field.dim, round(field.train_rmse, 4))     # 2  ~0.00x
```

### `extract_field_jet`

<!-- docs-test: signature -->
```python
extract_field_jet(field_nd, X, *, max_order=2) -> FieldJet
```

**What.** Compute *all* mixed partials up to `max_order` at points `X`. **When.**
Right after fitting (or building) a field, before any operator. **Theory.** One
fastpath call `σ^(k)` per total order `k`; each partial is then a cheap
contraction against the weight monomials.

```python
from omnibias.symbolic import extract_field_jet
jet = extract_field_jet(field, X, max_order=2)
print(jet.order, jet.dim, jet.n)        # 2 2 400
```

### `FieldJet`

The samples `X` plus `partials[α]` for every multi-index with `|α| ≤ order`.
Methods: `.value()` (the `(0,…,0)` partial), `.partial(α)`, and properties
`.dim`, `.n`, `.order`, `.var_names`. This is the universal currency of this
chapter and Chapters 3–4.

```python
jet.value().shape          # (400,)
jet.partial((1, 0)).shape  # (400,)  -> ∂u/∂x
```

### `analytic_field_jet`

<!-- docs-test: signature -->
```python
analytic_field_jet(X, partials, *, order, var_names=None) -> FieldJet
```

**What.** Build a `FieldJet` from explicitly supplied partial arrays. **When.**
You know a field's derivatives in closed form (exact synthetic PDE data, tests,
or a hand-derived solution) and want machine-precision operators on it.

```python
import numpy as np
from omnibias.symbolic import analytic_field_jet, field_laplacian

X = np.random.default_rng(0).uniform(-1, 1, size=(50, 2))
x, y = X[:, 0], X[:, 1]
# u = x^2 - y^2 is harmonic: Δu = 2 - 2 = 0
parts = {(0,0): x**2 - y**2, (1,0): 2*x, (0,1): -2*y,
         (2,0): np.full_like(x, 2.0), (1,1): np.zeros_like(x), (0,2): np.full_like(x, -2.0)}
jet = analytic_field_jet(X, parts, order=2, var_names=("x", "y"))
print(np.allclose(field_laplacian(jet), 0.0))     # True
```

### `field_derivative_jet`

<!-- docs-test: signature -->
```python
field_derivative_jet(jet, axis) -> FieldJet
```

**What.** The `FieldJet` of `∂u/∂x_axis`, one order lower. **When.** Building a
vector field from a potential (`[field_derivative_jet(phi, i) for i in range(d)]`)
or composing div/curl of derived fields. **Theory.** Shifts every stored partial
up by one along `axis` (`∂^α(∂_a u) = ∂^{α+e_a} u`).

```python
from omnibias.symbolic import field_derivative_jet
ux_jet = field_derivative_jet(jet, 0)    # field whose value is ∂u/∂x, order 1
print(ux_jet.order)                       # jet.order - 1
```

---

## The vector-calculus operators

All take a `FieldJet` (or a list of them) and return exact NumPy arrays. `axes`
selects which coordinates participate (default: all).

### `field_value`

<!-- docs-test: signature -->
```python
field_value(jet) -> (n,)
```
The field value `u` itself — the zeroth partial. Handy when you only need the
fitted surface.

### `field_gradient`

<!-- docs-test: signature -->
```python
field_gradient(jet, *, axes=None) -> (n, len(axes))
```
The gradient \(\nabla u\). Needs `order ≥ 1`.

```python
from omnibias.symbolic import field_gradient
g = field_gradient(jet)            # (400, 2)
```

### `field_hessian`

<!-- docs-test: signature -->
```python
field_hessian(jet, *, axes=None) -> (n, m, m)
```
The Hessian \(\partial^2_{ij}u\); symmetric by construction (exact mixed-partial
symmetry). Needs `order ≥ 2`.

```python
from omnibias.symbolic import field_hessian
H = field_hessian(jet)             # (400, 2, 2); H[:,0,1] == H[:,1,0]
```

### `field_laplacian`

<!-- docs-test: signature -->
```python
field_laplacian(jet, *, axes=None) -> (n,)
```
\(\Delta u = \sum_i \partial^2_{ii}u\) over `axes`. The closed-form Laplacian that
makes PINNs and heat/wave laws cheap.

### `field_grad_norm_sq`

<!-- docs-test: signature -->
```python
field_grad_norm_sq(jet, *, axes=None) -> (n,)
```
The eikonal term \(|\nabla u|^2 = \sum_i(\partial_i u)^2\); equals 1 for a signed
distance function.

### `field_divergence`

<!-- docs-test: signature -->
```python
field_divergence(components, *, axes=None) -> (n,)
```
\(\nabla\cdot F = \sum_i \partial_i F_i\) of a vector field given as a list of
component `FieldJet`s. `axes[i]` picks the differentiation axis of component `i`.

### `field_curl`

<!-- docs-test: signature -->
```python
field_curl(components) -> (n,) in 2-D  |  (n, 3) in 3-D
```
Scalar vorticity in 2-D (`∂₀F₁ − ∂₁F₀`) or the vector curl in 3-D.

```python
import numpy as np
from omnibias.symbolic import fit_neural_field_nd, extract_field_jet, field_divergence, field_curl

rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, size=(200, 2))
fx = fit_neural_field_nd(X, rng.normal(size=200), hidden=64, seed=1, var_names=("x", "y"))
fy = fit_neural_field_nd(X, rng.normal(size=200), hidden=64, seed=2, var_names=("x", "y"))
comps = [extract_field_jet(fx, X, max_order=1), extract_field_jet(fy, X, max_order=1)]
div = field_divergence(comps)      # (200,)
vort = field_curl(comps)           # (200,) scalar vorticity
```

### `field_ito_generator`

<!-- docs-test: signature -->
```python
field_ito_generator(jet, drift, diffusion, *, axes=None) -> (n,)
```
The Itô / backward-Kolmogorov generator
\(\mathcal L u = \sum_i b_i\partial_i u + \tfrac12\sum_{ij}(\sigma\sigma^\top)_{ij}\partial^2_{ij}u\)
of an SDE `dX = b dt + σ dW` (and the Fokker–Planck adjoint's principal part).
`drift` is `(m,)`, `diffusion` the `(m,m)` matrix `σσᵀ`. Needs `order ≥ 2`.

```python
import numpy as np
from omnibias.symbolic import field_ito_generator
b = np.array([0.1, -0.2]); D = np.array([[0.5, 0.0], [0.0, 0.3]])
Lu = field_ito_generator(jet, b, D)     # (400,)
```

### `field_anisotropic_laplacian`

<!-- docs-test: signature -->
```python
field_anisotropic_laplacian(jet, metric_inv, *, axes=None) -> (n,)
```
The **constant-metric** Laplace–Beltrami principal part
\(\sum_{ij} g^{ij}\partial^2_{ij}u\). For a *position-dependent* metric (with the
Christoffel drift) use `laplace_beltrami` in [Chapter 3](03-differential-geometry.md);
this is the honest constant-metric special case.

```python
import numpy as np
from omnibias.symbolic import field_anisotropic_laplacian
ginv = np.array([[2.0, 0.3], [0.3, 1.0]])
print(field_anisotropic_laplacian(jet, ginv).shape)    # (400,)
```

### `field_wirtinger`

<!-- docs-test: signature -->
```python
field_wirtinger(u_jet, v_jet=None, *, axes=(0, 1)) -> (d_z, d_zbar)
```
The Wirtinger derivatives \((\partial_z f,\ \partial_{\bar z}f)\) of a complex
field \(f = u + iv\) over the two `axes` `(x, y)`. The Cauchy–Riemann /
holomorphy test is `∂_{z̄} f = 0`; then `∂_z f = f'(z)`.

```python
import numpy as np
from omnibias.symbolic import analytic_field_jet, field_wirtinger
# f(z) = z^2 = (x^2 - y^2) + i(2xy) is holomorphic -> ∂_zbar f = 0
X = np.random.default_rng(0).uniform(-1, 1, size=(40, 2)); x, y = X[:, 0], X[:, 1]
u = analytic_field_jet(X, {(0,0): x**2 - y**2, (1,0): 2*x, (0,1): -2*y}, order=1)
v = analytic_field_jet(X, {(0,0): 2*x*y, (1,0): 2*y, (0,1): 2*x}, order=1)
dz, dzbar = field_wirtinger(u, v)
print(np.allclose(dzbar, 0.0))      # True (holomorphic)
```

---

## Discovering a PDE law

### `field_partial_name`

<!-- docs-test: signature -->
```python
field_partial_name(alpha, var_names, *, lhs="u") -> str
```
The readable operator name for a partial: `(0,0)→"u"`, `(1,0)→"u_x"`,
`(1,1)→"u_xt"`, `(0,2)→"u_tt"`. Used for every column label.

### `field_operator_columns`

<!-- docs-test: signature -->
```python
field_operator_columns(jet, *, lhs="u", max_partial_order=None, spatial_axes=None,
                       include_laplacian=True, include_grad_norm_sq=False) -> dict[str, (n,)]
```
The named base operator atoms of a scalar jet (the value, every partial, and
optionally `lap(u)` / `|grad u|²`) — the building blocks multiplied together by
the library builder.

```python
from omnibias.symbolic import field_operator_columns
cols = field_operator_columns(jet, include_laplacian=True)
print(sorted(cols))     # ['lap(u)', 'u', 'u_x', 'u_xx', 'u_xy', 'u_y', 'u_yy']
```

### `build_field_relation_library`

<!-- docs-test: signature -->
```python
build_field_relation_library(jet, *, lhs_index, max_degree=1, time_axis=None,
                             rhs_orders=None, spatial_axes=None, include_laplacian=False,
                             extra_columns=None, exclude=()) -> (design, term_names)
```

**What.** Polynomial design over operator columns, excluding the LHS partial.
**When.** Custom PDE search or inspecting candidate terms. **Theory.** Two
physically-motivated restrictions tame the notorious PDE-library degeneracy:

- `time_axis` — the *method-of-lines* restriction: drop any partial whose
  time-derivative order is ≥ the LHS's, so `u_t = F` has `F` free of `u_t, u_tt,
  u_xt`.
- `rhs_orders` — keep only partials whose *total* order lies in this set (e.g.
  `(2,)` recovers the elliptic principal part `u_xx = −u_yy`).

`extra_columns` injects custom exact atoms (this is how Chapter 3 adds the
Laplace–Beltrami column).

### `FieldLawDiscoverer`

<!-- docs-test: signature -->
```python
FieldLawDiscoverer(max_degree=1, time_axis=None, rhs_orders=None, spatial_axes=None,
                   include_laplacian=False, divergence_objective=None, ...)
.discover(train, val, test, *, lhs_index, lhs="u", exclude=(), extra_columns_fn=None) -> FieldLawResult
```

**What.** The multivariate twin of `NeuralJetDiscoverer`: search a sparse relation
for the `lhs_index` partial over the operator monomials, selecting on validation
RMSE + complexity. **When.** You have train/val/test `FieldJet`s of a space-time
field and want its governing PDE. The result's `.formula()` / `.active_terms()` /
`.validation_rmse` / `.test_rmse` report the law. `extra_columns_fn` injects a
per-jet exact column (e.g. a geometric operator).

```python
from omnibias.symbolic import make_heat_field_split, FieldLawDiscoverer
train, val, test, hidden = make_heat_field_split(seed=0)
disc = FieldLawDiscoverer(max_degree=1, time_axis=1)
res = disc.discover(train, val, test, lhs_index=(0, 1))   # u_t on the LHS
print(res.formula())          # "u_t = 0.12*u_xx"
print(res.test_rmse)          # ~1e-13 (exact analytic jets)
```

### `discover_field_pde_law`

<!-- docs-test: signature -->
```python
discover_field_pde_law(train, val, test, *, lhs_index, max_degree=1, time_axis=None,
                       rhs_orders=None, spatial_axes=None, include_laplacian=False, exclude=()) -> dict
```
The convenience wrapper: runs a `FieldLawDiscoverer` and returns a dict with
`"equation"`, `"selected_terms"`, `"validation_rmse"`, `"test_rmse"`,
`"target_scale"`.

```python
from omnibias.symbolic import make_wave_field_split, discover_field_pde_law
train, val, test, hidden = make_wave_field_split(seed=0)
print(discover_field_pde_law(train, val, test, lhs_index=(0, 2), time_axis=1)["equation"])
# "u_tt = 1.69*u_xx"   (wave speed c=1.3 -> c^2=1.69)
```

---

## Canonical PDE datasets

Each builder returns `(train, val, test, hidden_law_str)` of **exact analytic**
`FieldJet`s, so recovery is machine-precision. They deliberately stack
*multiple modes/shocks* so the only relation holding across the whole dataset is
the genuine PDE (not a single-mode shortcut).

| Builder | Coordinates | Hidden law | Discover with |
| --- | --- | --- | --- |
| `make_laplace_field_split` | `(x, y)` | `u_xx = −u_yy` (Δu=0) | `lhs_index=(2,0), rhs_orders=(2,)` |
| `make_heat_field_split` | `(x, t)` | `u_t = k·u_xx` | `lhs_index=(0,1), time_axis=1` |
| `make_wave_field_split` | `(x, t)` | `u_tt = c²·u_xx` | `lhs_index=(0,2), time_axis=1` |
| `make_burgers_field_split` | `(x, t)` | `u_t = −u·u_x + ν·u_xx` | `lhs_index=(0,1), max_degree=2, time_axis=1` |
| `make_heat2d_field_split` | `(x, y, t)` | `u_t = k(u_xx+u_yy)` | `lhs_index=(0,0,1), time_axis=2` |

```python
from omnibias.symbolic import make_burgers_field_split, discover_field_pde_law
train, val, test, hidden = make_burgers_field_split(seed=0)
print(hidden)   # "u_t = -u*u_x + 0.1*u_xx  (viscous Burgers)"
out = discover_field_pde_law(train, val, test, lhs_index=(0, 1), max_degree=2, time_axis=1)
print(out["equation"])     # nonlinear law: -1*u*u_x + 0.1*u_xx
```

### `evaluate_field_pde_discovery`

<!-- docs-test: signature -->
```python
evaluate_field_pde_discovery(*, seed=0) -> dict
```
Runs all five canonical recoveries with the right restrictions and returns the
discovered equation + metrics for each — a one-call regression/smoke test for the
whole multivariate surface.

```python
from omnibias.symbolic import evaluate_field_pde_discovery
report = evaluate_field_pde_discovery(seed=0)
print(report["heat"]["equation"], "|", report["wave"]["equation"])
```

---

**Next:** [Chapter 3 — Differential geometry](03-differential-geometry.md) gives
each point a metric, turning these flat operators into their curved-manifold
counterparts.
