# Chapter 3 — Differential geometry

> Module: `omnibias.symbolic` (source: `omnibias.symbolic.geometry_discovery`).
> [API reference](../api/symbolic.md) has the precise signatures.

## What this chapter gives you

Chapter 2's operators assumed flat Euclidean space. Give every point a **metric**
\(g_{ij}(x)\) and the same closed-form jet powers genuine Riemannian geometry:
the Laplace–Beltrami operator with its Christoffel drift, the covariant Hessian,
the Riemann / Ricci / scalar curvature tensors, and the pullback metric of a
*learned* coordinate chart.

!!! info "Honesty contract"
    **Field-function derivatives are exact closed form** (every `∂^α u` comes from
    the activation tower). **The metric and its derivatives are inputs**: for an
    analytic metric you supply exact arrays; for a learned chart `φ`,
    `pullback_metric_field` assembles the exact closed-form `g = JᵀJ` from the
    chart components' own jets. Either way the geometric operators below are exact —
    no finite differences.

```
 MetricField (g_ij, ∂g, ∂∂g) ─┬─ metric_inverse / determinant
                              ├─ christoffel_symbols ─ covariant_hessian ─ laplace_beltrami
                              ├─ riemann_tensor ─ ricci_tensor ─ scalar_curvature ─ gaussian_curvature_2d
                              └─ (from a learned chart) pullback_metric_field
```

---

## The metric

### `MetricField`

A frozen dataclass carrying, at each of `n` points:

- `X` — coordinates `(n, m)`,
- `g` — the metric `(n, m, m)` (symmetric positive-definite),
- `dg` — first derivatives `dg[:, k, i, j] = ∂_k g_ij`, shape `(n, m, m, m)`,
- `ddg` — optional second derivatives `(n, m, m, m, m)` (needed only for
  curvature),
- `var_names`.

Properties `.dim`, `.n`. Build it with the helpers below rather than by hand.

### `analytic_metric_field`

<!-- docs-test: signature -->
```python
analytic_metric_field(X, g, dg, *, ddg=None, var_names=None) -> MetricField
```
Assemble a `MetricField` from explicit exact arrays. Use when you know the metric
and its derivatives analytically.

### `flat_metric_field`

<!-- docs-test: signature -->
```python
flat_metric_field(X, *, var_names=None, with_second=True) -> MetricField
```
The Euclidean metric `g = I` (zero derivatives, zero curvature) — the sanity
baseline on which every curved operator reduces to its flat Chapter-2 twin.

```python
import numpy as np
from omnibias.symbolic import flat_metric_field, scalar_curvature
X = np.random.default_rng(0).uniform(-1, 1, size=(20, 2))
print(np.allclose(scalar_curvature(flat_metric_field(X)), 0.0))    # True
```

### `warped_product_metric_field`

<!-- docs-test: signature -->
```python
warped_product_metric_field(X, f, fp, fpp=None, *, var_names=("x", "y")) -> MetricField
```
The 2-D warped-product metric \(ds^2 = dx^2 + f(x)^2\,dy^2\) — a surface of
revolution / geodesic-polar form with Gaussian curvature \(K = -f''/f\). `f, fp,
fpp` are the warp and its derivatives *sampled at* `X[:, 0]`. Recovers the sphere
(`f = sin x`, `K = +1`), the hyperbolic plane (`f = e^x`, `K = −1`), and the flat
plane (`f = 1`).

```python
import numpy as np
from omnibias.symbolic import warped_product_metric_field, scalar_curvature

theta = np.linspace(0.4, np.pi - 0.4, 40)
X = np.column_stack([theta, np.zeros_like(theta)])
sphere = warped_product_metric_field(X, np.sin(theta), np.cos(theta), -np.sin(theta),
                                     var_names=("theta", "phi"))
print(np.round(scalar_curvature(sphere)[:3], 6))     # [2. 2. 2.]  -> R = 2 (unit sphere)
```

### `metric_inverse` and `metric_determinant`

<!-- docs-test: signature -->
```python
metric_inverse(metric) -> (n, m, m)      # g^{ij}
metric_determinant(metric) -> (n,)        # det g
```
The raised-index metric and the volume factor \(\sqrt{|g|}\) ingredient. Every
curved contraction below uses `g^{ij}`.

```python
from omnibias.symbolic import metric_inverse, metric_determinant
ginv = metric_inverse(sphere)             # (40, 2, 2)
detg = metric_determinant(sphere)         # (40,) = sin^2(theta)
```

---

## Connection and operators

### `christoffel_symbols`

<!-- docs-test: signature -->
```python
christoffel_symbols(metric) -> (n, m, m, m)   # Γ[:, k, i, j] = Γ^k_{ij}
```
The Levi-Civita connection
\(\Gamma^k_{ij} = \tfrac12 g^{kl}(\partial_i g_{lj} + \partial_j g_{li} - \partial_l g_{ij})\),
exact given the metric jet. These encode how the basis twists from point to point.

```python
from omnibias.symbolic import christoffel_symbols
G = christoffel_symbols(sphere)            # (40, 2, 2, 2)
# On the sphere, Γ^theta_{phi phi} = -sin(theta)cos(theta)
print(np.allclose(G[:, 0, 1, 1], -np.sin(theta) * np.cos(theta)))   # True
```

### `covariant_hessian`

<!-- docs-test: signature -->
```python
covariant_hessian(jet, metric, *, spatial_axes=None) -> (n, m, m)
```
The covariant Hessian \(\nabla^2_{ij}u = \partial^2_{ij}u - \Gamma^k_{ij}\partial_k u\).
The `−Γ∂u` correction is what makes it a genuine `(0,2)`-tensor on the manifold.
`spatial_axes` maps the metric's `m` coordinates onto the jet's axes (e.g. when
the jet also has a time axis).

### `laplace_beltrami`

<!-- docs-test: signature -->
```python
laplace_beltrami(jet, metric, *, spatial_axes=None) -> (n,)
```
The full position-dependent Laplacian \(\Delta_g u = g^{ij}\nabla^2_{ij}u =
\tfrac{1}{\sqrt{|g|}}\partial_i(\sqrt{|g|}\,g^{ij}\partial_j u)\). The Christoffel
drift distinguishes it from the constant-metric `field_anisotropic_laplacian` of
Chapter 2. On the flat metric it reduces exactly to `field_laplacian`.

```python
import numpy as np
from omnibias.symbolic import analytic_field_jet, laplace_beltrami

# u = cos(theta) is the l=1 zonal harmonic: Δ_g u = -2 cos(theta) on the unit sphere.
u_parts = {(0,0): np.cos(theta), (1,0): -np.sin(theta), (0,1): np.zeros_like(theta),
           (2,0): -np.cos(theta), (1,1): np.zeros_like(theta), (0,2): np.zeros_like(theta)}
ujet = analytic_field_jet(X, u_parts, order=2, var_names=("theta", "phi"))
lb = laplace_beltrami(ujet, sphere)
print(np.allclose(lb, -2.0 * np.cos(theta)))      # True  (eigenvalue -l(l+1) = -2)
```

### `metric_grad_norm_sq`

<!-- docs-test: signature -->
```python
metric_grad_norm_sq(jet, metric, *, spatial_axes=None) -> (n,)
```
The Riemannian squared gradient norm \(|\nabla u|_g^2 = g^{ij}\partial_i u\,
\partial_j u\) — the geometric eikonal term (equals 1 for a signed geodesic
distance). Reduces to `field_grad_norm_sq` when `g = I`.

---

## Curvature

### `riemann_tensor`

<!-- docs-test: signature -->
```python
riemann_tensor(metric) -> (n, m, m, m, m)   # R[:, ρ, σ, μ, ν]
```
The full Riemann curvature \(R^\rho{}_{\sigma\mu\nu}\) — the obstruction to flatness
(parallel transport around a loop). Requires `metric.ddg`.

### `ricci_tensor`

<!-- docs-test: signature -->
```python
ricci_tensor(metric) -> (n, m, m)      # R_{σν} = R^ρ_{σρν}
```
The trace of Riemann over the first and third indices — the source term of the
Einstein equations.

### `scalar_curvature`

<!-- docs-test: signature -->
```python
scalar_curvature(metric) -> (n,)       # R = g^{σν} R_{σν}
```
The fully-traced scalar curvature. For a surface `R = 2K`: `+2` on the unit
sphere, `0` on the plane, `−2` on the hyperbolic plane.

### `gaussian_curvature_2d`

<!-- docs-test: signature -->
```python
gaussian_curvature_2d(metric) -> (n,)   # K = R/2  (2-D only)
```
The Gaussian curvature of a surface. Raises if `metric.dim != 2`.

```python
import numpy as np
from omnibias.symbolic import warped_product_metric_field, scalar_curvature, gaussian_curvature_2d

x = np.linspace(-0.5, 0.5, 30)
Xh = np.column_stack([x, np.zeros_like(x)])
# warp f = e^x -> hyperbolic plane, K = -f''/f = -1
hyp = warped_product_metric_field(Xh, np.exp(x), np.exp(x), np.exp(x), var_names=("x", "y"))
print(np.round(gaussian_curvature_2d(hyp)[:3], 6))   # [-1. -1. -1.]
print(np.round(scalar_curvature(hyp)[:3], 6))        # [-2. -2. -2.]
```

---

## Learned charts and geometric discovery

### `pullback_metric_field`

<!-- docs-test: signature -->
```python
pullback_metric_field(chart_jets, *, var_names=None, with_curvature=False) -> MetricField
```

**What.** The closed-form pullback metric \(g = J^\top J\) of a *learned* chart
\(\phi: M \to \mathbb R^N\), given the chart's component jets \(\phi_a\). **When.**
You learned an embedding/immersion (each component is an omnibias field) and want
the induced metric and *every* curved operator on it — exactly. **Theory.** The
metric and its derivatives are pure products of the components' own closed-form
partials:
\[
  g_{ij} = \sum_a (\partial_i\phi_a)(\partial_j\phi_a),\quad
  \partial_k g_{ij} = \sum_a (\partial_{ki}\phi_a)(\partial_j\phi_a) + (\partial_i\phi_a)(\partial_{kj}\phi_a),
\]
with `with_curvature=True` (needs order-3 chart jets) adding `∂∂g`. This is the
symbolic-engine twin of the `omnibias.geometry` pullback.

```python
import numpy as np
from omnibias.symbolic import analytic_field_jet, pullback_metric_field, scalar_curvature

# Standard S^2 embedding φ(θ,φ) = (sinθcosφ, sinθsinφ, cosθ) -> recovers R = 2.
rng = np.random.default_rng(0)
TP = np.column_stack([rng.uniform(0.5, np.pi - 0.5, 60), rng.uniform(0, 2*np.pi, 60)])
th, ph = TP[:, 0], TP[:, 1]
def comp(val, dth, dph, dthth, dthph, dphph):
    return analytic_field_jet(TP, {(0,0): val, (1,0): dth, (0,1): dph,
                                   (2,0): dthth, (1,1): dthph, (0,2): dphph},
                              order=2, var_names=("theta", "phi"))
phi = [
    comp(np.sin(th)*np.cos(ph),  np.cos(th)*np.cos(ph), -np.sin(th)*np.sin(ph),
         -np.sin(th)*np.cos(ph), -np.cos(th)*np.sin(ph), -np.sin(th)*np.cos(ph)),
    comp(np.sin(th)*np.sin(ph),  np.cos(th)*np.sin(ph),  np.sin(th)*np.cos(ph),
         -np.sin(th)*np.sin(ph),  np.cos(th)*np.cos(ph), -np.sin(th)*np.sin(ph)),
    comp(np.cos(th),            -np.sin(th),             np.zeros_like(th),
         -np.cos(th),            np.zeros_like(th),       np.zeros_like(th)),
]
g = pullback_metric_field(phi, var_names=("theta", "phi"))
print(np.allclose(g.g[:, 0, 0], 1.0), np.allclose(g.g[:, 1, 1], np.sin(th)**2))   # True True
```

### `make_geometric_heat_split`

<!-- docs-test: signature -->
```python
make_geometric_heat_split(*, seed=0, counts=(500,320,320), degrees=(1,2), amps=(1.0,0.5))
    -> (train, val, test, (metric_tr, metric_va, metric_te), hidden_law)
```
Heat flow on the round sphere \(S^2\): exact train/val/test `FieldJet`s over
`(θ, φ, t)` plus their `S²` metrics. The solution is a sum of zonal eigenmodes
\(u = \sum_l a_l e^{-l(l+1)t}P_l(\cos\theta)\), so `u_t = Δ_g u` holds exactly.
The Laplace–Beltrami drift `cot(θ)·u_θ` is position-dependent, so this law is
**irreducible** to any constant-coefficient flat relation — the geometric atom is
genuinely necessary.

### `discover_geometric_heat_law`

<!-- docs-test: signature -->
```python
discover_geometric_heat_law(train, val, test, metrics, *, spatial_axes=(0,1),
                            time_axis=2, lhs_index=(0,0,1)) -> dict
```
Recover `u_t = Δ_g u` by injecting the exact Laplace–Beltrami column as an extra
operator atom (via `FieldLawDiscoverer(extra_columns_fn=...)`). Returns the
equation, selected terms, and validation/test RMSE.

```python
from omnibias.symbolic import make_geometric_heat_split, discover_geometric_heat_law
train, val, test, metrics, hidden = make_geometric_heat_split(seed=0)
print(hidden)                                                   # heat flow on S^2
out = discover_geometric_heat_law(train, val, test, metrics)
print(out["equation"])                                          # "u_t = 1*lap_g(u)"
```

### `evaluate_geometric_discovery`

<!-- docs-test: signature -->
```python
evaluate_geometric_discovery(*, seed=0) -> dict
```
One-call smoke run: build the spherical heat split and recover its geometric law —
the regression test for the whole geometry surface.

```python
from omnibias.symbolic import evaluate_geometric_discovery
print(evaluate_geometric_discovery(seed=0)["geometric_heat"]["equation"])
```

---

**Next:** [Chapter 4 — Exterior calculus](04-exterior-calculus.md) drops the
metric again and works with antisymmetric forms, where a single operator `d`
unifies grad, curl, and div — and `d∘d = 0` becomes machine-precision exact.
