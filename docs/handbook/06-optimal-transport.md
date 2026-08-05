# Chapter 6 — Optimal transport

> Modules: `omnibias.jax.information` / `omnibias.torch.information`
> (differentiable), `omnibias.core.verified.transport` (certified intervals).

## What this chapter gives you

Divergences (Chapter 5) ask *how much* two distributions differ pointwise.
**Optimal transport** asks *how far mass must move* to turn one into the other —
a geometry-aware distance that stays meaningful even when the supports barely
overlap. The Wasserstein-`p` distance is

\[
  W_p(\mu,\nu) = \Bigl(\inf_{\pi\in\Pi(\mu,\nu)} \int \lVert x-y\rVert^p\,d\pi(x,y)\Bigr)^{1/p}.
\]

In **one dimension** it collapses to a sort: \(W_p\) is the \(L^p\) distance
between the order statistics (or quantile functions), which is exact and cheap.
omnibias gives you the 1-D forms, a Gaussian closed form, the sliced and Sinkhorn
approximations for higher dimensions, and **certified** enclosures.

| | Differentiable | Certified |
| --- | --- | --- |
| two samples, `W₁` | `wasserstein1` | `certified_wasserstein1_samples` |
| two samples, `Wₚ` | `wassersteinp` | `certified_wasserstein2_samples` (`p=2`) |
| model CDF vs data | `wasserstein1_cdf` | `certified_wasserstein1` |
| two Gaussians, `W₂` | `wasserstein2_gaussian` | `certified_wasserstein2_gaussian` |
| high-D | `sliced_wasserstein`, `sinkhorn_distance` | — |

---

## Differentiable kernels (JAX / Torch)

Backprop-able and bit-identical across backends (examples in JAX).

### `wasserstein1`

<!-- docs-test: signature -->
```python
wasserstein1(u, v) -> Array
```
1-D `W₁` between two **equal-size** empirical samples: \(\tfrac1n\sum_i|u_{(i)} -
v_{(i)}|\) from the sorted order statistics. Exact in 1-D.

```python
import jax.numpy as jnp
from omnibias.jax.information import wasserstein1
u = jnp.array([0.0, 1.0, 2.0]); v = jnp.array([0.5, 1.5, 2.5])
print(float(wasserstein1(u, v)))      # 0.5 (a uniform shift)
```

### `wassersteinp`

<!-- docs-test: signature -->
```python
wassersteinp(u, v, *, p=1.0) -> Array
```
1-D `Wₚ`: \((\text{mean}_i|u_{(i)}-v_{(i)}|^p)^{1/p}\). `p=1` reproduces
`wasserstein1`; `p=2` is the quadratic transport distance.

```python
import jax.numpy as jnp
from omnibias.jax.information import wassersteinp
u = jnp.array([0.0, 1.0, 2.0]); v = jnp.array([0.5, 1.5, 2.5])
print(float(wassersteinp(u, v, p=2.0)))   # 0.5
```

### `wasserstein1_cdf`

<!-- docs-test: signature -->
```python
wasserstein1_cdf(name, samples, *, loc=0.0, scale=1.0) -> Array
```
Model-vs-empirical `W₁` between a location-scale CDF `F` (`"sigmoid"` / `"tanh"`)
and the empirical CDF of `samples`, via the exact integral
\(W_1 = \int|F - F_n|\,dx\) through the closed-form antiderivative
\(\Phi(x) = s\,\text{softplus}((x-\text{loc})/s)\). Differentiable in `loc` /
`scale` / `samples` — fit a CDF by minimising it. (`arctan` has no finite mean →
rejected.)

```python
import jax.numpy as jnp
from omnibias.jax.information import wasserstein1_cdf
samples = jnp.array([-1.2, -0.3, 0.1, 0.8, 1.5])
print(round(float(wasserstein1_cdf("sigmoid", samples, loc=0.0, scale=1.0)), 4))
```

### `wasserstein2_gaussian`

<!-- docs-test: signature -->
```python
wasserstein2_gaussian(mu1, sigma1, mu2, sigma2) -> Array
```
The closed-form Gaussian `W₂`:
\(\sqrt{(\mu_1-\mu_2)^2 + (\sigma_1-\sigma_2)^2}\) (standard deviations).
Differentiable in all four arguments — the matched-Gaussian objective behind the
residual diagnostics.

```python
from omnibias.jax.information import wasserstein2_gaussian
print(round(float(wasserstein2_gaussian(0.0, 1.0, 1.0, 2.0)), 5))   # sqrt(1+1)=1.41421
```

### `sliced_wasserstein`

<!-- docs-test: signature -->
```python
sliced_wasserstein(X, Y, directions, *, p=2.0) -> Array
```
The sliced `Wₚ` between two equal-size point clouds `(n, d)`: average the 1-D
`Wₚ` over each unit projection in `directions` `(k, d)`. Supply the directions
(e.g. random unit vectors) so the estimate is deterministic and cross-backend
identical.

```python
import jax.numpy as jnp
from omnibias.jax.information import sliced_wasserstein
rng = __import__("numpy").random.default_rng(0)
X = jnp.asarray(rng.normal(size=(64, 3)))
Y = jnp.asarray(rng.normal(size=(64, 3)) + 1.0)
dirs = jnp.asarray(rng.normal(size=(16, 3)))
dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
print(round(float(sliced_wasserstein(X, Y, dirs)), 3))     # ~ shift magnitude
```

### `sinkhorn_distance`

<!-- docs-test: signature -->
```python
sinkhorn_distance(a, b, cost, *, epsilon=0.1, num_iters=200) -> Array
```
Entropic OT between two histograms: solves
\(\min_P\langle P, C\rangle - \epsilon H(P)\) by log-domain Sinkhorn iterations
and returns the transport cost \(\langle P^*, C\rangle\). `a` `(n,)`, `b` `(m,)`
(each summing to 1), `cost` `(n, m)`. Differentiable; as `epsilon → 0` it
approaches the exact OT cost.

```python
import jax.numpy as jnp
from omnibias.jax.information import sinkhorn_distance
a = jnp.array([0.5, 0.5]); b = jnp.array([0.5, 0.5])
cost = jnp.array([[0.0, 1.0], [1.0, 0.0]])
print(round(float(sinkhorn_distance(a, b, cost, epsilon=0.05)), 4))   # ~0 (identical marginals)
```

---

## Certified enclosures

Each returns a guaranteed `Interval` (`.lo`/`.hi`) containing the true value —
the proof-carrying twins of the differentiable kernels.

### `certified_wasserstein1_samples` and `certified_wasserstein2_samples`

<!-- docs-test: signature -->
```python
certified_wasserstein1_samples(u, v) -> Interval
certified_wasserstein2_samples(u, v) -> Interval
```
Two-sample 1-D `W₁` / `W₂` between equal-size samples, accumulated in
outward-rounded interval arithmetic (the interval bounds only the floating-point
rounding of the exact closed-form distance).

```python
from omnibias.core.verified.transport import certified_wasserstein1_samples
iv = certified_wasserstein1_samples([0.0, 1.0, 2.0], [0.5, 1.5, 2.5])
print(iv.lo, iv.hi)         # tight interval around 0.5
```

### `certified_wasserstein1`

<!-- docs-test: signature -->
```python
certified_wasserstein1(name, samples, *, loc=0.0, scale=1.0) -> Interval
```
Rigorous enclosure of the model-CDF-vs-empirical `W₁` (the certified twin of
`wasserstein1_cdf`), via the *exact antiderivative* of the logistic CDF rather
than quadrature — so the enclosure is tight. `arctan` raises (infinite `W₁`).

```python
from omnibias.core.verified.transport import certified_wasserstein1
iv = certified_wasserstein1("sigmoid", [-1.2, -0.3, 0.1, 0.8, 1.5], loc=0.0, scale=1.0)
print(iv.lo, iv.hi)
```

### `certified_wasserstein2_gaussian`

<!-- docs-test: signature -->
```python
certified_wasserstein2_gaussian(mu1, sigma1, mu2, sigma2) -> Interval
```
Rigorous enclosure of the Gaussian `W₂` closed form — every operation
outward-rounded.

```python
from omnibias.core.verified.transport import certified_wasserstein2_gaussian
iv = certified_wasserstein2_gaussian(0.0, 1.0, 1.0, 2.0)
print(iv.lo <= 2.0 ** 0.5 <= iv.hi)     # True
```

!!! tip "Differentiable ⊂ certified"
    The certified enclosure *contains* the differentiable value: use
    `wasserstein1_cdf` in a training loop, then certify the converged fit with
    `certified_wasserstein1` for an audit-grade bound.

---

## Transport as a residual diagnostic

`omnibias.symbolic.diagnostics` exposes the 1-D Wasserstein distance of a residual
sample to a mean/variance-matched Gaussian — `wasserstein_to_gaussian` (`W₁`) and
`wasserstein2_to_gaussian` (`W₂`) — scale-sensitive companions to the
information-theoretic scores of Chapter 5. They are selectable as the
`"wasserstein_gaussian"` / `"wasserstein2_gaussian"` discovery objectives.

```python
import numpy as np
from omnibias.symbolic.diagnostics import wasserstein_to_gaussian
white = np.random.default_rng(0).normal(size=4000)
print(round(wasserstein_to_gaussian(white), 4))      # ~0 for white residuals
```

---

**Next:** [Chapter 7 — Information geometry](07-information-geometry.md) turns the
metric inward — onto the space of model *parameters* — for Fisher-efficient
natural-gradient optimisation.
