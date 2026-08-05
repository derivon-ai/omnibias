# Chapter 5 — Information theory & divergences

> Modules: `omnibias.jax.information` / `omnibias.torch.information`
> (differentiable), `omnibias.symbolic.diagnostics` (NumPy + residual reports),
> `omnibias.core.verified.information` (certified intervals).

## What this chapter gives you

Three **flavours** of the same functionals, so you can pick the right tool:

| Flavour | Import from | Returns | Use when |
| --- | --- | --- | --- |
| **Differentiable** | `omnibias.jax.information` / `omnibias.torch.information` | a backend scalar (backprop-able) | training losses, model selection |
| **NumPy diagnostics** | `omnibias.symbolic.diagnostics` | a Python/NumPy float | inspecting residuals, reports |
| **Certified** | `omnibias.core.verified.information` | an `Interval` `(lo, hi)` | guarantees, proofs, audits |

All three share the `xlogy` convention \(0\ln 0 := 0\) bit-for-bit (parity
tested), so nothing is forked — only the glue differs. Everything is in **nats**.

!!! note "Inputs are probability vectors"
    `p`, `q` are discrete distributions (`sum == 1`). The differentiable and NumPy
    twins assume you pass normalised vectors; the certified enclosures validate
    `0 ≤ p ≤ 1` and clamp results to their proven sign (`H ≥ 0`, `D ≥ 0`).

---

## Differentiable kernels (JAX / Torch)

These are the bit-identical twins on the two backends; swap
`omnibias.jax.information` for `omnibias.torch.information` and the math is
identical. Below uses JAX.

### `entropy`

<!-- docs-test: signature -->
```python
entropy(p, *, dim=-1) -> Array
```
Shannon entropy \(H(p) = -\sum_i p_i\ln p_i\). Maximised by the uniform
distribution (`ln k`), zero for a point mass.

```python
import jax.numpy as jnp
from omnibias.jax.information import entropy
print(float(entropy(jnp.array([0.5, 0.5]))))      # 0.6931 = ln 2
print(float(entropy(jnp.array([1.0, 0.0]))))      # 0.0
```

### `cross_entropy`

<!-- docs-test: signature -->
```python
cross_entropy(p, q, *, dim=-1) -> Array
```
\(H(p,q) = -\sum_i p_i\ln q_i\) — the average code length using `q` for data from
`p`. The basis of the cross-entropy loss; `cross_entropy(p, p) = entropy(p)`.

### `kl_divergence`

<!-- docs-test: signature -->
```python
kl_divergence(p, q, *, dim=-1) -> Array
```
\(D(p\|q) = \sum_i p_i\ln(p_i/q_i) = H(p,q) - H(p) \ge 0\). The fundamental
"extra nats" of using `q` instead of the truth `p`. Asymmetric.

```python
import jax.numpy as jnp
from omnibias.jax.information import kl_divergence
p = jnp.array([0.2, 0.3, 0.5]); q = jnp.array([0.1, 0.4, 0.5])
print(round(float(kl_divergence(p, q)), 5))       # 0.05232
```

### `js_divergence`

<!-- docs-test: signature -->
```python
js_divergence(p, q, *, dim=-1) -> Array
```
The symmetric Jensen–Shannon divergence \(\tfrac12 D(p\|m) + \tfrac12 D(q\|m)\)
with \(m=(p+q)/2\). Bounded in `[0, ln 2]`; its square root is a metric.

### `mutual_information`

<!-- docs-test: signature -->
```python
mutual_information(p_joint) -> Array
```
\(I(X;Y) = D(P_{XY}\,\|\,P_X\otimes P_Y)\) from a joint table (last two axes) —
how many nats `X` and `Y` share. Zero iff independent.

```python
import jax.numpy as jnp
from omnibias.jax.information import mutual_information
joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])       # perfectly correlated bits
print(round(float(mutual_information(joint)), 5))  # 0.69315 = ln 2
```

### Generalised divergences & entropies

| Function | Formula | Notes |
| --- | --- | --- |
| `total_variation_distance(p, q)` | \(\tfrac12\sum_i\lvert p_i-q_i\rvert\) | metric, in `[0,1]` |
| `hellinger_distance(p, q)` | \(\sqrt{\tfrac12\sum_i(\sqrt{p_i}-\sqrt{q_i})^2}\) | metric, finite with zeros |
| `chi_squared_divergence(p, q)` | \(\sum_i (p_i-q_i)^2/q_i\) | local curvature of KL |
| `renyi_divergence(p, q, alpha)` | \(\tfrac1{\alpha-1}\ln\sum_i p_i^\alpha q_i^{1-\alpha}\) | `α>0, α≠1`; `α→1` is KL |
| `renyi_entropy(p, alpha)` | \(\tfrac1{1-\alpha}\ln\sum_i p_i^\alpha\) | collision entropy at `α=2` |
| `tsallis_entropy(p, order)` | \((1-\sum_i p_i^q)/(q-1)\) | non-extensive, `q→1` is Shannon |
| `f_divergence(p, q, f)` | \(\sum_i q_i\,f(p_i/q_i)\) | generic Csiszár; `f` convex, `f(1)=0` |

```python
import jax.numpy as jnp
from omnibias.jax.information import (
    total_variation_distance, hellinger_distance, renyi_divergence, f_divergence,
)
p = jnp.array([0.2, 0.3, 0.5]); q = jnp.array([0.1, 0.4, 0.5])
print(round(float(total_variation_distance(p, q)), 4))          # 0.1
print(round(float(renyi_divergence(p, q, 0.5)), 5))             # Rényi-1/2
# f(t) = t ln t recovers the KL divergence:
print(round(float(f_divergence(p, q, lambda t: t * jnp.log(t))), 5))   # 0.05232 (= KL)
```

---

## NumPy diagnostics & residual reports

The point-estimate twins (same `xlogy` convention) plus the glue that turns a
*sample* into a distribution and scores how "white and Gaussian" a model's
residuals are — the information-theoretic side of model selection (a correct law
leaves structureless residuals).

### Discrete twins

`entropy`, `kl_divergence`, `js_divergence`, `mutual_information`,
`total_variation_distance`, `hellinger_distance`, `chi_squared_divergence`,
`renyi_divergence` — identical formulas to the backends, on NumPy arrays.

```python
import numpy as np
from omnibias.symbolic.diagnostics import entropy, kl_divergence
print(round(float(entropy(np.array([0.5, 0.5]))), 5))             # 0.69315
print(round(float(kl_divergence(np.array([0.2, 0.8]), np.array([0.5, 0.5]))), 5))
```

### `histogram_pmf`, `gaussian_entropy`, `differential_entropy`

<!-- docs-test: signature -->
```python
histogram_pmf(samples, *, bins=32, value_range=None) -> (pmf, edges)
gaussian_entropy(std) -> float                 # 0.5 ln(2πe σ²)
differential_entropy(samples, *, bins=32) -> float
```
Turn a continuous sample into a binned pmf, then estimate its differential
entropy; `gaussian_entropy` is the maximum-entropy reference for a given variance.

```python
import numpy as np
from omnibias.symbolic.diagnostics import differential_entropy, gaussian_entropy
s = np.random.default_rng(0).normal(0, 2.0, size=20000)
print(round(differential_entropy(s, bins=64), 3), round(gaussian_entropy(2.0), 3))  # ~equal
```

### Divergence-to-Gaussian scores

For a residual sample, the distance to a mean/variance-matched Gaussian is a
non-Gaussianity / leftover-structure score (zero only for white residuals):
`kl_to_gaussian`, `js_to_gaussian`, `total_variation_to_gaussian`,
`hellinger_to_gaussian`, `chi_squared_to_gaussian`, `renyi_to_gaussian`,
`wasserstein_to_gaussian`, `wasserstein2_to_gaussian` (the last two are transport,
[Chapter 6](06-optimal-transport.md)).

```python
import numpy as np
from omnibias.symbolic.diagnostics import kl_to_gaussian
white = np.random.default_rng(0).normal(size=5000)
skewed = np.random.default_rng(0).exponential(size=5000)
print(round(kl_to_gaussian(white), 4), "<", round(kl_to_gaussian(skewed), 4))   # ~0 < bigger
```

### `feature_residual_mutual_information`

<!-- docs-test: signature -->
```python
feature_residual_mutual_information(feature, residuals, *, bins=16, bias_correction=True) -> float
```
Leftover dependence \(I(\text{feature};\text{residual})\) via a 2-D histogram with
the Miller–Madow bias correction — `~0` for a correct model, positive when
structure remains. (Scale-invariant: read alongside the RMSE.)

```python
import numpy as np
from omnibias.symbolic.diagnostics import feature_residual_mutual_information
x = np.random.default_rng(0).uniform(-2, 2, size=4000)
print(round(feature_residual_mutual_information(x, np.random.default_rng(1).normal(size=4000)), 4))  # ~0
print(round(feature_residual_mutual_information(x, np.sin(3 * x)), 4))                               # > 0
```

### Residual reports

<!-- docs-test: signature -->
```python
surrogate_residual_diagnostics(x, y_true, y_pred, *, bins=32, dependence_bins=16) -> dict
```
The full residual report for a discovered model: RMSE/std, differential entropy
vs the Gaussian reference, KL/`W₁`-to-Gaussian, and the max input–residual MI.
`residual_distribution_report` and `residual_dependence_report` are its halves.
These are what `discover_interpretable_surrogate` attaches automatically.

```python
import numpy as np
from omnibias.symbolic.diagnostics import surrogate_residual_diagnostics
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, size=(500, 2)); y = x[:, 0] ** 2
rep = surrogate_residual_diagnostics(x, y, y + rng.normal(0, 1e-3, size=500))
print(round(rep["rmse"], 4), round(rep["max_feature_residual_mi"], 4))
```

### `divergence_objective_term` and `DIVERGENCE_OBJECTIVES`

<!-- docs-test: signature -->
```python
divergence_objective_term(name, x, residuals, *, bins=32, dependence_bins=16) -> float
```
One non-negative penalty for a divergence-aware selection objective — pass the
`name` (one of `DIVERGENCE_OBJECTIVES`: `"kl_gaussian"`, `"js_gaussian"`,
`"tv_gaussian"`, `"hellinger_gaussian"`, `"chi2_gaussian"`,
`"wasserstein_gaussian"`, `"wasserstein2_gaussian"`, `"residual_mi"`) as the
`divergence_objective` of a discoverer to make information theory / optimal
transport co-drive model selection.

```python
from omnibias.symbolic import make_symbolic_regression_dataset, discover_interpretable_surrogate
data = make_symbolic_regression_dataset(seed=0)
out = discover_interpretable_surrogate(data, divergence_objective="kl_gaussian", divergence_weight=0.5)
print(out["family"], "->", out["equation"])
```

---

## Certified enclosures

Every functional, returned as a guaranteed `Interval` (with `.lo`/`.hi`) built
from a monotone `ln` enclosure and outward-rounded interval algebra. The true
value provably lies in `[lo, hi]`.

### The family

<!-- docs-test: signature -->
```python
entropy_enclosure(probs) -> Interval
cross_entropy_enclosure(p, q) -> Interval
kl_divergence_enclosure(p, q) -> Interval
js_divergence_enclosure(p, q) -> Interval
mutual_information_enclosure(joint) -> Interval
total_variation_enclosure(p, q) -> Interval
hellinger_enclosure(p, q) -> Interval
chi_squared_enclosure(p, q) -> Interval
```

Inputs are sequences of `IntervalLike` (plain floats, or `Interval`s — e.g.
certified band masses). KL/JS/χ²/cross-entropy need `q_i > 0` where `p_i > 0`;
each result is clamped to its proven range.

```python
from omnibias.core.verified.information import entropy_enclosure, kl_divergence_enclosure
he = entropy_enclosure([0.5, 0.5])
print(he.lo, he.hi)                       # tight interval around ln 2 = 0.6931...
kl = kl_divergence_enclosure([0.2, 0.3, 0.5], [0.1, 0.4, 0.5])
print(kl.lo, kl.hi)                       # proven bracket around 0.0523248 (width ~1e-15)
```

### `binned_distribution_enclosure`

<!-- docs-test: signature -->
```python
binned_distribution_enclosure(name, edges, *, loc=0.0, scale=1.0) -> list[Interval]
```
Certified per-bin masses of a location-scale model CDF (`"sigmoid"` / `"tanh"`)
over `edges`. Feed the result straight into `entropy_enclosure` /
`kl_divergence_enclosure` for a **proof-carrying** entropy/divergence of a binned
model distribution.

```python
from omnibias.core.verified.information import binned_distribution_enclosure, entropy_enclosure
masses = binned_distribution_enclosure("sigmoid", [-3.0, -1.0, 0.0, 1.0, 3.0], loc=0.0, scale=1.0)
H = entropy_enclosure(masses)
print(H.lo, H.hi)                         # rigorous enclosure of the binned model entropy
```

---

**Next:** [Chapter 6 — Optimal transport](06-optimal-transport.md) measures the
*geometry* between distributions — how far mass must move, not just how much
differs.
