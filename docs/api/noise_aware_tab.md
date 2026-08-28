# Noise-aware tabular training (05-03)

A frozen, independent estimate of per-row aleatoric (data) noise, turned into
an **exact curvature term** for `omnibias-tab`'s exact-Newton trainer and into
a **closed-form soft-binning feature embedding** -- aimed at the regime
Kartashev, Rubachev and Babenko (arXiv:2509.04430) show the GBDT-vs-tabular-DL
gap actually concentrates in: high-label-noise rows.

Status is **gated**. `benchmarks/tabular_uncertainty.py` has run `--full`
end to end, G0-G5: **G0/G1b passed** (every mechanism's plumbing and Hessian
identity are correct), but **all three falsifiable hypothesis gates
failed** -- G1 (noise-damped exact Newton, `fit_noise_aware`), G2 (the
band-embedding pullback-metric loss), and G3 (GLS-reweighted boosting) each
miss their predeclared bar. **G4 passed** (the public-suite win/loss report is
complete on all 9 sets -- a reporting-completeness gate, not a win claim) and
**G5 failed** (3 of 9 public sets regress beyond their own across-seed
noise). Per the spec's predeclared rule, every result is recorded plainly
rather than tuned away; see
[theory 05-03](https://github.com/derivon-ai/omnibias/blob/main/theory/05-applications/03-data-uncertainty-training-signal.md)
section 8 for every measured number. Every mechanism below is retained in the
codebase as a correctly-implemented, tested primitive; none is claimed as a
benchmarked win.

**Two mechanisms, not one -- do not conflate them.** `fit_noise_aware` adds a
*zero-gradient* curvature penalty: it changes nothing about the task loss's
gradient or minimizer, only the trust-region geometry, and reproduces
`fit_second_order` bit-identically at `lam=0` (gate G0). `fit_boosted_heteroscedastic`
is the opposite, older idea -- reweighting the boosting target by `1/s_hat**2`
(GLS), which *does* change the fitted function.

**`s_hat` is `GuaranteeKind.MODEL_BASED`**
([`omnibias.core.uncertainty`](conformal_slabs.md), theory 04-02), never a
sound enclosure. It must come from a frozen, train-only-fit, architecturally
distinct estimator (CatBoost's `RMSEWithUncertainty`) and must never reach
`certify_tab` as if it were an interval. `BandFeatureEmbedder`'s `beta -> inf`
is **temperature collapse** (feasibility sense) -- the same sense as
`SoftTreeConfig`'s existing gate anneal -- never the founding `delta -> 0`
bias collapse. Regression only, matching the source paper's own stated scope.

## The closed-form heteroscedastic Gaussian

`y | x ~ N(u(x), s(x)**2)`, `s = e^v`. `gaussian_nll_grad_hess` returns the two
exact gradients plus the **exact Fisher block** `diag(1/s**2, 2)` -- positive
definite by inspection, unlike the observed per-sample Hessian, whose cross
term changes sign with the residual:

```python
import numpy as np
from omnibias.tab._core.heteroscedastic import gaussian_nll_grad_hess

u = np.array([0.0, 1.0])
v = np.array([0.0, -0.5])  # s = exp(v)
y = np.array([0.3, 1.2])
grad_u, grad_v, fisher = gaussian_nll_grad_hess(u, v, y)
assert fisher.shape == (2, 2, 2)
assert np.allclose(fisher[..., 0, 0], np.exp(-2.0 * v))  # 1/s**2
assert np.allclose(fisher[..., 1, 1], 2.0)  # exact, model-independent
```

::: omnibias.tab._core.heteroscedastic
    options:
      show_root_heading: false
      heading_level: 3

## The zero-gradient Gauss-Newton penalty (falsified as a robust win -- G1)

`fit_noise_aware` adds `(lam/n) * sum_i s_i**2 * (phi_i - stopgrad(phi_i))**2`
inside `fit_second_order`'s closure. At the point it is added this term is
exactly zero with an exactly zero gradient, and contributes exactly
`lam * Sigma_noise` to the Hessian the trust-region optimizer sees -- so
`lam=0` reproduces `fit_second_order` bit-for-bit (G0), and `lam > 0` damps
the Newton step specifically along high-noise-row directions. `HeteroscedasticHead`
/ `fit_heteroscedastic` jointly fit `(f_hat, log_scale)` by the NLL above, for
when no independent estimator is available. **Retained in the codebase as a
mathematically-correct, tested mechanism (G0, G1b); not claimed as a
benchmarked win** -- see the module docstring and theory 05-03 section 8/11
for the two distinct failure shapes G1 measured.

::: omnibias.tab.torch.heteroscedastic
    options:
      show_root_heading: false
      heading_level: 3

## Soft-binning band / integral embedding (numpy core)

`band_embed` / `integral_embed` are a batched **lift** of
[`omnibias.core.ftc.ftc_block`](core.md)'s window formula (also
`OperatorBlock(op="band"|"integral")`) over a grid of thresholds, giving one
numeric feature `J + 1` soft histogram bins. Two identities hold **exactly**
for any finite `beta > 0`, not just as `beta -> inf`:

```python
import numpy as np
from omnibias.core.band_embed import band_embed, integral_embed

t = np.array([-1.0, 0.0, 1.0])
x = np.array([0.3])
bins = band_embed(x, t, beta=4.0)
assert bins.shape == (1, 4)
assert np.allclose(bins.sum(axis=-1), 1.0)  # telescoping: exact partition of unity

# FTC: d/dx integral_embed == beta * band_embed (checked here by finite difference)
eps = 1e-6
d_num = (integral_embed(x + eps, t, beta=4.0) - integral_embed(x - eps, t, beta=4.0)) / (2 * eps)
assert np.allclose(d_num, 4.0 * bins, atol=1e-4)
```

::: omnibias.core.band_embed
    options:
      show_root_heading: false
      heading_level: 3

## `BandFeatureEmbedder` and local target consistency (torch)

`BandFeatureEmbedder` reimplements the same closed form natively in torch so
learnable per-feature thresholds and a learnable `beta` differentiate through
it (bit-identical forward parity with the numpy reference and the JAX twin is
a test, not shared code). `local_target_consistency_loss` is the closed-form
angle on the source paper's "neighbors in embedding space should have similar
targets" (their section 5.1, measured there with a sampled triplet loss):
given a usable `df/dx`, it maximizes a stable surrogate of the Rayleigh
quotient `R = ||J df/dx||^2 / ||df/dx||^2` (`J` the embedder's Jacobian) via a
single `torch.autograd.functional.jvp` call, with no sampled triplets. This
needs `df/dx` to be usable, which is **not** available for arbitrary tabular
data -- gate G2 runs it only where the gradient is exact (a synthetic
generator) or a frozen surrogate estimate (public data, error reported, not
hidden). **`--full` result: G2 failed on both cases** -- a near-wash on the
exact-gradient synthetic set (worst-seed ratio `1.05`, need `<= 0.95`) and a
decisive loss on the surrogate-gradient public set (`kin8nm`, worst-seed
ratio `1.87`), where the surrogate gradient's own error appears to actively
mislead the objective. See theory 05-03 section 8.

```python
import numpy as np
import torch
from omnibias.tab.torch.embed import BandFeatureEmbedder, local_target_consistency_loss

rng = np.random.default_rng(0)
X_ref = rng.normal(size=(200, 3))
embedder = BandFeatureEmbedder(n_features=3, n_bins=4, role="band", init="quantile", X_ref=X_ref)

X = torch.tensor(X_ref[:32], dtype=torch.float64)
E = embedder(X)
assert E.shape == (32, 3 * (4 + 1))
assert torch.allclose(E.reshape(32, 3, 5).sum(-1), torch.ones(32, 3, dtype=torch.float64))

df_dx = X.clone()  # a known target's gradient, e.g. f(x) = 0.5 * ||x||^2
loss = local_target_consistency_loss(embedder, X, df_dx=df_dx)
assert torch.isfinite(loss)
```

::: omnibias.tab.torch.embed
    options:
      show_root_heading: false
      heading_level: 3

### JAX twin (forward parity only)

`omnibias.tab.jax.embed` matches `BandFeatureEmbedder`'s forward bit-for-bit
(parity test, not shared code). There is no JAX trainer here because no JAX
trainer exists anywhere in `omnibias.tab` today -- matching the package's
existing convention, not a new gap.

::: omnibias.tab.jax.embed
    options:
      show_root_heading: false
      heading_level: 3

## Newton-boosted GLS reweighting

`fit_boosted_heteroscedastic` reuses
[`fit_boosted`](tab.md#newton-boosting-driver-torch)'s existing per-sample
Hessian-weighted weak-learner fit, replacing the weight `h_i` with
`h_i / s_hat_i**2` when `weighting="gls"` -- a standard
generalized-least-squares reweighting (in the lineage of NGBoost) that gives
low-noise rows more say in each weak learner. `weighting="shrinkage"` leaves
`h_i` unchanged and reproduces `fit_boosted` bit-identically (a second
G0-style check, gate G3's baseline arm).

**`--full` results: G3 failed on both synthetic datasets** (worst-seed
`gls`/`shrinkage` top-decile-RMSE ratio `1.01` on each, need `<= 0.95`) --
GLS reweighting is statistically indistinguishable from plain shrinkage
there, a **wash, not a harm**. **G5 (public suite) failed on 2 of 9 sets**
specifically against plain `fit_boosted` (`energy_efficiency`, `auto_mpg`):
on those two, GLS reweighting *does* measurably regress overall RMSE beyond
seed noise, so "no measured win" (G3) and "no measured harm" (G5) are not the
same claim everywhere. See theory 05-03 section 8 for the full tables.

```python
import numpy as np
from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab.torch.boosting import fit_boosted_heteroscedastic

rng = np.random.default_rng(0)
n = 64
X = rng.normal(size=(n, 3))
noise_scale = 0.1 + 0.4 * (X[:, 0] > 0)  # heteroscedastic: some rows noisier
y = X[:, 0] + 0.5 * X[:, 1] + noise_scale * rng.normal(size=n)
log_scale = np.log(noise_scale)  # a frozen noise estimate (CatBoost, in practice)

config = SoftTreeConfig(n_features=3, n_trees=2, depth=1, task="regression", n_outputs=1, seed=0)
model, result = fit_boosted_heteroscedastic(
    X, y, config, log_scale=log_scale, weighting="gls", n_stages=3, inner_steps=5,
)
pred = model.score(X)
assert pred.shape == (n, 1)
```

This is a thin extension of `omnibias.tab.torch.boosting`
(documented in full on [`tab.md`](tab.md#newton-boosting-driver-torch));
only `fit_boosted_heteroscedastic` is new.

## Benchmark harness

`benchmarks/tabular_uncertainty.py` follows the repo's `--full` convention:
smoke (default) proves every code path end to end at reduced scale; `--full`
is the multi-seed acceptance run. G0/G1b are unconditional wiring gates; G1-G5
are recorded in every run but only gate the exit code in `--full` -- a
falsifiable scientific hypothesis is not something a reduced budget gets to
decide. G4 is a **reporting-completeness** gate (the full win/loss table
against tuned LightGBM / CatBoost / RealMLP / TabM on `>= 6` public sets),
never an aggregate-only "we win" claim -- matching theory 05-02 G3's explicit
rule. **`--full` result: G4 passed** (9/9 public sets completed; the
`fit_boosted_heteroscedastic(gls)` arm is the majority-seed winner on 8 of 36
dataset x baseline comparisons, 22% -- a mixed table, as predeclared, not a
win). See theory 05-03 for the full acceptance-gate writeup and every measured
G1-G5 result.

::: omnibias.tab.bench
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - saw_wave_2d
        - mlp_heteroscedastic_20d
        - fit_predict_catboost_uncertainty
        - fit_predict_realmlp
        - fit_predict_tabm
        - NOISE_PUBLIC_SUITE
