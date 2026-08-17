# Chapter 1 — The neural jet in 1-D

> Module: `omnibias.symbolic` (source: `omnibias.symbolic.discovery`). Precise
> signatures and parameter docs render in the [API reference](../api/symbolic.md);
> this chapter is the example-driven companion.

## What this chapter gives you

A *jet* is the stack of a function and its derivatives at a point:
\(\;\mathbf{j}(x) = \bigl(y,\ y',\ y'',\ \dots,\ y^{(N)}\bigr)\). omnibias builds
that stack in **closed form** — no finite differences, no autodiff graph — because
a random-feature field \(y(x) = \sum_h c_h\,\sigma(W_h \tilde x + \beta_h) + b\)
has an affine inner map, so

\[
  y^{(k)}(x) = \sum_h c_h\,\sigma^{(k)}(z_h)\,(W_h/s)^k,\qquad z_h = W_h\tilde x + \beta_h,
\]

and each order needs a *single* activation-tower evaluation \(\sigma^{(k)}\).

Once you have the jet, **equation discovery** is linear algebra: search for a
sparse relation among the generic coordinates \(x, y, y', y'', \dots\). No named
basis like `sin`/`exp` is supplied — recovering `dy = y` (exponential),
`d2y = -y` (sinusoid) or `dy = 1 - y²` (tanh) is a *library-free* SINDy variant.

```
 data ──fit──▶ NeuralField1D ──extract──▶ JetBundle ──discover──▶ SparseEquation
       (ridge readout)        (closed-form σ^(k))     (sparse STLSQ on jet library)
```

---

## Fields and jets

### `NeuralField1D`

The fitted (or exactly-specified) 1-D field — a frozen dataclass holding the
frozen random weights `W, beta`, the solved readout `c, b`, the input
standardisation `x_mean, x_scale`, and the activation name. You rarely build it
by hand; `fit_neural_field_1d` and `exact_activation_field_1d` return it.

**Theory.** It is a one-hidden-layer network with a *linear* output, so it is
linear in the trainable parameters `c, b` (closed-form least squares) yet
nonlinear in `x` (rich enough to fit smooth targets).

### `fit_neural_field_1d`

<!-- docs-test: signature -->
```python
fit_neural_field_1d(x, y, *, hidden=192, ridge=1e-5, activation="tanh", seed=0, weight_scale=1.0, y_prime=None, deriv="none", deriv_weight=1.0) -> NeuralField1D
```

**What.** Fit a smooth field to samples `(x, y)` by ridge-solving the output
layer. Optional ``deriv="spline"`` / ``y_prime`` collocates the closed-form
first jet; the default is value-only and bit-identical to the historical
solve. **When.** You have noisy 1-D observations and want a differentiable
closed-form surrogate to read derivatives off.

**Theory.** Random features \(\phi_h(x) = \sigma(W_h\tilde x + \beta_h)\) with
frozen `W, beta` turn regression into the linear solve
\(\min_{c,b}\lVert \Phi c + b - y\rVert^2 + \lambda\lVert c\rVert^2\) — a random
kitchen-sink / ELM estimator that is convex and has no training loop.

```python
import numpy as np
from omnibias.symbolic import fit_neural_field_1d

x = np.linspace(-np.pi, np.pi, 400)
y = np.sin(x) + 0.01 * np.random.default_rng(0).normal(size=x.size)

field = fit_neural_field_1d(x, y, hidden=256, ridge=1e-4, seed=0)
print(field.activation, round(field.train_rmse, 4))   # tanh  ~0.01
```

### `exact_activation_field_1d`

<!-- docs-test: signature -->
```python
exact_activation_field_1d(activation) -> NeuralField1D
```

**What.** Represent a single activation \(\sigma(x)\) *exactly* as a one-neuron
field (`W=[1], beta=[0], c=[1], b=0`). **When.** You want the ground-truth jet of
an activation to discover its defining ODE (theory demos, identity certification).

```python
from omnibias.symbolic import exact_activation_field_1d, extract_neural_jets
import numpy as np

field = exact_activation_field_1d("tanh")
jets = extract_neural_jets(field, np.linspace(-1, 1, 5), max_order=1)
# column 0 is tanh(x), column 1 is its exact derivative 1 - tanh(x)^2
print(np.allclose(jets.jets[:, 1], 1 - jets.jets[:, 0] ** 2))   # True
```

### `extract_neural_jets`

<!-- docs-test: signature -->
```python
extract_neural_jets(field, x, *, max_order=3) -> JetBundle
```

**What.** Evaluate the closed-form jet `[y, dy, d2y, …]` of a `NeuralField1D` at
points `x`. **When.** Always — it is the bridge from a field to discovery.

**Theory.** Implements \(y^{(k)} = \sum_h c_h\,\sigma^{(k)}(z_h)(W_h/s)^k\). One
fastpath call per order; `max_order < 0` raises `ValueError`, and an activation
without a derivative fastpath raises `TypeError`.

```python
import numpy as np
from omnibias.symbolic import fit_neural_field_1d, extract_neural_jets

x = np.linspace(-2, 2, 300)
field = fit_neural_field_1d(x, np.exp(x), hidden=256, seed=0)
bundle = extract_neural_jets(field, x, max_order=2)
print(bundle.jets.shape)        # (300, 3): y, dy, d2y
```

### `JetBundle`

The samples `x` plus the `(n, max_order+1)` array `jets`. Helper `.name(order)`
returns the human label (`y`, `dy`, `d2y`, …). This is the input type the
discoverer consumes (one bundle each for train / val / test).

```python
bundle.name(0), bundle.name(1), bundle.name(2)   # ("y", "dy", "d2y")
```

### `jet_name`

<!-- docs-test: signature -->
```python
jet_name(order) -> str
```

**What.** The canonical label for a derivative order: `0→"y"`, `1→"dy"`,
`k→"d{k}y"`. **When.** Pretty-printing discovered equations and building library
column names.

### `split_x_grid`

<!-- docs-test: signature -->
```python
split_x_grid(*, xmin=-1.0, xmax=1.0, n_train=160, n_val=120, n_test=120) -> (train, val, test)
```

**What.** Three deterministic, slightly inset 1-D grids for train/val/test.
**When.** You need a leakage-free three-way split for discovery demos and want a
near-uniform grid (required by the fractional features below).

```python
from omnibias.symbolic import split_x_grid
xtr, xva, xte = split_x_grid(xmin=-np.pi, xmax=np.pi)
```

---

## The discoverer

### `NeuralJetDiscoverer`

<!-- docs-test: signature -->
```python
NeuralJetDiscoverer(max_library_degree=2, alphas=..., thresholds=..., complexity_weight=2e-3,
                    cdf_feature_bases=(), info_feature_bases=(), fractional_orders=(),
                    divergence_objective=None, ...)
```

**What.** The library-free differential-identity search. Its `.discover(train,
val, test, *, candidate_lhs_orders=None, selection_criterion=None)` tries each
derivative order as the left-hand side, builds a polynomial library from the
*other* (lower-order) jet coordinates, fits a sparse equation over a
ridge/threshold grid, and selects on validation RMSE + complexity (or an
information criterion when `selection_criterion` is set; see *Coefficient
uncertainty & model selection* below). **When.** You want the ODE/identity behind
a fitted field.

**Theory.** For a true relation \(y^{(\ell)} = F(x, y, \dots, y^{(\ell-1)})\) the
residual is white; the search minimises validation RMSE plus a parsimony penalty
`complexity_weight * (#active terms)`. Restricting the right side to *lower*
orders avoids trivial derivative-shift identities. The opt-in
`cdf_feature_bases` / `info_feature_bases` / `fractional_orders` enrich the
library with probability, information, and non-local atoms (see below);
`divergence_objective` lets information theory co-drive selection (Chapter 5).

```python
from omnibias.symbolic import (
    exact_activation_field_1d, extract_neural_jets, NeuralJetDiscoverer, split_x_grid,
)

xtr, xva, xte = split_x_grid(xmin=-2.0, xmax=2.0)
field = exact_activation_field_1d("exp")          # the exponential, represented exactly
mk = lambda x: extract_neural_jets(field, x, max_order=2)
res = NeuralJetDiscoverer(max_library_degree=1).discover(
    mk(xtr), mk(xva), mk(xte), candidate_lhs_orders=(1,)
)
print(res.formula())            # "dy = 1*y"  (the exponential ODE)
```

When you instead fit a field to *finite, noisy* data the recovered coefficients
are close but not exactly integer (e.g. `dy = 0.997*y + …`); see
`discover_from_noisy_observations` below for that realistic path.

### `JetDiscoveryResult`

The frozen result: `lhs_order`, the fitted `equation` (a `SparseEquation`),
`validation_rmse`, `test_rmse`, `selection_score`, `target_scale`, and a
`diagnostics` dict (entropy / divergence / mutual-information of the residuals —
Chapter 5). `.formula()` renders the equation with the right LHS label;
`.active_terms()` lists the surviving `{name, coefficient}` rows.

```python
res.lhs_order, round(res.test_rmse, 8)     # (1, ~1e-12)
res.active_terms()                          # [{"name": "y", "coefficient": 1.0}]
```

### `SparseEquation`

The fitted sparse linear law: `term_names`, `coefficients`, `intercept`, and the
hyper-parameters used, plus optional `coefficient_ci` / `coefficient_intervals` /
`selection_frequency` (populated by `attach_uncertainty`; see *Coefficient
uncertainty & model selection*). Methods: `.predict(design)`,
`.active_terms(min_abs=1e-10)`, `.formula(lhs="y", digits=4)`, and
`.uncertainty_formula(lhs="y")`.

**Theory.** It encodes \(\text{lhs} = \text{intercept} + \sum_k \beta_k\,
\text{term}_k\); only `|β_k|` above threshold survive, so the formula is compact.

```python
eq = res.equation
print(eq.formula(lhs="dy"))      # "dy = 1*y"
# `predict` takes a design matrix with one column per fitted term.
design = np.ones((3, len(eq.term_names)))
print(eq.predict(design))
```

### `discover_activation_identity`

<!-- docs-test: signature -->
```python
discover_activation_identity(activation, *, x_range=(-1.0, 1.0), max_order=3,
                             candidate_lhs_orders=None) -> JetDiscoveryResult
```

**What.** End-to-end: take an activation, build its exact one-neuron jet on a
grid, and discover its defining ODE. **When.** The fastest way to see the engine
work and to certify activation identities.

```python
from omnibias.symbolic import discover_activation_identity

print(discover_activation_identity("exp", candidate_lhs_orders=(1,)).formula())  # dy = 1*y
print(discover_activation_identity("sin", candidate_lhs_orders=(2,)).formula())  # d2y = -1*y
print(discover_activation_identity("tanh", candidate_lhs_orders=(1,)).formula()) # dy = 1 - y^2
```

### `discover_from_noisy_observations`

<!-- docs-test: signature -->
```python
discover_from_noisy_observations(*, seed=0, noise_std=0.01, hidden=256) -> dict
```

**What.** The realistic pipeline: fit a field to *noisy* `sin(x)` samples, then
discover `d2y = -y` from the closed-form jet. Returns the field train RMSE, the
discovered equation, the jet test RMSE, and the RMSE against the true identity.
**When.** A reference for "from messy data to a law" you can adapt to your own.

```python
from omnibias.symbolic import discover_from_noisy_observations
out = discover_from_noisy_observations(noise_std=0.01)
print(out["equation"])           # e.g. "d2y = -0.98*y + ..."  (dominant term ≈ -y)
```

---

## Sparse regression core

### `fit_sparse_equation`

<!-- docs-test: signature -->
```python
fit_sparse_equation(design, target, term_names, *, alpha=1e-8, threshold=1e-4, max_iter=8, loss="ridge") -> SparseEquation
```

**What.** Sequential thresholded ridge regression (STLSQ) — the workhorse behind
every discoverer. **When.** You already have a design matrix of candidate columns
and a target, and want a sparse linear fit. **Theory.** Iterate: ridge-solve,
zero out coefficients below `threshold` (in *standardised* space, so the criterion
is scale-invariant), repeat until the support stabilises. `alpha` is the ridge
strength; the surviving coefficients are returned in raw units.

```python
import numpy as np
from omnibias.symbolic import fit_sparse_equation

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = 2.0 * X[:, 0] - 0.5 * X[:, 2]                 # column 1 is irrelevant
eq = fit_sparse_equation(X, y, ["a", "b", "c"], alpha=1e-8, threshold=1e-3)
print(eq.formula(lhs="y"))        # "y = 2*a - 0.5*c"  (b dropped)
```

### `build_jet_relation_library`

<!-- docs-test: signature -->
```python
build_jet_relation_library(bundle, *, lhs_order, max_degree=2, include_x=True, lower_order_only=True)
    -> (design, term_names)
```

**What.** Polynomial design over jet coordinates, excluding the LHS order.
**When.** Building a custom discovery loop, or inspecting which terms the engine
considers. **Theory.** Generates monomials up to `max_degree` from `x` and the
admissible jets; `lower_order_only=True` keeps the right side strictly below the
LHS order to avoid collinear derivative-shift identities.

```python
import numpy as np
from omnibias.symbolic import exact_activation_field_1d, extract_neural_jets, build_jet_relation_library

b = extract_neural_jets(exact_activation_field_1d("tanh"), np.linspace(-1, 1, 50), max_order=1)
design, names = build_jet_relation_library(b, lhs_order=1, max_degree=2)
print(names)        # ['x', 'y', 'x^2', 'x*y', 'y^2']  -> tanh picks 'y^2' (and intercept)
```

### `rmse` and `mae`

<!-- docs-test: signature -->
```python
rmse(y_true, y_pred) -> float
mae(y_true, y_pred) -> float
```

Root-mean-square and mean-absolute error — the scoring primitives used
throughout. `rmse` drives model selection; `mae` is reported alongside.

```python
from omnibias.symbolic import rmse, mae
import numpy as np
a, b = np.zeros(4), np.array([0.1, -0.1, 0.2, -0.2])
round(rmse(a, b), 4), round(mae(a, b), 4)     # (0.1581, 0.15)
```

---

## Coefficient uncertainty & model selection

A point estimate `dy = 2*y` is only half a discovery — you also need *how sure*
each coefficient is, and *which model order* the data actually supports. These
live in `omnibias.symbolic.uncertainty` (rigor on the coefficients) and
`omnibias.symbolic.selection` (rigor on the model order). Both wrap
`fit_sparse_equation`, so they apply to any sparse fit.

### `bootstrap_coefficients`

<!-- docs-test: signature -->
```python
bootstrap_coefficients(design, target, term_names, *, alpha=1e-8, threshold=1e-4,
                       n_boot=200, ci_level=0.95, seed=0) -> dict
```

**What.** Nonparametric (resampling) confidence intervals **and** per-term
*selection frequency*. **When.** You want honest error bars and a stability score
without assuming a noise model. **Theory.** Resample rows with replacement, refit
STLSQ on each, and read percentile CIs off the coefficient distribution; the
fraction of resamples in which a term survives the threshold is its
stability-selection score.

```python
import numpy as np
from omnibias.symbolic import bootstrap_coefficients

rng = np.random.default_rng(1)
X = rng.normal(size=(400, 4))                      # cols: x1 x2 x3 x1*x2
X[:, 3] = X[:, 0] * X[:, 1]
y = 2.0 * X[:, 0] - 1.5 * X[:, 2] + 0.01 * rng.normal(size=400)
boot = bootstrap_coefficients(X, y, ["x1", "x2", "x3", "x1x2"], n_boot=200, seed=7)
print(np.round(boot["ci_lower"], 3))         # ~[ 1.999, 0, -1.501, 0]
print(boot["selection_frequency"][[0, 2]])   # ~[1. 1.]  (x1, x3 always selected)
```

### `ridge_coefficient_covariance`

<!-- docs-test: signature -->
```python
ridge_coefficient_covariance(design, target, *, alpha=1e-8) -> dict
```

**What.** The analytic ridge/OLS covariance and standard errors. **When.** You
want fast parametric error bars under a homoscedastic-Gaussian noise assumption.
**Theory.** With centered columns and `A = XᵀX + alpha I`, the estimator
`c = A⁻¹Xᵀy` has covariance `σ² A⁻¹(XᵀX)A⁻¹`, with `σ²` from the residual sum of
squares over `n − p` degrees of freedom.

```python
from omnibias.symbolic import ridge_coefficient_covariance
cov = ridge_coefficient_covariance(X, y, alpha=1e-10)
print(np.round(cov["coefficients"], 3))      # ~[ 2. 0. -1.5 0.]
print(np.round(cov["std_errors"], 4))        # small, ~1e-3
```

### `certified_coefficient_intervals`

<!-- docs-test: signature -->
```python
certified_coefficient_intervals(design, target, *, alpha=0.0) -> dict
```

**What.** *Rigorous* interval enclosures of the exact normal-equation solution —
the bridge from discovery to the verified substrate
(`omnibias.core.verified.linalg`). **When.** You need a theorem-grade box that
*provably* contains the true coefficient vector, not just a statistical CI.
**Theory.** With a float approximate inverse `B` and solution `c0`,
\(\lVert c^\* - c_0\rVert_\infty \le \lVert A^{-1}\rVert_\infty\,\lVert b - A c_0\rVert_\infty\);
the inverse-norm is certified by the Neumann lemma and the residual is bounded in
outward-rounded interval arithmetic. For `alpha=0` and exact data, `c*` is the
generating vector, so the box contains the ground truth.

```python
from omnibias.symbolic import certified_coefficient_intervals
y_exact = 2.0 * X[:, 0] - 1.5 * X[:, 2]      # noiseless
cert = certified_coefficient_intervals(X[:, [0, 2]], y_exact, alpha=0.0)
print(cert["certified"], cert["radius"] < 1e-9)   # True True
lo, hi = cert["intervals"][0]
print(lo <= 2.0 <= hi)                            # True  (truth enclosed)
```

### `attach_uncertainty` and the enriched `SparseEquation`

<!-- docs-test: signature -->
```python
attach_uncertainty(equation, design, target, *, bootstrap=True, certified=True,
                   n_boot=200, ci_level=0.95, seed=0) -> SparseEquation
```

**What.** Run the chosen layers and return a copy of the fit carrying
`coefficient_ci`, `selection_frequency`, and `coefficient_intervals` (each
aligned to `term_names`). `SparseEquation.active_terms()` then includes the bounds
per row, and `.uncertainty_formula()` renders `±` annotations.

```python
from omnibias.symbolic import attach_uncertainty, fit_sparse_equation
eq = fit_sparse_equation(X, y, ["x1", "x2", "x3", "x1x2"], threshold=1e-2)
eq = attach_uncertainty(eq, X, y, n_boot=120, seed=2)
print(eq.uncertainty_formula())              # "y = (2 +/- 0.001)*x1 + (-1.5 +/- 0.001)*x3"
```

### Information criteria — `aic`, `aicc`, `bic`, `mdl`

<!-- docs-test: signature -->
```python
aic(n, rss, k); aicc(n, rss, k); bic(n, rss, k); mdl(n, rss, k, n_candidates=None)
information_criterion(name, n, rss, k, *, n_candidates=None)
equation_information_criterion(equation, design, target, *, name="bic")
```

**What.** Penalised-likelihood model scores on the common deviance scale
`−2 log L + penalty` (lower is better). **When.** Choosing model order without a
held-out set. **Theory.** Under Gaussian residuals the log-likelihood is a
function of the residual sum of squares; AIC penalises `2k`, BIC `k·ln n`, and
MDL adds a `2·ln C(p,k)` structure-coding term for large libraries.

```python
import numpy as np
from omnibias.symbolic import equation_information_criterion, fit_sparse_equation

rng = np.random.default_rng(12)
xx = rng.uniform(-2, 2, size=400)
yy = 1.0 + 2.0 * xx**2 + 0.01 * rng.normal(size=400)        # true order 2
full = np.stack([xx, xx**2, xx**3, xx**4], axis=1)
names = ["x", "x^2", "x^3", "x^4"]
eq2 = fit_sparse_equation(full[:, :2], yy, names[:2], threshold=1e-8)
eq4 = fit_sparse_equation(full, yy, names, threshold=1e-8)
b2 = equation_information_criterion(eq2, full[:, :2], yy, name="bic")
b4 = equation_information_criterion(eq4, full, yy, name="bic")
print(b2 <= b4)                              # True — BIC prefers the order-2 model
```

### `kfold_select` and `stability_selection`

<!-- docs-test: signature -->
```python
kfold_select(design, target, term_names, *, alphas=..., thresholds=..., k=5, seed=0) -> KFoldSelection
stability_selection(design, target, term_names, *, alpha, threshold,
                    n_resample=100, sample_fraction=0.5, seed=0) -> dict
```

**What.** `kfold_select` picks the most *predictive* `(alpha, threshold)` by
cross-validated RMSE; `stability_selection` ranks terms by Meinshausen–Bühlmann
subsampling frequency. **When.** Hyper-parameter choice and robust term ranking.

```python
from omnibias.symbolic import kfold_select, stability_selection
sel = kfold_select(X, y, ["x1", "x2", "x3", "x1x2"], k=5, seed=14)
print(sel.alpha, sel.threshold, round(sel.cv_rmse, 4))
ss = stability_selection(X, y, ["x1", "x2", "x3", "x1x2"], alpha=1e-8,
                         threshold=1e-2, n_resample=100, seed=16)
print(ss["ranking"][:2])     # [('x1', 1.0), ('x3', 1.0)] — true terms on top
```

### Wiring selection into the discoverers

Both `NeuralJetDiscoverer.discover(..., selection_criterion="bic")` and
`discover_interpretable_surrogate(..., selection_criterion="aic")` accept an
optional `selection_criterion` (`"aic"`/`"aicc"`/`"bic"`/`"mdl"`). The default is
`None`, which preserves the validation-RMSE + complexity score; when set, models
are ranked by that information criterion on the training residuals.

---

## Probability, information & fractional library atoms

These let the discoverer express laws no polynomial can — saturating
(sigmoidal), log-likelihood (surprisal), and non-local (fractional) terms. Each
comes as a **fit-on-train, apply-everywhere** pair to avoid leakage.

### `gl_fractional_derivative`

<!-- docs-test: signature -->
```python
gl_fractional_derivative(y, *, alpha, h) -> np.ndarray
```

**What.** The Grünwald–Letnikov fractional derivative \(D^\alpha y\) on a uniform
grid (spacing `h`). **When.** Modelling memory / anomalous relaxation, or
discovering fractional ODEs. **Theory.** Causal convolution
\(D^\alpha y[i] = h^{-\alpha}\sum_{k} w_k\,y[i-k]\) with binomial weights
\(w_k = (-1)^k\binom{\alpha}{k}\); integer `alpha` reduces to backward finite
differences, so it interpolates *between* derivative orders. Non-local: each
output depends on the whole history. `alpha < 0` or `h <= 0` raises.

```python
import numpy as np
from omnibias.symbolic import gl_fractional_derivative

x = np.linspace(0, 4, 400); h = x[1] - x[0]
half = gl_fractional_derivative(np.exp(x), alpha=0.5, h=h)   # the half-derivative of e^x
print(half.shape)        # (400,)
```

### `build_jet_fractional_features`

<!-- docs-test: signature -->
```python
build_jet_fractional_features(bundle, *, orders, source_order=0) -> (design, names)
```

**What.** Add fractional-derivative columns \(D^\alpha v\) of a jet signal (by
default `y`) for each `alpha` in `orders`, over the bundle's (uniform, sorted)
`x`. **When.** Enabling fractional-law discovery — or pass `fractional_orders=...`
to `NeuralJetDiscoverer` to wire it in automatically.

```python
import numpy as np
from omnibias.symbolic import exact_activation_field_1d, extract_neural_jets, build_jet_fractional_features

b = extract_neural_jets(exact_activation_field_1d("exp"), np.linspace(0, 2, 200), max_order=1)
cols, names = build_jet_fractional_features(b, orders=(0.25, 0.5, 0.75))
print(names)        # ['D^0.25(y)', 'D^0.5(y)', 'D^0.75(y)']
```

### CDF (probability) jet features — `fit_jet_cdf_plan` / `build_jet_cdf_features`

`fit_jet_cdf_plan(train, *, lhs_order, bases=("sigmoid","tanh"), …)` fits
per-variable locations (train quantiles) and scales (train-std multiples);
`build_jet_cdf_features(bundle, …)` then maps every right-hand jet variable
through the monotone CDF bases \(F((v-\text{loc})/\text{scale})\) on that *fixed*
grid. Together they let the discoverer recover saturating laws like
`dy = σ((x - loc)/scale)`. Enable via `NeuralJetDiscoverer(cdf_feature_bases=...)`.

### Surprisal (information) jet features — `fit_jet_info_plan` / `build_jet_info_features`

The information-theoretic twins: the same leakage-free grid, but each variable is
mapped through the *surprisal* \(-\ln f((v-\text{loc})/\text{scale})\) of each
base density. These express log-likelihood / energy structure. Enable via
`NeuralJetDiscoverer(info_feature_bases=...)`.

```python
import numpy as np
from omnibias.symbolic import (
    exact_activation_field_1d, extract_neural_jets,
    fit_jet_cdf_plan, build_jet_cdf_features,
)
b = extract_neural_jets(exact_activation_field_1d("tanh"), np.linspace(-2, 2, 120), max_order=2)
locs, scales = fit_jet_cdf_plan(b, lhs_order=2, bases=("sigmoid",), n_locations=3)
design, names = build_jet_cdf_features(b, lhs_order=2, bases=("sigmoid",),
                                       per_variable_locations=locs, per_variable_scales=scales)
print(design.shape[1], "CDF columns")
```

---

## AutoML surrogate modelling

The other face of the module: pick the best compact representation of a tabular
target among several **candidate libraries**.

### `discover_interpretable_surrogate`

<!-- docs-test: signature -->
```python
discover_interpretable_surrogate(data, specs=None, *, include_cdf_band=True,
                                 include_information=False, divergence_objective=None, ...) -> dict
```

**What.** Auto-select the best sparse surrogate family on validation data, refit
on train+val, and report the equation, selected terms, test metrics, the
selection trail, and residual diagnostics. **When.** "Give me an interpretable
formula for this `SplitData`." **Theory.** For each candidate family it sweeps
`(alpha, threshold)`, scores by `val_rmse + complexity_weight·#terms` (plus an
optional information/transport `divergence_objective`, Chapter 5), and keeps the
winner.

```python
from omnibias.symbolic import make_symbolic_regression_dataset, discover_interpretable_surrogate
data = make_symbolic_regression_dataset(seed=0)
out = discover_interpretable_surrogate(data)
print(out["family"], "->", out["equation"])
print(round(out["metrics"]["rmse"], 4))
```

### Library builders

<!-- docs-test: signature -->
```python
build_taylor_library(x, *, max_degree=2)            # polynomial monomials
build_fourier_library(x, *, max_frequency=2)        # per-coordinate sin/cos modes
build_hybrid_library(x, *, max_degree=2, max_frequency=2)   # the union
```

Stateless `(design, names)` builders for the three default families. Use directly
for a custom search, or rely on `default_surrogate_specs` to wrap them.

```python
import numpy as np
from omnibias.symbolic import build_hybrid_library
X = np.random.default_rng(0).uniform(-1, 1, size=(8, 2))
design, names = build_hybrid_library(X, max_degree=2, max_frequency=1)
print(names[:5])
```

### `default_surrogate_specs` and `LibrarySpec`

`LibrarySpec(name, builder, description)` bundles a named feature transform;
`default_surrogate_specs(max_degree=2, max_frequency=2)` returns the
Taylor / Fourier / hybrid trio. Pass a custom list of specs to
`discover_interpretable_surrogate` to control the search space.

### CDF / information surrogate libraries

`build_cdf_band_library(x, *, bases, locations, scales)` is the stateless
location-scale CDF design (a difference of two columns is a slab probability);
`fit_cdf_band_library_plan(x_train, …)` returns a *train-fitted* `LibrarySpec`
(quantile locations, std scales) — the leakage-free way to feed CDF features into
the surrogate search. `build_information_library` / `fit_information_library_plan`
are the surprisal twins (`-ln f` features, whose model expectation is entropy).

```python
import numpy as np
from omnibias.symbolic import build_cdf_band_library, build_information_library
X = np.random.default_rng(0).uniform(-2, 2, size=(16, 1))
cdf, _ = build_cdf_band_library(X, bases=("sigmoid",), locations=np.array([0.0]), scales=np.array([1.0]))
info, _ = build_information_library(X, bases=("arctan",), locations=np.array([0.0]), scales=np.array([1.0]))
print(cdf.shape, info.shape)
```

### `FeatureLibraryPlan` and `fit_screened_feature_library_plan`

For high-dimensional tables, `fit_screened_feature_library_plan(x_train, y_train,
…)` screens raw / squared / `sin` / pairwise-product candidates by train
correlation and returns a `FeatureLibraryPlan` whose `.transform(x)` reproduces
the chosen columns on any split — the front end for sparse recovery in many
irrelevant dimensions.

---

## Datasets, metrics and evaluators

These synthesise reproducible problems and run end-to-end validations.

- **`make_symbolic_regression_dataset(*, n_samples=900, noise_std=0.02, seed=0)`** →
  `SplitData` for the hidden law `y = 1.5·x1² − 2·x2·x3 + sin(2·x4) + 0.4·cos(x4)`.
  **`symbolic_hidden_law(x)`** evaluates that target directly.
- **`SplitData`** — the `(x_train, y_train, x_val, y_val, x_test, y_test)` container
  every surrogate routine consumes.
- **`make_high_dim_sparse_dataset(*, n_features=60, …)`** → `(SplitData, hidden)` with
  only ~5 active features among many; **`evaluate_high_dim_sparse_validation(…)`**
  runs the screened sparse recovery and reports recovery rate / false positives.
- **`make_heat_equation_operator_data(*, diffusivity=0.12, …)`** builds exact
  `u, u_x, u_xx` columns of a two-mode heat field; **`build_pde_operator_library(u,
  ux, uxx)`** assembles the candidate operator design; **`discover_pde_operator_law(…)`**
  recovers `u_t = k·u_xx` from them. (The multivariate, field-based version is
  Chapter 2.)
- **`evaluate_poc(…)`** runs the whole proof-of-concept (surrogate + operator +
  high-dim + neural-jet identities + noisy field); **`evaluate_real_world_tabular_validation(dataset=…)`**
  runs the sparse auto-regressor on an sklearn dataset; **`write_artifacts(results, out_dir)`**
  dumps `metrics.json` + a `report.md`.

```python
from omnibias.symbolic import make_high_dim_sparse_dataset, evaluate_high_dim_sparse_validation
data, hidden = make_high_dim_sparse_dataset(n_features=60, seed=0)
report = evaluate_high_dim_sparse_validation(n_features=60, seed=0)
print("recovery rate:", report["recovery_rate"])         # 1.0 — all hidden terms found
print("false positives:", report["false_positive_count"])
```

```python
from omnibias.symbolic import discover_pde_operator_law
print(discover_pde_operator_law(diffusivity=0.12)["equation"])   # "u_t = 0.12*u_xx"
```

---

**Next:** [Chapter 2 — Vector calculus & PDE discovery](02-vector-calculus-pde.md)
lifts everything here from one input to many, unlocking gradients, Laplacians,
and genuine PDE laws.
