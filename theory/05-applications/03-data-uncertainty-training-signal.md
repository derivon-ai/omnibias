# 05-03 Data uncertainty as a training signal (tabular)

## 1. Thesis and status

A frozen, independent estimate of per-row aleatoric (data) noise can be turned
into an **exact curvature term** for `omnibias-tab`'s exact-Newton trainer and
into a **closed-form soft-binning feature embedding**, both aimed at the
specific regime -- high label-noise rows -- where Kartashev, Rubachev and
Babenko (arXiv:2509.04430) show the GBDT-vs-tabular-DL performance gap
actually lives.

- **Status**: gated (`benchmarks/tabular_uncertainty.py` exists and has run
  `--full` end to end, G0-G5 -- see section 8 for every measured number.
  **All three falsifiable hypothesis gates failed**: G1 (noise-damped exact
  Newton), G2 (the band-embedding pullback-metric loss), and G3 (GLS-reweighted
  boosting) each miss their predeclared bar at the tested budgets. **G4
  passed** (the public-suite report is complete on all 9 sets, a
  reporting-completeness gate, not a win claim) and **G5 failed** (3 of 9
  public sets regress beyond their own across-seed noise). G0/G1b passed
  throughout, confirming every mechanism's plumbing and math are correct --
  the code is retained, correct, and tested; none of it is claimed as a
  benchmarked win.)
- **Depends on**: 01-13 (the `band` / `integral` operator roles), 04-02
  (`GuaranteeKind`, the discipline that a model-based estimate is never a
  sound enclosure), 05-02 (the `omnibias.tab` package this extends)
- **Blocks**: none

This spec is deliberately sequenced **falsifier-first**: Gate G1 (section 8)
is designed to be cheap and to run before any of the Phase 2 machinery
(sections 6(b), 6(d)) is built. If G1 fails, Phase 2 is re-scoped rather than
built anyway and softened into "promising future work" -- the same discipline
05-02 section 11 states for its own falsifiers. **G1 ran first (5 seeds,
`--full`) and failed on both datasets**; per the predeclared rule at the end
of section 8, Proposal A (section 4(b), `fit_noise_aware`) was recorded as
falsified there. Phase 2 proceeded anyway because `fit_boosted_heteroscedastic`
(section 4(d)) and the band/integral embedding (section 6(b)) are
architecturally independent of `fit_noise_aware` -- built, then run through
their *own* predeclared falsifiers (G2, G3) rather than assumed to inherit
G1's fate either way. **G2 and G3 have since also run `--full` and also
failed** (section 8). The honest reading across all three: this spec's central
empirical claim -- that a frozen, independent noise estimate turned into a
training signal produces a robust, absolute win on the noisy tail -- is not
supported at the budgets and datasets tested, even though every underlying
mechanism is mathematically correct and correctly implemented (G0, G1b, and
the parity/unit tests throughout).

## 2. Where it lands

Entirely inside two existing packages; no new package.

- `omnibias.tab._core.heteroscedastic` -- new numpy module (the closed-form
  Gaussian-NLL gradient and Fisher block), placed exactly where
  `omnibias.tab._core.loss.score_grad_hess` already lives, **not** in
  `omnibias.core`. This corrects the module's originally planned location
  before any code was written, per the "verify, don't guess" rule this repo
  enforces on itself (05-02 section 3: "if the delta has shrunk... fix the
  spec before writing code"). `omnibias-core` does use numpy in places
  (`omnibias.core.ftc._lstsq`), so the reason is not a dependency
  restriction; it is audience: `score_grad_hess` and
  `gaussian_nll_grad_hess` are both per-sample loss primitives consumed only
  by `omnibias.tab`'s own boosting / natural-gradient machinery, with no
  other consumer anywhere in the repo, so they belong beside `score_grad_hess`
  rather than in the shared core every package imports.
- `omnibias.core.band_embed` -- new numpy module (vectorized soft-binning /
  soft-CDF window over a threshold grid), staying in `omnibias.core` because
  it is a direct, general-purpose lift of `omnibias.core.ftc.ftc_block`'s
  window formula (section 3) -- not tabular-specific, and the right home for
  the same reason `ftc_block` itself is there rather than in a consuming
  package.
- `omnibias.tab.torch.heteroscedastic` -- new torch module (`HeteroscedasticHead`,
  `fit_heteroscedastic`, `fit_noise_aware`).
- `omnibias.tab.torch.embed` / `omnibias.tab.jax.embed` -- new modules (Phase 2;
  `BandFeatureEmbedder` forward, bit-identical torch/jax; `local_target_consistency_loss`
  is torch-only, matching every other `omnibias.tab` trainer).
- `omnibias.tab.torch.boosting` -- extended in place with
  `fit_boosted_heteroscedastic` (Phase 2), reusing the existing per-sample
  weight hook rather than adding new machinery.
- `omnibias.tab.bench` -- extended with a regression-capable
  `train_val_test_split`, two synthetic generators with **known** ground-truth
  noise, and a CatBoost baseline.

None of this earns independent existence under the AGENTS.md rule: it is one
more training mode and one more feature map for an existing tabular package,
sharing its audience, its dependency tier, and its benchmark harness.

## 3. Prior art in omnibias

**Training.** `omnibias.tab.torch.train.fit_second_order` (`optimizer=` one of
`"trust_region"` / `"cubic"` / `"kfac"`) already trains a
`SoftTreeEnsemble` (`omnibias.tab.torch.model`) by calling
`omnibias.torch.optim.TrustRegionNewtonCG` / `CubicNewton` with a `closure()`
that recomputes the task loss with its autograd graph intact. The optimizer
itself is **functional**: it forms curvature matrix-free from whatever scalar
`closure()` returns, via `_CurvatureOptimizer._grad(loss, create_graph=True)`
and one further reverse pass in `_hvp` (`packages/omnibias-torch/src/omnibias/torch/optim.py`).
**This is the load-bearing fact for section 4(b) below: the noise-aware term
can be added entirely inside `closure()`, with zero changes to
`omnibias-torch`.**

**Boosting.** `omnibias.tab.torch.boosting.fit_boosted` already computes the
closed-form per-sample score-space gradient and Hessian
(`omnibias.tab._core.loss.score_grad_hess`: `g = 2(F-y)`, `h = 2` for
regression), forms the Newton leaf target `-g/h`, and fits each stage's weak
learner by a **weighted** least-squares call,
`_fit_weak_learner(config, X, target, weight, ...)`, where `weight` is
currently always the Hessian `h`. That parameter is already the exact hook
Phase 2's `fit_boosted_heteroscedastic` needs (section 6(c)): no new
machinery, just a different `weight`.

**The band / integral role.** `omnibias.core.ftc.ftc_block(x, w, b_lo, b_hi)`
already computes, for a *scalar* input, the literal window difference
`integral = softplus(w x + b_hi) - softplus(w x + b_lo)` and
`deriv = w * (sigmoid(w x + b_hi) - sigmoid(w x + b_lo))`. The tensor twin is
`OperatorBlock(op="band")` / `OperatorBlock(op="integral")`
(`packages/omnibias-torch/src/omnibias/torch/blocks/operator.py`,
mirrored in `omnibias.keras.blocks`), which computes exactly the same formula
over a channel axis with learnable `(b_lo, b_hi)` (`band`) or `(center, width)`
(`integral`, `width = softplus(raw_width)` so the window is always ordered).
`docs/operator-surface.md` names this the `band` (K=2, order 0) and `integral`
(K=2, order 0) roles of the six-role `OperatorBlock` catalog.
**Confirmed gap**: no vectorized numpy soft-binning / soft-histogram primitive
exists anywhere in `omnibias-core` (searched `scan.py`, `multipack.py`,
`spectral_design.py`, `frames.py`, `ftc.py`, `integral_kernel.py`); the scalar
formula exists twice (`ftc_block`, `OperatorBlock`) but never as a batched
threshold-grid lift over a feature axis. Section 6(b) is that lift -- it must
reuse the formula, not fork it.

**Certificates and uncertainty typing.** `omnibias.tab.certify.certify_tab` /
`TabCertificate` already give sound output-bound / Lipschitz / monotonicity
certificates for a `SoftTreeEnsemble`, and `omnibias.core.uncertainty`
(theory 04-02) already defines `GuaranteeKind.{SOUND_ENCLOSURE, CONFORMAL,
MODEL_BASED}` with `__add__` raising across kinds. **Confirmed gap**: nothing
in the repo produces a `MODEL_BASED` per-row aleatoric estimate today, so
there is no risk yet of it being confused with a sound enclosure -- and this
spec's job, per the 04-02 precedent, is to make sure that stays true once one
exists.

**Benchmark harness.** `omnibias.tab.bench` has `Dataset`, `load_dataset`,
`train_test_split`, `fit_predict_lightgbm`, `score_predictions`, and three
named suites. **Confirmed gap, precisely bounded**:
`train_val_test_split` raises `ValueError` for `ds.task == "regression"`
(`bench.py`, the stratified-split guard), only two regression datasets exist
(`diabetes`, `california_housing`), and `fit_predict_lightgbm`'s tuned
counterpart in `benchmarks/tabular_arrangement.py` (`_fit_lightgbm`) is
`LGBMClassifier`-only. There is no CatBoost, RealMLP, TabM, or pytabkit
baseline of any kind. Section 9 is real, not incremental, work.

## 4. Mathematics

### (a) The heteroscedastic Gaussian and its exact Fisher block

Following the source paper's parameterization exactly: model
`y_i | x_i ~ N(u(x_i), s(x_i)^2)` with `s(x) = e^{v(x)}`, so `v = log s` is
unconstrained. The per-sample negative log-likelihood in `(u, v)` is

```
NLL(u, v; y) = 0.5 log(2 pi) + v + 0.5 (y - u)^2 e^{-2v}
```

Differentiating once:

```
d NLL / du = -(y - u) e^{-2v} = -(y - u) / s^2
d NLL / dv = 1 - (y - u)^2 e^{-2v} = 1 - (y - u)^2 / s^2
```

Differentiating twice (the **observed** Hessian, still a function of the
realized `y`):

```
d^2 NLL / du^2   = e^{-2v}                       = 1 / s^2
d^2 NLL / du dv  = 2 (y - u) e^{-2v}              = 2 (y - u) / s^2
d^2 NLL / dv^2   = 2 (y - u)^2 e^{-2v}             = 2 (y - u)^2 / s^2
```

The observed Hessian is not positive-definite in general (the cross term
changes sign with the residual). Taking the expectation over
`y ~ N(u, s^2)` -- i.e. the **Fisher information**, using
`E[y - u] = 0` and `E[(y-u)^2] = s^2` under the model -- every cross term
vanishes and the diagonal collapses to constants:

```
Fisher(u, v) = diag(1 / s^2, 2)
```

exactly, with no approximation beyond the model being correctly specified.
This is the block `gaussian_nll_grad_hess` returns: the two exact gradients,
plus this exact `2 x 2` (per-sample) Fisher block, which is positive-definite
by inspection and safe to use as a natural-gradient metric where the observed
Hessian is not.

No collapse limit (founding or temperature) appears in this subsection --
it is a plain likelihood computation.

### (b) The zero-gradient Gauss-Newton penalty (the falsifier's mechanism)

Let `phi(x; theta)` be a model's score. Fix a frozen, per-row noise estimate
`s_i > 0` (section 4(c)) and a scalar `lam >= 0`. At the *start* of every
`closure()` call, define the constant

```
c_i := phi(x_i; theta_now)          (detached; no gradient flows through c_i)
```

and add the term

```
P(theta) = (lam / n) * sum_i s_i^2 * (phi(x_i; theta) - c_i)^2
```

to the task loss, where `theta_now` is the parameter vector at the moment
`closure()` is called (a fresh `c_i` every call -- this is not a one-time
regularizer, it is recomputed exactly once per optimizer step, matching the
existing closure contract in `omnibias.torch.optim`).

**Lemma.** At `theta = theta_now`, `P = 0` exactly, `grad_theta P = 0`
exactly, and `Hess_theta P = lam * Sigma_noise` exactly, where

```
Sigma_noise := (2 / n) * sum_i s_i^2 * g_i g_i^T,      g_i := d phi(x_i; theta_now) / d theta
```

*Proof.* Write `r_i(theta) = phi(x_i;theta) - c_i`. By construction
`r_i(theta_now) = phi(x_i;theta_now) - c_i = 0` for every `i`, exactly (not to
first order -- `c_i` is defined to make this hold). Then
`P(theta_now) = (lam/n) sum_i s_i^2 r_i(theta_now)^2 = 0`. Differentiating,
`dP/dtheta = (2 lam/n) sum_i s_i^2 r_i(theta) g_i(theta)`, which vanishes at
`theta_now` because every `r_i(theta_now) = 0` -- again exactly. Differentiating
once more, `d^2P/dtheta^2 = (2 lam/n) sum_i s_i^2 [g_i g_i^T + r_i * H_i]`
where `H_i = d^2 phi_i/dtheta^2`; at `theta_now` the second term vanishes
because `r_i(theta_now) = 0` exactly, leaving
`Hess P|_{theta_now} = (2 lam/n) sum_i s_i^2 g_i g_i^T = lam Sigma_noise`. QED.

So adding `P` to the task loss inside `closure()` changes **only the
curvature the trust-region / cubic-Newton optimizer sees**, at exactly the
point it evaluates it, and changes nothing else: the gradient the optimizer
follows is the task loss's own gradient, undamped. `lam = 0` makes `P`
identically the zero function, so `fit_noise_aware(..., lam=0.0)` must
reproduce `fit_second_order` bit-for-bit -- gate G0, true by construction and
checked by exact-equality test, not by a numerical tolerance.

**Why `+s_i^2`, not `+1/s_i^2`.** A Newton / trust-region step is
`theta <- theta - H^{-1} g` (or the trust-region analogue). Adding
`lam Sigma_noise` to `H` *shrinks* the step specifically along directions
`g_i` belonging to **high-noise** rows (`s_i` large) -- exactly "damp the step
along directions the noise makes untrustworthy," the mechanism named in the
thesis. This is the opposite weighting from a Fisher / GLS reweighting of the
*primary* loss (which would use `1/s_i^2` to trust low-noise rows more in the
objective itself, section 4(c)/(d)); the two are complementary mechanisms
touching different parts of the optimizer and must not be conflated (section
10 returns to this).

No collapse limit is invoked here either: `lam` is a fixed, chosen scalar,
not a limit of anything. The gate anneal `beta_init -> beta_final` already
present in `SoftTreeConfig` is the (unrelated) temperature collapse and is
untouched by this mechanism.

### (c) The independent estimator, and why it must be independent

`s_hat(x)` is estimated by a **frozen, train-only-fit, architecturally
distinct** model -- CatBoost with `loss_function="RMSEWithUncertainty"` --
never by an `omnibias.tab` model judging its own noise. `CatBoost.predict(X)`
under that loss returns a `(n, 2)` raw array whose columns are
`(mean, w)` with the variance `s^2 = e^{w}`, so

```
log_scale(x) = w(x) / 2      (since s = e^{w/2})
```

This is the estimator whose independence makes the top-uncertainty-decile
evaluation in gate G1 non-circular: if the same model produced both the noise
estimate and the prediction being evaluated, "wins where it says it is
uncertain" would be closer to tautology than to evidence.

### (d) The GLS reweighting used by Newton boosting (a different, older mechanism)

`fit_boosted_heteroscedastic` (section 6(c)) does not use the penalty of
4(b). It reuses the existing Hessian-weighted weak-learner fit in
`_fit_weak_learner`, replacing the weight `h_i` with `h_i / s_hat_i^2`: a
standard heteroscedastic / generalized-least-squares reweighting of the
boosting target (in the spirit of NGBoost, Duan et al. 2020, already cited by
the source paper). This *does* change the fitted function (low-noise rows
get more say in each weak learner), unlike 4(b)'s zero-gradient penalty. It
is included because the hook already exists and the change is three lines,
not because it is a new idea -- section 10 says this plainly.

### (e) Soft-binning as a batched band/integral window

For one numerical feature `x` and `J` interior thresholds
`t_1 < ... < t_J` (padded with `t_0 = -inf`, `t_{J+1} = +inf`), define

```
band_embed(x, t)[j]     = sigma(beta (x - t_j)) - sigma(beta (x - t_{j+1}))          j = 0..J
integral_embed(x, t)[j] = S(beta (x - t_j))     - S(beta (x - t_{j+1})), S' = sigma   j = 0..J
```

This is literally `J+1` copies of the `band` / `integral` role's formula
(section 3) sharing one direction `w = beta` per feature, evaluated over a
threshold grid instead of a single learned window -- a batched **lift**, not
a new formula. Two identities that must hold as unit tests, both exact:

```
sum_j band_embed(x, t)[j] = sigma(beta(x - t_0)) - sigma(beta(x - t_{J+1})) = 1 - 0 = 1   (telescoping)
d/dx integral_embed(x, t) = beta * band_embed(x, t)                                       (FTC, per ftc_block)
```

`beta -> inf` hardens each row into a one-hot hard-histogram bin indicator --
**temperature collapse**, the same feasibility sense as `SoftTreeConfig`'s
gate anneal, not the founding `delta -> 0` bias collapse. The window itself
has a **fixed, finite** gap `t_{j+1} - t_j`, which is the opposite of the
founding collapse, exactly as 04-02 states for the same role; this is the
second spec to build on that role and repeats the disclaimer rather than
assuming the reader remembers it.

### (f) Local target consistency as a pullback-metric condition

The source paper's section 5.1 explains numerical embeddings' disproportionate
win in high-uncertainty regions by **local target consistency**: neighbors in
embedding space should have similar targets. They *measure* this with a
k-NN target-difference curve and *train* it with a sampled triplet loss.

omnibias's closed-form angle: let `e = embed(x): R^d -> R^D` (the embedder,
with a closed-form Jacobian `J = de/dx` because `band_embed` / `integral_embed`
differentiate in closed form) and let `df/dx` be the clean target's gradient.
"Neighbors in embedding space have similar targets" is, to first order, the
statement that the embedding's pullback metric `J^T J` should not suppress
the direction `df/dx` -- equivalently, the Rayleigh quotient

```
R(x) = (df/dx)^T J^T J (df/dx)  /  ||df/dx||^2
```

should be large relative to the pullback metric's other directions.
`local_target_consistency_loss` maximizes (a numerically stable surrogate of)
`R` averaged over `x`, which is differentiable in closed form given `J` and
`df/dx` -- no sampled triplets. **This requires a usable `df/dx`**, which is
not available for arbitrary tabular data; section 10 states exactly where this
gate can and cannot be run.

## 5. Worked example

**(i) The zero-gradient penalty, by hand.** Toy model `phi(x;theta) =
theta_0 + theta_1 x` (`P = 2` parameters), `n = 3` rows
`x = (-1, 0, 2)`, noise `s = (1, 2, 0.5)`, `theta_now = (0, 1)` so
`phi = (-1, 0, 2)` and every `c_i = phi(x_i;theta_now)` by construction, so
`r_i(theta_now) = 0` for all three rows -- confirming `P(theta_now) = 0`
before touching a computer. The per-row Jacobian `g_i = d phi_i/d theta =
(1, x_i)` is `(1,-1)`, `(1,0)`, `(1,2)`. Then

```
Sigma_noise = (2/3) * [ 1*(1,-1)(1,-1)^T + 4*(1,0)(1,0)^T + 0.25*(1,2)(1,2)^T ]
            = (2/3) * [ [[1,-1],[-1,1]] + [[4,0],[0,0]] + [[0.25,0.5],[0.5,1]] ]
            = (2/3) * [[5.25, -0.5], [-0.5, 2]]
            = [[3.5, -1/3], [-1/3, 4/3]]
```

a symmetric positive-definite `2x2` matrix, computable in ten lines of numpy
and matched exactly (not approximately) by finite-differencing `P` around
`theta_now` -- gate G1b.

**(ii) The CatBoost conversion, by hand.** If
`CatBoostRegressor(loss_function="RMSEWithUncertainty").predict(x)` returns
raw `[3.00, -1.3863]` for a row, then `s_hat = exp(-1.3863 / 2) = 0.500` and
`log_scale = -0.6931`, since `exp(-1.3863) = 0.250 = 0.5^2`.

**(iii) `band_embed`, by hand.** One feature, thresholds `t = (-1, 0, 1)`,
`beta = 4`, `x = 0.3`. The four padded edges are
`(-inf, -1, 0, 1, +inf)`, giving sigmoid arguments
`beta(x - t) = (+inf, 5.2, 1.2, -2.8, -inf)` and
`sigma(.) = (1, 0.9945, 0.7685, 0.0573, 0)`. Consecutive differences:

```
band_embed(0.3, t) = (1 - 0.9945, 0.9945 - 0.7685, 0.7685 - 0.0573, 0.0573 - 0)
                    = (0.0055, 0.2260, 0.7112, 0.0573)
```

summing to `1.0000` (telescoping, section 4(e)) -- the point `x=0.3` sits
mostly in the `[0, 1]` bin (weight `0.71`), a little in `[-1, 0]`
(`0.23`), and almost nothing in the outer bins, matching the geometric
picture exactly.

## 6. Proposed API

Does not exist yet. `torch.float64` throughout the torch modules, matching
the existing package-wide convention already established by
`omnibias.tab.torch.{model,train,boosting}` (all define
`_DTYPE = torch.float64` for the exact-curvature optimizers' conditioning) --
this spec continues that convention rather than introducing a mixed-precision
inconsistency inside one package. `band_embed` / `integral_embed` in
`omnibias.core` are plain numpy (`float64`), consistent with every other
`omnibias.core` primitive; the jax twin uses `jax.numpy` with the caller
responsible for `JAX_ENABLE_X64` exactly as every other jax module in the
repo. There is no jax trainer (`fit_noise_aware`, `fit_boosted_heteroscedastic`)
because no jax trainer exists anywhere in `omnibias.tab` today -- matching
the existing convention, not a new gap.

### (a) Phase 1 -- `omnibias.tab._core.heteroscedastic`

```python
def gaussian_nll_grad_hess(
    u: FloatArray, v: FloatArray, y: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Closed-form (grad_u, grad_v, fisher) for the heteroscedastic Gaussian NLL.

    ``fisher`` has shape ``(..., 2, 2)``: the EXACT Fisher block
    ``diag(1/s**2, 2)`` (section 4(a)), not the indefinite observed Hessian.
    """
```

### (a') Phase 1 -- `omnibias.tab.torch.heteroscedastic`

```python
class HeteroscedasticHead(nn.Module):
    """Two SoftTreeEnsembles sharing n_features/n_trees/depth, predicting (f_hat, log_scale)."""
    def __init__(self, config: SoftTreeConfig, *, scale_width: int | None = None) -> None: ...
    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]: ...  # (f_hat, log_scale), both (..., n_outputs)

def fit_heteroscedastic(
    model: HeteroscedasticHead, X: np.ndarray, y: np.ndarray, *,
    steps: int = 60, optimizer: str = "trust_region", **opt_kwargs: Any,
) -> TrainResult:
    """Trains (f_hat, log_scale) jointly by the Gaussian NLL (section 4(a))."""

def fit_noise_aware(
    model: SoftTreeEnsemble, X: np.ndarray, y: np.ndarray, *,
    log_scale: np.ndarray, lam: float = 1.0,
    optimizer: str = "trust_region", steps: int = 60, **opt_kwargs: Any,
) -> TrainResult:
    """fit_second_order plus the closure-local penalty of section 4(b).

    lam=0.0 reproduces fit_second_order bit-identically (gate G0).
    log_scale is frozen input (section 4(c)); never trained here.
    """
```

### (b) Phase 2 -- `omnibias.core.band_embed`

```python
def band_embed(x: FloatArray, thresholds: FloatArray, *, beta: float) -> FloatArray:
    """(...,) -> (..., J+1). Row j is sigma(beta(x-t_j)) - sigma(beta(x-t_{j+1})).

    thresholds is 1-D, strictly increasing, length J (padded with +-inf
    internally). Rows sum to 1 exactly (telescoping, section 4(e))."""

def integral_embed(x: FloatArray, thresholds: FloatArray, *, beta: float) -> FloatArray:
    """Antiderivative twin. d/dx integral_embed(x, t, beta=b) == b * band_embed(x, t, beta=b)."""
```

### (c) Phase 2 -- `omnibias.tab.{torch,jax}.embed`

```python
class BandFeatureEmbedder(nn.Module):  # torch; jax twin is a pure function + pytree params
    """Per-feature band_embed/integral_embed, concatenated across n_features.

    role: "band" | "integral". init: "quantile" (thresholds from training data) | "uniform".
    learnable_beta / learnable_thresholds default True; beta stored via softplus (positive).
    """
    def __init__(self, n_features: int, n_bins: int = 16, *, beta_init: float = 1.0,
                 role: str = "band", init: str = "quantile",
                 learnable_beta: bool = True, learnable_thresholds: bool = True) -> None: ...
    def forward(self, X: Tensor) -> Tensor: ...  # (..., n_features * (n_bins+1))

def local_target_consistency_loss(
    embedder: BandFeatureEmbedder, X: Tensor, *, df_dx: Tensor,
) -> Tensor:
    """Closed-form pullback-metric loss (section 4(f)). df_dx: (n, n_features), the
    clean target's gradient -- known exactly for a synthetic generator, or a frozen
    surrogate estimate for real data (section 10 states the caveat)."""
```

### (c') Phase 2 -- `omnibias.tab.torch.boosting`

```python
def fit_boosted_heteroscedastic(
    X: np.ndarray, y: np.ndarray, config: SoftTreeConfig, *,
    log_scale: np.ndarray, n_stages: int = 30, learning_rate: float = 0.3,
    weighting: str = "gls",  # h_i / s_i**2, section 4(d)
    **kw: Any,
) -> tuple[SoftTreeEnsemble, BoostResult]:
    """fit_boosted with the weak-learner weight h -> h / s_hat**2. weighting="shrinkage"
    (h unchanged) reproduces fit_boosted bit-identically -- a second G0-style check."""
```

### (d) Phase 1 -- `omnibias.tab.bench` additions

```python
def saw_wave_2d(n: int, *, seed: int) -> Dataset:
    """Kartashev et al. subsection 4.2: x1~U[0,1], x2~U[0,10]; f=1 inside five
    wedges (2i,0)-(2i+1,1)-(2i+2,0) for i=0..4 in (x2,x1); noise std x2**6/62500.
    s(x) is returned via Dataset (a documented extra field), known exactly."""

def mlp_heteroscedastic_20d(n: int, *, seed: int) -> Dataset:
    """Kartashev et al. subsection 4.1 / Appendix F: x~N(0,I_20); f, g random MLPs
    (3-layer / 2-layer ReLU); y = f(x) + e^{g(x)} N(0,1). df/dx available via autograd
    on the frozen generator MLP -- used by gate G2, not by G1."""

def fit_predict_catboost_uncertainty(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, *, seed: int = 0, **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits CatBoostRegressor(loss_function="RMSEWithUncertainty") on train only;
    returns (mean_pred, log_scale_pred) on Xte per section 4(c)."""
```

## 7. Practical use cases

1. **Heteroscedastic regression with a known noise source** (sensor
   measurement, pricing, physical assay data): a practitioner who suspects
   noise varies across the input space gets a trainer that explicitly does
   not chase it, instead of an ordinary MSE fit that overfits the noisy rows
   as readily as the clean ones.
2. **Newton-boosted regression**, already `omnibias-tab`'s strongest existing
   mode -- `fit_boosted_heteroscedastic` is a three-line change to a trainer
   that already exists and is already benchmarked against LightGBM.
3. **A visual and quantitative diagnostic** any `omnibias-tab` user can run
   on their own dataset once they have a CatBoost uncertainty fit: the same
   qualitative read as the source paper's Figure 5, without leaving the
   package.
4. **Certificates that are honest about where they were earned.** Combining
   `certify_tab` with a noise-aware fit does not change the certificate's
   soundness, but it does mean the certified model is the one that was
   asked not to overfit the noisy regions -- relevant to a user who wants
   both a bound and a model that deserves it.
5. **A tabular benchmark that engages the actual current frontier.**
   `omnibias.tab.bench` compares only against LightGBM today; sections 6(d)
   and 9 add CatBoost, RealMLP, and TabM, so "beats known methods" becomes a
   checkable claim against the methods a 2025-generation tabular-DL paper
   considers state of the art, on the specific axis (uncertainty deciles)
   where that literature shows the gap is real.

## 8. Acceptance gates

All multi-seed gates use `>= 5` seeds, worst-seed, via
`benchmarks._gates.require_all_seeds`. Every gate below is falsifiable and
absolute (a named baseline, a predeclared threshold), per the `_gates.py`
doctrine.

- **G0 plumbing identity.** `fit_noise_aware(model, X, y, log_scale=s, lam=0.0)`
  produces bit-identical output to `fit_second_order` on the same model,
  data, and optimizer kwargs -- exact equality (`require_backend_parity`),
  not a tolerance. True by construction (section 4(b)); the test exists to
  catch an implementation bug, not to discover a fact.
- **G1b Hessian correctness.** For a small model and a fixed `theta_now`,
  finite-differencing the penalized closure's Hessian matches the analytic
  `H_task + lam * Sigma_noise` to float64 tolerance (`<= 1e-6` relative), for
  `lam in {0.1, 1.0, 10.0}`. Validates the mechanism, not the hypothesis.
- **G1 -- the falsifier.** On **both** `saw_wave_2d` and
  `mlp_heteroscedastic_20d` (section 6(d)), with `lam` selected on a
  validation split from the predeclared grid `{0.1, 0.3, 1.0, 3.0, 10.0}`
  (never on test), the three-question form of `benchmarks/_gates.py`:
  1. *Reference valid*: `Spearman(s_hat, s_true) >= 0.6` on test, else the run
     is `INVALID EXPERIMENT` and no downstream number is read.
  2. *Skill*: both the `lam=0` arm and the selected `lam>0` arm beat the
     constant mean-predictor (`skill_score > 0`) on the top decile of test
     rows by `s_true` (synthetic, so the *true* noise ranks the decile, not
     the estimate -- removing the estimator from the ranking step entirely).
  3. *Absolute*: the `lam>0` arm's top-decile RMSE is `<= 0.95x` the `lam=0`
     arm's top-decile RMSE (a predeclared `5%` improvement, `tau=0.05`).
  **If G1 fails on either dataset, Proposal A (section 4(b)) is recorded as
  falsified and Phase 2 items that depend on `fit_noise_aware` specifically
  (not `fit_boosted_heteroscedastic`, section 4(d), which is independent) are
  not built.**

### Measured results (G1, `--full`, 5 seeds, worst-seed-of-5)

Run 2026-08-27, `docs/benchmarks/tabular_uncertainty.json`
(`config.budget = {n: 4000, steps: 80, n_trees: 32, depth: 2}`,
`catboost_iterations: 500`). **G0 and G1b both passed** (`g0.passed = true`;
`g1b.worst_rel_error = 1.4e-8 <= 1e-6`) -- the plumbing and the Hessian
identity are correct. **G1 failed on both datasets**
(`g1_all_passed = false`):

| Gate | `saw_wave_2d` (worst seed) | `mlp_heteroscedastic_20d` (worst seed) |
| --- | --- | --- |
| 1. reference valid (Spearman >= 0.6) | **pass** -- 0.867 | **pass** -- 0.770 |
| 2. skill, `lam=0` arm (>= 0) | **fail** -- -0.274 | **fail** -- -0.661 |
| 2. skill, `lam>0` arm (>= 0) | **fail** -- -0.034 | **fail** -- -0.031 |
| 3. absolute (`ratio <= 0.95`) | **fail** -- 1.003 | **pass** -- 0.882 |
| `g1_passed` | **false** | **false** |

Two distinct failure shapes, both informative:

- **On `saw_wave_2d`, the penalty is nearly inert.** For 4 of 5 seeds the
  validation RMSE across the entire `lam` grid agrees to 8+ significant
  figures, so `lam` selection is effectively arbitrary and the selected-`lam`
  top-decile RMSE sits within a percent of the `lam=0` arm either way (ratio
  range `0.88`-`1.00`). The zero-gradient penalty is, by construction, only a
  curvature term; on this 2-D wedge target at this budget the exact-Newton
  trust-region step apparently does not need the extra damping to avoid
  overfitting the noisy tail, so there is little for `Sigma_noise` to fix.
- **On `mlp_heteroscedastic_20d`, damping produces a real, robust absolute
  win (12-23% top-decile RMSE reduction, all 5 seeds) but both arms still
  trail a constant mean predictor on the noisiest decile.** The `lam=0` arm's
  skill is strongly negative (`-0.22` to `-0.66`): the plain exact-Newton
  soft-tree overfits the top-noise decile badly enough that a trivial mean
  baseline beats it there. Damping closes most of that gap (`lam>0` skill
  `-0.03` to `+0.01`) without fully closing it. So the mechanism has a real
  effect in the intended direction on this dataset -- it is just not enough,
  at this budget, to cross the bar of beating a constant predictor, which G1
  correctly refuses to waive.

**Verdict: Proposal A (`fit_noise_aware`, section 4(b)) is falsified as a
robust, both-dataset improvement at the tested budget.** Per the predeclared
rule above, no further Phase 2 work is built specifically around promoting
`fit_noise_aware`. The mechanism itself (G0, G1b) is retained in the codebase
-- it is mathematically correct and may be revisited at a different budget or
model capacity (section 11) -- but it is not claimed as a win, and
`fit_boosted_heteroscedastic` (section 4(d)) and the band/integral embedding
(section 6(b)) proceed as the independent, unaffected Phase 2 tracks.
- **G2 local target consistency** (Phase 2; `mlp_heteroscedastic_20d` and any
  public set only -- section 10 explains why not `saw_wave_2d`).
  `local_target_consistency_loss`-trained `BandFeatureEmbedder` improves the
  top-decile RMSE of a downstream `SoftTreeEnsemble` head over the same head
  on raw features by a predeclared `5%`, matching the source paper's
  qualitative finding (their Figure 6) as a quantitative one.
- **G3 boosted heteroscedastic win** (Phase 2). On both synthetic sets,
  `fit_boosted_heteroscedastic(weighting="gls")` beats
  `fit_boosted_heteroscedastic(weighting="shrinkage")` (= plain `fit_boosted`,
  a second G0-style bit-identity check) on top-decile RMSE by `5%`.
- **G4 public-suite honesty report** (Phase 2). On `>= 6` public regression
  sets (section 9), report the **full** top-decile-RMSE win/loss table for
  omnibias-tab (best Phase 1+2 arm) against tuned LightGBM, CatBoost,
  RealMLP, and TabM. No aggregate-only reporting -- matching 05-02 G3's
  explicit rule. The expected outcome, stated before running it, is a mixed
  table; the deliverable is the table, not a win.
- **G5 no regression on the aggregate metric** (Phase 2). Across the same
  suite, the best omnibias-tab arm is not worse than plain `fit_second_order`
  / `fit_boosted` on **overall** (not decile-conditioned) RMSE by more than
  the baseline's own across-seed noise -- so any uncertainty-decile win is
  not bought by getting worse everywhere else.

### Measured results (G2-G5, `--full`, 5 seeds, worst-seed-of-5)

Run 2026-08-27, `docs/benchmarks/tabular_uncertainty.json`
(`config.budget_g2g3 = {n: 2000, n_trees: 16, depth: 2, steps: 40, n_stages: 10,
inner_steps: 20, catboost_iterations: 200, n_bins: 12, embed_epochs: 200,
surrogate_epochs: 200}`, `config.budget_g4g5 = {max_rows: 2000, n_trees: 16,
depth: 2, n_stages: 10, inner_steps: 20, catboost_iterations: 200,
realmlp_epochs: 30, tabm_epochs: 30}`). Total wall time for the entire
G0-G5 run, 8 workers: 4146s (69 min).

**G2 failed on both datasets** (`g2_all_passed = false`):

| Dataset | `df/dx` | worst-seed ratio (embed/raw) | need `<= 0.95` |
| --- | --- | --- | --- |
| `mlp_heteroscedastic_20d` | exact | 1.052 | fail |
| `kin8nm` | surrogate (fitted MLP) | 1.867 | fail |

`local_target_consistency_loss`-trained `BandFeatureEmbedder` does not improve
-- and on the surrogate-gradient public case, substantially harms --
downstream top-decile RMSE at this budget. On `mlp_heteroscedastic_20d` (exact
`df/dx`), 4 of 5 seeds show the embedding slightly worse than raw features and
one slightly better (per-seed ratios `0.949`-`1.052`) -- a near-wash, not a
useful signal beyond what the raw 20 features already carry for this
MLP-generated target. On `kin8nm` (surrogate `df/dx` from a small fitted MLP,
section 10's honestly-harder case), the embedding is decisively worse on
*every* seed (ratios `1.57`-`1.87`) -- the surrogate gradient's own estimation
error appears to actively mislead the pullback-metric objective rather than
merely fail to help, exactly the risk section 11 flagged before running this.

**G3 failed on both datasets** (`g3_all_passed = false`):

| Dataset | worst-seed ratio (gls/shrinkage) | need `<= 0.95` |
| --- | --- | --- |
| `saw_wave_2d` | 1.014 | fail |
| `mlp_heteroscedastic_20d` | 1.014 | fail |

All 10 per-seed ratios (5 seeds x 2 datasets) fall in a narrow band
(`0.956`-`1.014`) indistinguishable from noise around `1.0`. Unlike G2's
`kin8nm` result, this is a **wash, not a directional harm**:
`weighting="gls"` neither reliably helps nor hurts the boosted top decile
relative to `weighting="shrinkage"` at `n_stages=10`, `inner_steps=20`.

**G4 passed** (`g4.passed = true`, 9/9 public sets completed, `>= 6`
required) -- the full win/loss table (top-decile RMSE, 5-seed majority
winner) for `fit_boosted_heteroscedastic(weighting="gls")`, the predeclared
"best Phase 1+2 arm":

| Dataset | vs LightGBM | vs CatBoost | vs RealMLP | vs TabM |
| --- | --- | --- | --- | --- |
| `kin8nm` | **W** 4/5 | L 0/5 | L 0/5 | L 0/5 |
| `wine_quality` | L 1/5 | L 0/5 | **W** 5/5 | **W** 3/5 |
| `energy_efficiency` | L 0/5 | L 0/5 | **W** 5/5 | L 0/5 |
| `auto_mpg` | L 2/5 | L 2/5 | **W** 3/5 | L 2/5 |
| `forest_fires` | **W** 5/5 | **W** 5/5 | **W** 4/5 | L 2/5 |
| `house_sales` | L 0/5 | L 0/5 | L 0/5 | L 0/5 |
| `bank32nh` | L 0/5 | L 0/5 | L 0/5 | L 0/5 |
| `bike_sharing` | L 0/5 | L 0/5 | L 0/5 | L 0/5 |
| `miami_housing` | L 0/5 | L 0/5 | L 0/5 | L 0/5 |

omnibias-tab is the majority-seed winner on 8 of 36 dataset x baseline
comparisons (22%): it sweeps `forest_fires` (3/4 baselines) and specifically
beats RealMLP on 4 of 9 datasets, but loses **every** comparison outright on
`house_sales`, `bank32nh`, `bike_sharing`, and `miami_housing`. This is the
mixed table stated as the expected outcome before running it (section 8) --
the deliverable is the table, not a win, and G4 gates on completeness
(9/9 >= 6), not on any cell's sign.

**G5 failed** (`g5.passed = false`) -- regression beyond the baseline's own
across-seed noise on 3 of 9 public sets:

| Dataset | vs `fit_second_order` | vs `fit_boosted` |
| --- | --- | --- |
| `kin8nm` | pass | pass |
| `wine_quality` | pass | pass |
| `energy_efficiency` | pass | **fail** |
| `auto_mpg` | pass | **fail** |
| `forest_fires` | pass | pass |
| `house_sales` | pass | pass |
| `bank32nh` | pass | pass |
| `bike_sharing` | **fail** | **fail** |
| `miami_housing` | pass | pass |

Two distinct causes: (1) GLS reweighting measurably regresses *overall* RMSE
relative to plain `fit_boosted` on `energy_efficiency` and `auto_mpg` -- the
same direction as G3's near-miss, now crossing the no-regression bar on the
un-conditioned metric specifically; (2) on `bike_sharing`, boosting itself
(heteroscedastic or not) is worse than the single-tree `fit_second_order` fit
at this reduced `n_stages=10`/`inner_steps=20` budget -- a budget effect on
this one dataset, not a GLS-specific defect, since it fails against
`fit_second_order` even before GLS reweighting enters.

**Overall Phase 2 verdict: none of the three falsifiable hypothesis gates
(G1, G2, G3) survived contact with data at the tested budgets**; only the
reporting-completeness gate (G4) and the plumbing gates (G0, G1b) pass.
`fit_boosted_heteroscedastic`, `BandFeatureEmbedder`, and
`local_target_consistency_loss` are retained in the codebase as
correctly-implemented, tested mechanisms -- backed by the same G0-style
bit-identity and forward-parity tests as everything else in `omnibias-tab` --
but none is claimed as a benchmarked win. The spec's status stays `gated`,
not `shipped`: implementation is complete (section 12), every falsifier ran
honestly, and the measured answer on all three is "not at this budget, not
yet."

## 9. Benchmark plan

- Extend `packages/omnibias-tab/src/omnibias/tab/bench.py`: a
  regression-capable `train_val_test_split` (drop the stratify call for
  `task == "regression"` rather than raising), `saw_wave_2d`,
  `mlp_heteroscedastic_20d`, `fit_predict_catboost_uncertainty`, and (Phase 2)
  a public regression suite plus `fit_predict_realmlp` / `fit_predict_tabm`
  via `pytabkit`.
- `benchmarks/tabular_uncertainty.py`, following the exact convention of
  `benchmarks/tabular_arrangement.py`: a single `--full` flag, smoke to
  `docs/benchmarks/tabular_uncertainty_smoke.json`, full to
  `docs/benchmarks/tabular_uncertainty.json` plus a copy under
  `$OMNIBIAS_SCRATCH/tabular_uncertainty/`, `gates_block(...)` +
  a hand-authored `honesty` block, `raise SystemExit(1)` when
  `not gates["all_passed"]`. Phase 1 gates (G0/G1b/G1) run in the smoke tier
  on CPU; the full sweep (`lam`-selection across both synthetic sets, G2-G5
  across the public suite, all `>= 5` seeds) is heavy enough to run as a
  CPU-cluster job (`--workers N` process-pool parallel, no GPU required --
  every trainer here is CPU torch), with the per-run JSON collected back into
  the repo.
- The Figure-5-style panel and the Figure-1-style smoothed decile curve
  (`scipy.ndimage.gaussian_filter1d`, matching the source paper's Appendix B
  exactly) are rendered by an addition to `docs/img/generate_figures.py` from
  a cached prediction grid under `$OMNIBIAS_SCRATCH` (too large to commit);
  the PNG is committed to `docs/img/`, and the panel carries no `passed` key
  -- it is a diagnostic, never a gate (section 10).
- A smoke-tier CI step added to the `tab` job in `.github/workflows/ci.yml`,
  gated on the existing `gbm` extra plus a new optional extra for CatBoost
  (heavier baselines -- RealMLP/TabM via `pytabkit` -- stay out of CI,
  matching the existing convention that only LightGBM runs in CI while richer
  baselines are `--full`-only).

## 10. Honesty and scope

- **Two mechanisms, not one, and they must not be conflated.** Section 4(b)'s
  zero-gradient curvature penalty is the actually new piece: it changes
  nothing about the task loss's gradient or minimizer, only the trust-region
  geometry the optimizer uses to get there, and is exact by construction, not
  a first-order approximation. Section 4(d)'s GLS reweighting of the boosting
  weak-learner weight is a well-established heteroscedastic-regression
  technique (in the lineage of NGBoost); it is included because the hook
  already exists in `_fit_weak_learner`, not presented as a new idea.
- **`s_hat` is `GuaranteeKind.MODEL_BASED`** (`omnibias.core.uncertainty`,
  theory 04-02), never a sound enclosure. It must never be passed to
  `certify_tab` / `certify_composed` as if it were an interval, and
  `conformal_sealed` stays `False` throughout. Training on `s_hat` does not
  change what `certify_tab` can prove about the resulting model; it changes
  only which model gets fitted.
- **Anti-circularity is load-bearing, not decoration.** `s_hat` must come
  from a frozen, train-only-fit, architecturally distinct estimator
  (CatBoost) before any `omnibias.tab` training happens, and G1's
  reference-validity sub-gate ranks the top decile by the **true** `s`, not
  the estimate, on synthetic data specifically so the falsifier does not rest
  on the estimator being good. On real data (Phase 2 G4/G5) there is no
  ground truth, so those gates are honestly softer than G1.
- **Regression only**, matching the source paper's own stated limitation
  (its section 9): "our investigation was primarily focused on regression
  tasks." `omnibias.tab`'s classification Hessian (`p(1-p)`, bounded) is
  structurally different from the free-scale Gaussian case here, and
  extending this mechanism to classification is out of scope for this spec,
  not silently assumed to transfer.
- **`beta -> inf` in `band_embed` / `BandFeatureEmbedder`** is temperature
  collapse (feasibility sense) -- the same sense as `SoftTreeConfig`'s
  existing gate anneal. The window's finite, fixed gap
  (`t_{j+1} - t_j`) is the **opposite** of the founding `delta -> 0` bias
  collapse, exactly as 04-02 states for the same `band`/`integral` role; this
  spec is a second, independent consumer of that role and repeats the
  disclaimer rather than assuming it carries over.
- **`local_target_consistency_loss` needs a usable `df/dx`.**
  `saw_wave_2d`'s clean target is piecewise-constant, so its gradient is zero
  almost everywhere by construction and the loss is not meaningful there --
  gate G2 is evaluated only on `mlp_heteroscedastic_20d` (where the generator
  MLP's gradient is exact and free) and on public data (where `df/dx` is a
  frozen surrogate estimate, whose error propagates into the loss and is
  reported, not hidden).
- **The Figure-5-style panel is a diagnostic, never evidence.** It carries no
  `passed` key and never enters `gates["all_passed"]`. A panel that looks
  persuasive while G1 fails is a failed experiment and is reported as one.
- No `theorem_prover_verified` or `mathlib_verified` claim anywhere in this
  spec; no certificate tier is added or changed.

## 11. Open questions and risks

- **Falsifier -- realized.** G1 failed on both datasets (section 8, measured
  results). The anticipated failure mode partly matches: on `saw_wave_2d` the
  penalty was nearly inert (consistent with the package's existing implicit
  regularization already absorbing the benefit at this budget), while on
  `mlp_heteroscedastic_20d` the penalty had a real, robust absolute effect
  but both arms trailed a constant predictor on the noisiest decile, so the
  skill bar -- not the absolute one -- is what sank the gate there. This is
  recorded as a genuine negative result for Proposal A specifically; `lam`
  was not widened or re-tuned after seeing it.
- **`lam` selection discipline.** The candidate grid
  `{0.1, 0.3, 1.0, 3.0, 10.0}` is predeclared in section 8 and selected on
  validation, never on test; widening the grid after seeing a test-set number
  would invalidate G1 and must not happen.
- **`Sigma_noise` cost.** `Sigma_noise v = (2/n) sum_i s_i^2 g_i (g_i^T v)` is
  a Gauss-Newton-shaped matrix-vector product and should be computable
  without materializing the `(n, P)` per-sample-gradient matrix, analogous to
  the existing `CubicGaussNewton` linearization pattern
  (`omnibias.torch.optim._linearize_gn`) -- but this needs a real
  implementation, not an assumption that it will be cheap. If it is not, `lam
  > 0` may be too slow to sweep at the scale G1 needs, and that cost must be
  measured and reported, not absorbed silently into a longer smoke timeout.
- **CatBoost dependency.** `s_hat`'s quality on real data is bounded by
  CatBoost's own fit; the source paper's Appendix H shows the conclusion is
  estimator-robust (MLP-based estimates agree), but this spec uses one
  estimator family only and states that as a scope limit rather than
  re-deriving Appendix H.
- **The pullback-metric loss (section 4(f)) -- risk realized.** G2 failed on
  both cases (section 8): a near-wash on the exact-gradient synthetic set, and
  a decisive loss on the surrogate-gradient public set, where a noisy `df/dx`
  measurably misleads the objective rather than merely failing to help. This
  is the failure this bullet anticipated, now measured rather than
  hypothesized -- and it is a bounded failure of section 4(f) specifically,
  not of section 4(b)/(d), which do not depend on it and were falsified
  independently (G1, G3).
- **G3's boosted-GLS reweighting is a wash, not a harm -- except where G5 says
  otherwise.** G3's ten per-seed ratios (section 8) cluster tightly around
  `1.0`: GLS reweighting is statistically indistinguishable from plain
  shrinkage on the *decile-conditioned* metric at `n_stages=10`,
  `inner_steps=20`. G5 (section 8) shows the same reweighting *does*
  measurably regress **overall** RMSE on 2 of 9 public sets
  (`energy_efficiency`, `auto_mpg`) -- "no measured win" and "no measured
  harm" are not the same claim across every metric, and both are reported
  rather than one hidden behind the other.
- **`bike_sharing` regresses under boosting itself, independent of GLS.** Both
  `fit_boosted` and `fit_boosted_heteroscedastic(gls)` post a *worse* overall
  RMSE than plain `fit_second_order` on `bike_sharing` at the G4/G5 budget
  (`n_stages=10`, `inner_steps=20`, `max_rows=2000`) -- the only dataset where
  G5 fails against `fit_second_order` specifically. This reads as an
  under-budgeted-boosting effect on this one dataset, not a GLS-specific
  defect, and is reported rather than smoothed over by a larger, unmeasured
  budget.

## 12. Implementation checklist

Phase 1 (falsifier):

- [x] `omnibias/tab/_core/heteroscedastic.py` (`gaussian_nll_grad_hess`)
- [x] `omnibias/tab/torch/heteroscedastic.py` (`HeteroscedasticHead`,
      `fit_heteroscedastic`, `fit_noise_aware`)
- [x] `bench.py`: regression-capable `train_val_test_split`, `saw_wave_2d`,
      `mlp_heteroscedastic_20d`, `fit_predict_catboost_uncertainty`
- [x] G0 exact-equality test, G1b finite-difference Hessian test, Fisher
      positive-definiteness test, NLL gradient-vs-autograd test
- [x] CPU smoke run validating the code path end to end
- [x] `benchmarks/tabular_uncertainty.py` (Phase 1 gates: G0, G1b, G1)
- [x] The Figure-5-style panel and Figure-1-style decile curve, cached
      prediction grid under `$OMNIBIAS_SCRATCH`, PNG in `docs/img/`
- [x] G1 run as a CPU-cluster job; result recorded in this file (section 1,
      section 8) -- **failed on both datasets; Proposal A falsified**
- [x] Index row in `theory/README.md`

Phase 2 (the parts independent of the falsified `fit_noise_aware`, per the
predeclared rule in section 8):

- [x] `omnibias/core/band_embed.py` (`band_embed`, `integral_embed`) reusing
      the `ftc_block` / `OperatorBlock` formula, with the telescoping-sum and
      FTC-derivative identities as unit tests
- [x] `omnibias/tab/torch/embed.py` + `omnibias/tab/jax/embed.py`
      (`BandFeatureEmbedder`, bit-identical forward parity test) and
      `local_target_consistency_loss` (torch only)
- [x] `fit_boosted_heteroscedastic` in `boosting.py` (`weighting="gls"` /
      `"shrinkage"`; the latter bit-identical to `fit_boosted`)
- [x] Public regression suite (`>= 6` sets) plus tuned CatBoost / RealMLP /
      TabM baselines (`pytabkit`) in `bench.py`
- [x] `benchmarks/tabular_uncertainty.py` extended with G2-G5 (G4/G5's
      "best Phase 1+2 arm" no longer includes `fit_noise_aware`)
- [x] Optional `pyproject.toml` extras for CatBoost / pytabkit, smoke-only in
      CI (heavier baselines stay `--full`)
- [x] Docs page + mkdocs nav entry carrying the two-mechanisms and
      anti-circularity disclaimers verbatim, and stating the G1 result
      plainly for `fit_noise_aware`
- [x] CHANGELOG entry
- [x] `--full` G2-G5 sweep run on a CPU-cluster job; results recorded in this
      file (section 8) -- **G2 and G3 also failed** (both falsifiable
      Phase 2 hypotheses); **G4 passed** (reporting completeness, 9/9
      datasets); **G5 failed** (3/9 datasets regress beyond seed noise)
- [x] Spec status stays `gated` (not promoted to `shipped`): all three
      falsifiable hypothesis gates (G1, G2, G3) failed at the tested budgets;
      index row in `theory/README.md` updated to match
