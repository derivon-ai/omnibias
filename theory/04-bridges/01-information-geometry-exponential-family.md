# 04-01 The information geometry of collapse

## 1. Thesis and status

A pack is a parametric family, so it has a Fisher information metric — and
because the tower is exact, that metric is **closed form**, which turns natural
gradient, model distinguishability, and the geometry of the collapse limit into
computable objects rather than estimated ones.

- **Status**: gated
- **Depends on**: 01-01, 01-10
- **Blocks**: 03-01, 04-02

Wave-0 falsifier G2 is recorded in
`docs/benchmarks/information_geometry.json` (`all_passed: true`). The
section-6 product API and gates G1 / G3 / G4 / G5 remain unearned.

## 2. Where it lands

`packages/omnibias-curvature/src/omnibias/curvature/information/` — the
curvature package already owns Fisher and natural gradient — with the manifold
side in `omnibias.geometry` where a metric is a first-class object.

## 3. Prior art in omnibias

- `packages/omnibias-curvature/` — closed-form Hessian, Fisher and KFAC, with
  `NaturalGradient` in `omnibias.torch.optim`. The existing Fisher symbols
  (`omnibias.curvature.fisher_information_metric`, `glm_fisher`, and
  `omnibias.{torch,jax}.information.fisher_information`) are the **scalar
  exponential-family Fisher** `A''(theta)` — a different object from the
  pack-parameter metric this spec measures.
- `omnibias-geometry` — metric, Christoffel symbols, covariant derivative,
  curvature, geodesics, and the pullback metric of a learned chart
  `g = J^T h J`. Everything needed to treat a Fisher metric as a Riemannian
  metric is already there.
- `omnibias-measure` — the `Measure` abstraction with pushforward and importance
  reweighting.
- `omnibias.score` — score functions, the Itô generator, Fokker-Planck; the
  score is the object whose covariance is the Fisher metric.
- `benchmarks/information_geometry.py` — Wave-0 G2 falsifier (gated):
  cancellation-free two-bias logistic pack density, closed-form integrand
  quadrature, Monte Carlo score^2 baseline.

**Confirmed gap.** Fisher information exists as a *curvature object for
optimization* and as a *scalar GLM Fisher*. It has never been treated as a
**metric on the pack parameter manifold** beyond the G2 degeneracy measurement,
and the section-6 API (`fisher_metric`, geodesics, distinguishability) is not
implemented.

## 4. Mathematics

### The Fisher metric of a pack family

Take a density parameterized by pack parameters `theta = (mu, alpha, c, ...)`:

```
p_theta(x) = sum_g c_g alpha_g sigma'( alpha_g (x - mu_g) ),   c_g >= 0, sum c_g = 1
```

The Fisher information matrix is

```
G_{ij}(theta) = E_{x ~ p_theta} [ ( d log p / d theta_i )( d log p / d theta_j ) ]
```

Both `dp/dtheta_i` and `p` are closed form from the tower — differentiating with
respect to `mu` raises the order, differentiating with respect to `alpha` uses
the exact scaling law. So the **integrand is closed form**, and `G` requires only
the expectation, which for these families is a small number of moment integrals
(spec 03-06) or, where those are available in closed form, no quadrature at all.

That is the concrete claim: for activation mixtures, the Fisher metric is
computable without sampling.

### What the metric says about collapse

Here is the part that is genuinely new. Consider a two-bias pack at spread
`delta`, so the biases are `b +- delta/2`. As `delta -> 0` the family collapses
to `sigma'`. What happens to the Fisher metric in the `delta` direction?

The two-bias combination is

```
f_delta(z) = ( sigma(z + delta/2) - sigma(z - delta/2) ) / delta  ->  sigma'(z)
```

Expanding, `f_delta = sigma' + (delta^2 / 24) sigma''' + O(delta^4)`. So the
derivative with respect to `delta` is

```
d f_delta / d delta = ( delta / 12 ) sigma''' + O(delta^3)
```

which **vanishes linearly as `delta -> 0`**. Therefore the Fisher metric
component `G_{delta delta}` vanishes **quadratically**:

```
G_{delta delta} ~ ( delta^2 / 144 ) E[ ( sigma''' / f )^2 ]  =  O(delta^2)
```

The collapse limit is a **degenerate point of the Fisher metric**: the parameter
`delta` becomes statistically unidentifiable at exactly the rate `delta^2`.

That is a precise and useful statement. It says:

1. Near collapse, `delta` is *not* estimable from data — the information about it
   goes to zero quadratically. Attempts to fit `delta` near zero will be
   ill-conditioned, and now the conditioning is quantified.
2. The natural-gradient step in the `delta` direction is correspondingly
   amplified (the inverse metric blows up as `delta^-2`), which is exactly the
   pathology natural gradient is supposed to fix and here does not, because the
   degeneracy is real rather than a parameterization artifact.
3. The right coordinate near collapse is **not `delta`** but the collapsed order
   itself. The tower's discrete order parameter is the well-conditioned
   coordinate; `delta` is a chart that degenerates at the origin.

Point 3 is the design consequence, and it retroactively justifies the whole
architecture of the library: **omnibias parameterizes by order, not by spread,
and the Fisher metric says that is the right choice.** That is worth writing
down properly, and it is the main content of this spec.

### Geodesics and interpolation

With a metric, interpolation between two models should follow a geodesic, not a
straight line in parameters. `omnibias-geometry`'s geodesic solver applies
directly once `G` is supplied. Practical uses: model averaging, curriculum paths
between architectures, and continuation methods.

### Distinguishability

The Fisher metric induces a distance whose local form measures
distinguishability: two parameter settings within `1/sqrt(N)` in Fisher distance
are indistinguishable from `N` samples. So the metric answers "how many samples
do I need to tell these two packs apart", exactly, for this family.

## 5. Worked example

**The degeneracy, computed.**

Take a single logistic pack, `sigma(z) = 1/(1 + e^{-z})`, with the two-bias
finite-difference form at spread `delta`, all other parameters fixed.

Expand:

```
( sigma(z + h) - sigma(z - h) ) / (2h) = sigma'(z) + (h^2/6) sigma'''(z) + O(h^4)
```

with `h = delta/2`, so the correction is `(delta^2 / 24) sigma'''(z)`, matching
the general statement above.

At `z = 0` for the logistic: `sigma(0) = 0.5`, `sigma'(0) = 0.25`,
`sigma''(0) = 0`, `sigma'''(0) = -0.125`.

So near `z = 0`:

```
f_delta(0) = 0.25 - 0.125 delta^2 / 24 = 0.25 - 0.0052083 delta^2
d f_delta(0) / d delta = -0.0104167 delta
```

At `delta = 0.1`: the derivative is `-1.0417e-3`, and the value is
`0.25 - 5.2083e-5 = 0.2499479`. A 10 percent change in `delta` (from `0.1` to
`0.11`) changes the function by

```
-0.0052083 * (0.0121 - 0.0100) = -1.094e-5
```

four orders of magnitude below the function value of `0.25`.

At `delta = 0.01`: the derivative is `-1.0417e-4`, and the same 10 percent
change moves the function by `-1.09e-7` — a hundred times less, for the same
relative change in the parameter.

**The information ratio.** The Fisher component scales as the squared derivative,
so going from `delta = 0.1` to `delta = 0.01` reduces `G_{delta delta}` by a
factor of `100`. To estimate `delta` to the same relative precision then requires
`100x` more data. That is the degeneracy, quantified, and it is a hard
statistical fact about the collapse limit rather than an implementation
weakness.

**The exact leading coefficient.** Substituting `t = sigma(x)` and taking
`delta -> 0` with `f -> sigma'` gives

```
E[(sigma''' / f)^2] -> int_0^1 (1 - 6t + 6t^2)^2 dt = 1/5
```

so with the prefactor `1/144` from `(delta / 12)^2`:

```
G_{delta delta} = delta^2 / 720 + O(delta^4)
```

Measured at `delta = 1e-4`: `G / delta^2 = 1.388888888227e-03` versus
`1/720 = 1.388888888889e-03` (relative deviation `4.8e-10`). Gate G2 of this
spec records the exponent; the prefactor is a sharpening of the same claim.

**The contrast.** The order parameter `n` is discrete and changing it changes the
function by an `O(1)` amount (from `sigma'` to `sigma''` is a completely
different shape). So the order coordinate carries `O(1)` information where the
spread coordinate carries `O(delta^2)`. **Parameterizing by order is
statistically well-posed; parameterizing by spread near collapse is not.**

## 6. Proposed API

Does not exist yet.

```python
# omnibias/curvature/information/_core.py
def fisher_metric(family: PackFamily, theta, *, closed_form: bool = True) -> FloatArray:
    """Closed form where the family's moments are; falls back to quadrature and
    records which path ran."""

def fisher_distance(family, theta_a, theta_b, *, steps: int = 64) -> float:
    """Geodesic distance under the Fisher metric, via omnibias-geometry."""

def distinguishability_samples(family, theta_a, theta_b, *, power: float = 0.8) -> int:
    """Samples needed to distinguish, from the Fisher distance."""

def collapse_degeneracy(family, theta, *, direction: str = "spread") -> DegeneracyReport:
    """Reports the metric eigenvalue along the collapse direction and its
    scaling exponent. Warns when the coordinate is near-degenerate."""

@dataclass
class DegeneracyReport:
    eigenvalue: float
    exponent: float           # expected 2 for the spread direction
    condition_number: float
    recommendation: str       # e.g. "reparameterize by order"
```

## 7. Practical use cases

1. **Justifying the architecture.** The degeneracy result is the statistical
   argument for order-based parameterization, and it belongs in the book (spec
   06-04).
2. **Natural gradient that knows about degeneracy**, damping the directions the
   metric says are uninformative rather than amplifying them.
3. **Identifiability analysis in inverse problems** (spec 05-01): which
   parameters can be recovered from the available data, quantitatively.
4. **Experiment design.** The metric says where to sample to maximize
   information about a parameter of interest.
5. **Geodesic model interpolation** for continuation and averaging.

## 8. Acceptance gates

Baselines: Monte Carlo Fisher estimation, and the empirical Fisher from
gradients.

- **G1 closed-form correctness.** The closed-form metric matches a
  high-precision Monte Carlo estimate to within the estimate's own standard
  error, on a randomized suite, and is `1000x` faster. **Unearned** — the
  Monte Carlo agreement arm of the G2 artifact covers the two-bias family
  only; the randomized mixture suite is D8.
- **G2 degeneracy exponent.** The measured scaling of `G_{delta delta}` with
  `delta` has exponent `2.00 +- 0.02` over at least three decades of `delta`.
  **This is the spec's central claim and must be measured, not derived only.**
  **Earned** — see `docs/benchmarks/information_geometry.json`: three overlapping
  3-decade windows fit to `1.999999996`, `1.99996`, `1.99636`; prefactor
  `1/720` within `1e-6` relative at `delta = 1e-4`; Monte Carlo agreement
  within 3 sigma on every seed.
- **G3 metric properties.** `G` is symmetric positive semi-definite everywhere
  tested, and positive definite away from the degeneracy, to numerical
  tolerance. **Unearned.**
- **G4 distinguishability calibration.** The predicted sample count for
  distinguishing two parameter settings matches an empirical hypothesis test's
  requirement within a factor of `2`. **Unearned.**
- **G5 natural-gradient improvement.** Damping by the degeneracy report improves
  convergence on a problem with a near-collapse parameter, versus undamped
  natural gradient, over five seeds. **Unearned.**

## 9. Benchmark plan

- `benchmarks/information_geometry.py`: closed-form versus Monte Carlo accuracy
  and speed, the degeneracy-exponent measurement across decades,
  distinguishability calibration, natural-gradient comparison.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/infogeom/`.

## 10. Honesty and scope

- The `delta -> 0` limit here **is** the founding bias collapse, and this spec is
  a statement about its statistical geometry. No temperature collapse appears.
- The degeneracy result is derived for the **two-bias finite-difference family**
  and generalizes to `K` biases with exponent depending on `K`; state the case
  analysed rather than implying full generality until the `K`-bias version is
  computed and tested.
- Closed-form Fisher requires closed-form moments of the family. Where they are
  unavailable the implementation falls back to quadrature and must record it.
- Information geometry (Amari, Rao) is classical. The contribution is the
  closed-form metric for pack families and the specific degeneracy result at the
  collapse limit.
- No certificate tier. Interval enclosures of the metric are possible and would
  make identifiability statements sound; that is a natural extension.

## 11. Open questions and risks

- **The `K`-bias exponent.** For `K >= 3` a central pack collapses toward
  `sigma^(K-1)`, which **changes sign**, so the pack is not a probability
  density and the Fisher metric does not apply. Recorded as *inapplicable*
  (not unmeasured) in the G2 artifact's `honesty.k_ge_3_fisher` field. The
  honest generalization needs a different object — an `L^2` Gram metric or a
  normalized mixture family — and belongs to the full D8 implementation. Do
  not guess an exponent.
- **Positive definiteness in practice.** Near-degenerate metrics are numerically
  singular; the implementation needs a principled pseudo-inverse rather than a
  tolerance chosen to make tests pass.
- **Non-normalized families.** The Fisher metric requires a probability density.
  For a general pack used as a basis function rather than a density, the right
  object is a different metric (for example the `L^2` Gram matrix), and
  conflating them would be an error. The `K >= 3` finding above is an instance
  of this trap.
- **Falsifier.** If the measured exponent is not `2` (G2 fails), the derivation
  is wrong and the architectural argument built on it must be withdrawn.
  **G2 passed**; the architectural argument stands for the two-bias logistic
  family.

## 12. Implementation checklist

- [ ] `packages/omnibias-curvature/src/omnibias/curvature/information/_core.py`
- [ ] Reuse `omnibias-geometry`'s metric and geodesic machinery
- [ ] Closed-form-versus-Monte-Carlo validation (randomized mixture suite; G1)
- [x] Degeneracy-exponent measurement across at least three decades (G2)
- [x] `K`-bias case classified as inapplicable (not a density); deferred to D8
- [ ] Principled pseudo-inverse with a documented threshold
- [ ] Explicit guard against applying the Fisher metric to non-densities
- [x] `benchmarks/information_geometry.py` plus smoke JSON
- [ ] Docs page and nav entry
- [x] Index row in `theory/README.md`
