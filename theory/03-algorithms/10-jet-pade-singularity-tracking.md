# 03-10 Jet-Padé singularity tracking

## 1. Thesis and status

A high-order jet is a truncated Taylor series, and the poles of its Padé
approximant locate the nearest complex singularity — so a field with an exact
tower can **watch a singularity approach** in real time, with a certified
remainder rather than a heuristic indicator.

- **Status**: designed
- **Depends on**: 01-01
- **Blocks**: 03-13, 07-02, 07-03, 07-06

## 2. Where it lands

`packages/omnibias-difference/src/omnibias/difference/singularity.py` — the
difference package already owns Padé and certified remainders — with a field
front end in `omnibias.fields.singularity`.

## 3. Prior art in omnibias

- `packages/omnibias-difference/` — the founding `delta -> 0` register:
  certified finite-difference to derivative extraction, umbral / Sheffer
  calculus, asymptotic-coefficient reading (Stirling, Bernoulli, Euler), and
  `pade_certified_remainder`.
- `omnibias.{torch,jax}.jet` and `jet_mv` — exact towers to arbitrary order,
  including `mlp_jet` through depth.
- `omnibias.dynamics` — validated dynamics: QR-Lohner variational and monodromy
  flow, Poincaré-section enclosures, certified Lyapunov bounds, periodic-orbit
  proofs by the radii polynomial.
- `omnibias.core.verified.sequence_space` — geometric-decay tail bounds, which
  are exactly the coefficient-decay statements a singularity analysis needs.
- `benchmarks/reproduce_deepmind_ccf.py` — the unstable-singularity campaign,
  the obvious consumer.

**Confirmed gap.** Padé machinery exists for *asymptotic coefficient reading*.
Nothing applies it to jets of a field to locate singularities, and nothing
tracks a singularity's motion during a solve.

## 4. Mathematics

### Coefficients encode the nearest singularity

For an analytic `f` with Taylor coefficients `a_k` about `x_0`, the radius of
convergence is

```
R = 1 / limsup |a_k|^{1/k}
```

and `R` is the distance to the nearest complex singularity. So the coefficient
growth *is* the singularity distance. Two standard estimators:

**Domb-Sykes.** For a singularity of the form `(1 - x/x_s)^{-p}`,

```
a_k / a_{k-1} = (1/x_s) ( 1 + (p - 1)/k + O(1/k^2) )
```

so plotting `a_k / a_{k-1}` against `1/k` gives `1/x_s` as the intercept and
`(p-1)/x_s` as the slope. **Both the location and the exponent** come out of a
linear fit on a handful of coefficients.

**Padé.** The `[L/M]` Padé approximant's denominator roots approximate the
singularities directly, and for a function with a single dominant pole the
convergence is rapid. Padé also handles branch points reasonably (as a cluster
of poles approximating a cut), which Domb-Sykes does not.

Use both: they fail differently, and agreement between them is evidence.

### Why omnibias is well placed

Both methods need **many accurate Taylor coefficients**, and that is exactly what
is expensive with autodiff and cheap with the tower. `mlp_jet` returns a jet of
order `N` through a deep composition in one pass, so `a_0 .. a_N` come out
together and exactly.

Concretely: getting 20 Taylor coefficients of a depth-4 network by repeated
autodiff is prohibitive; getting them from `mlp_jet` is one pass with `O(N^2)`
Cauchy-product work.

### Certified remainder

`pade_certified_remainder` bounds the error of the Padé approximant on a stated
domain. Combined with the geometric tail bounds of `sequence_space`, the
statement "the nearest singularity lies in this annulus" becomes a sound
enclosure rather than a fit.

That is the difference between this and a standard singularity-tracking
heuristic, and it is the part worth building carefully.

### Tracking in time

For a time-dependent field `u(x, t)`, compute the jet in `x` at each `t` and
track `x_s(t)`. Blow-up corresponds to `Im x_s -> 0`: the complex singularity
hits the real axis. The **rate** at which it approaches gives a blow-up time
estimate

```
| Im x_s(t) | ~ C ( t_c - t )^gamma
```

and fitting `(C, t_c, gamma)` is the standard complex-singularity method for
predicting blow-up.

This must be stated with maximum care, because it is adjacent to a famous open
problem. What the method provides is:

- a **diagnostic** that the solution is developing a singularity in the analytic
  continuation,
- an **estimate** of when, with a fit whose uncertainty is reportable,
- and, with certified remainders, an **enclosure** of the singularity location
  at each fixed time.

What it does not provide, under any circumstance, is a proof of finite-time
blow-up for the underlying PDE. That would require controlling the whole
solution, not its truncated Taylor data at sampled times. Spec 07-02 carries the
full honesty framing.

## 5. Worked example

**A known singularity, recovered.**

Take `f(x) = 1 / (1 - x/0.3)` with a simple pole at `x_s = 0.3`. Taylor
coefficients about `0`:

```
a_k = (1/0.3)^k = 3.3333333^k
a_0 = 1,  a_1 = 3.3333333,  a_2 = 11.111111,  a_3 = 37.037037,  a_4 = 123.45679
```

**Domb-Sykes.** The ratios are all exactly `3.3333333`, so the intercept is
`1/x_s = 3.3333333`, giving `x_s = 0.3` exactly, and the slope is zero, giving
`p = 1` — a simple pole. Both correct, from four coefficients.

**Padé.** For a geometric series `sum r^k x^k`, the `[0/1]` approximant is
exactly `1 / (1 - r x)`, so the single denominator root is `x = 1/r = 0.3`:
**exact from two coefficients**, because the function is in the Padé class. The
ideal case, and it shows the mechanism cleanly.

**A branch point, the realistic case.** Take `f(x) = (1 - x/0.3)^{-1/2}`, a
square-root branch point. Coefficients:

```
a_k = binom(-1/2, k) (-1/0.3)^k = (1/0.3)^k * (2k)! / (4^k (k!)^2)
a_0 = 1
a_1 = 0.5 * 3.3333333            = 1.6666667
a_2 = 0.375 * 11.111111          = 4.1666667
a_3 = 0.3125 * 37.037037         = 11.574074
a_4 = 0.2734375 * 123.45679      = 33.757716
```

Ratios: `a_1/a_0 = 1.6666667`, `a_2/a_1 = 2.5`, `a_3/a_2 = 2.7777778`,
`a_4/a_3 = 2.9166667`. These are *not* constant — they climb toward
`1/x_s = 3.3333`.

Domb-Sykes fit: plot ratio against `1/k`:

```
1/k     ratio
1.0000  1.6666667
0.5000  2.5000000
0.3333  2.7777778
0.2500  2.9166667
```

A least-squares line through the last three points gives slope and intercept.
Using the two extreme of those three, `(0.5, 2.5)` and `(0.25, 2.9166667)`:

```
slope     = (2.5 - 2.9166667) / (0.5 - 0.25) = -1.6666668
intercept = 2.9166667 - (-1.6666668)(0.25)   = 3.3333334
```

So `1/x_s = 3.3333334`, giving `x_s = 0.2999999`, and
`(p - 1)/x_s = slope = -1.6666668` gives `p - 1 = -0.5`, so `p = 0.5`.

**Both the location `0.3` and the exponent `1/2` recovered from five
coefficients**, to six digits, with nothing but a straight-line fit. That is why
the method is worth having, and why having many exact coefficients matters: the
whole thing rests on the coefficients being right.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/difference/singularity.py
@dataclass(frozen=True)
class SingularityEstimate:
    location: complex
    exponent: float
    method: Literal["domb_sykes", "pade", "both"]
    residual: float                  # fit quality
    enclosure: Interval | None       # populated when certified

def domb_sykes(coeffs: Sequence[float], *, drop_first: int = 1) -> SingularityEstimate: ...
def pade_singularities(coeffs, *, l: int, m: int) -> tuple[complex, ...]: ...
def certified_singularity_annulus(coeffs, *, tail_bound: Interval) -> Interval:
    """Sound enclosure of |x_s| from coefficient bounds plus a tail bound."""
def agreement(a: SingularityEstimate, b: SingularityEstimate) -> float:
    """Disagreement between methods is the honest uncertainty signal."""
```

```python
# omnibias/fields/singularity.py
def track_singularity(field, *, times, jet_order: int = 20, x0) -> SingularityTrack: ...

@dataclass
class SingularityTrack:
    times: FloatArray
    locations: ComplexArray
    exponents: FloatArray
    blowup_fit: BlowupFit | None
    disclaimer: str = "diagnostic and estimate; not a proof of blow-up"
```

Carrying the disclaimer as a field of the result object, not only in the docs, is
deliberate: this output will be copied into papers.

## 7. Practical use cases

1. **Adaptive refinement** (spec 03-13): refine where the singularity is
   approaching, using the tracked distance as the indicator.
2. **Step-size control** in time integration: the singularity distance bounds the
   usable step.
3. **Blow-up diagnostics** in fluid and reaction-diffusion problems, as an
   estimate with reported uncertainty.
4. **Self-similar exponent extraction** for the CCF campaign (spec 07-03), where
   the exponent `p` is physically meaningful.
5. **Analytic continuation.** Padé extends a solution beyond its Taylor radius,
   which is useful for far-field matching.

## 8. Acceptance gates

Baselines: coefficient-ratio estimation without the `1/k` extrapolation, and a
direct numerical singularity search.

- **G1 known-singularity recovery.** For a suite of functions with analytically
  known singularities (poles, branch points, essential singularities, and
  several singularities at comparable distance), location and exponent are
  recovered to `<= 1e-6` relative where the method applies, and the method
  *reports failure* on the essential-singularity cases rather than returning a
  confident wrong answer.
- **G2 enclosure soundness.** `certified_singularity_annulus` contains the true
  `|x_s|` on every instance of a randomized suite, with zero violations.
- **G3 coefficient cost.** Obtaining 20 exact coefficients from a depth-4
  network via `mlp_jet` is at least `20x` faster than repeated autodiff, and
  bit-identical to it where autodiff is feasible.
- **G4 method agreement.** On the known suite, Domb-Sykes and Padé agree within
  their reported uncertainties; where they disagree, the disagreement flags a
  genuinely hard case (verified by construction).
- **G5 tracking.** On a scalar model ODE with an analytically known blow-up
  time, the predicted `t_c` is within `1` percent well before blow-up, and the
  reported uncertainty contains the true value throughout.
- **G6 disclaimer integrity.** A test asserts that `SingularityTrack` always
  carries the disclaimer field and that it is included in any serialized output.

## 9. Benchmark plan

- `benchmarks/singularity_tracking.py`: recovery suite, enclosure soundness,
  coefficient-cost comparison, method-agreement study, model-ODE tracking.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/singularity/`.

## 10. Honesty and scope

- Jets come from the founding bias collapse (`delta -> 0`) tower. No temperature
  collapse appears.
- **This is a diagnostic and an estimate, not a proof of blow-up.** Locating a
  complex singularity of a truncated Taylor expansion at sampled times does not
  control the PDE solution. The `disclaimer` field exists because this output
  will be quoted, and spec 07-02 carries the full framing.
- Domb-Sykes assumes a **single dominant algebraic singularity**. With two
  singularities at comparable distance, or an essential singularity, it gives a
  confidently wrong answer, which is why G1 includes explicit failure cases and
  why method agreement is reported.
- Certified enclosures bound `|x_s|` given a coefficient tail bound. Obtaining
  the tail bound is the hard part and requires the field's own structure; where
  it is unavailable, the estimate is not certified and must be labelled so.
- Complex-singularity analysis is classical (Domb-Sykes, Sulem-Sulem-Frisch and
  successors). The contribution is exact cheap coefficients from the tower plus
  the certified remainder.

## 11. Open questions and risks

- **Coefficient conditioning.** High-order Taylor coefficients grow
  geometrically and lose relative precision; even exact formulas evaluated in
  float64 have a usable order ceiling. Measure it.
- **Padé spurious poles** (Froissart doublets) are endemic and must be filtered;
  the standard filter is pole-zero proximity, and it needs a threshold that must
  be justified rather than tuned to the test set.
- **Multivariate singularities.** The construction is along a ray. A genuine
  multivariate singular-variety analysis is much harder and is not claimed.
- **Falsifier.** If the tracked singularity location is too noisy to drive
  refinement decisions on realistic fields, the operational value is limited to
  post-hoc diagnostics.

## 12. Implementation checklist

- [ ] `packages/omnibias-difference/src/omnibias/difference/singularity.py`
- [ ] `packages/omnibias-fields/src/omnibias/fields/singularity.py`
- [ ] Reuse `pade_certified_remainder` and `sequence_space` tail bounds
- [ ] Recovery suite including deliberate failure cases
- [ ] Enclosure soundness test
- [ ] Froissart-doublet filter with a justified threshold
- [ ] Coefficient-order ceiling measured and recorded
- [ ] `disclaimer` field asserted present in serialized output
- [ ] `benchmarks/singularity_tracking.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
