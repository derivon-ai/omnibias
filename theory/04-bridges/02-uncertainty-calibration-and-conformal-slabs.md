# 04-02 Uncertainty and conformal slabs

## 1. Thesis and status

The band role's two hyperplanes are a **slab**, which is the natural geometry of
a prediction interval — so calibrated uncertainty, conformal coverage, and sound
interval enclosures can share one object, with the crucial discipline that
their guarantees are of three different kinds and must never be blended.

- **Status**: designed
- **Depends on**: 01-03, 03-08, 04-01
- **Blocks**: 05-01

## 2. Where it lands

`packages/omnibias-verify/src/omnibias/verify/uncertainty.py` for the
conformal and calibration machinery, since that package already owns the
sound-enclosure side and the contrast is the point.

## 3. Prior art in omnibias

- `packages/omnibias-verify/` — `TaylorModelMV` propagation, reachable-set
  certificates, robustness / Lipschitz / monotonicity certificates, branch and
  bound. This is the **sound-enclosure** register.
- `omnibias.core.verified.Interval` — outward-rounded interval arithmetic.
- `omnibias.core.proof.certificate` — hash-sealed certificate format v1 with
  `Verdict` and `verify_certificate_digest`.
- `omnibias-curvature` — Fisher information, which is the **frequentist**
  register's local approximation.
- `docs/operator-surface.md` — the `band` and `integral` roles: the slab
  geometry.

**Confirmed gap.** Sound enclosures exist. There is no conformal prediction, no
calibration machinery, and no statistical uncertainty of any kind — so there is
also no risk of confusing them yet, and this spec's main job is to make sure
that stays true when they coexist.

## 4. Mathematics

### Three kinds of guarantee, stated before anything else

| Register | Object | Guarantee | Assumptions |
|---|---|---|---|
| **Sound enclosure** | `Interval` from `omnibias.verify` | the true value of *this function on this box* is inside, with probability 1 | correctness of interval arithmetic only |
| **Conformal** | a prediction set | marginal coverage `>= 1 - alpha` over resampling | exchangeability of the data |
| **Bayesian / Fisher** | a credible or Wald interval | coverage under the model | the model is correct |

These are not interchangeable, they do not compose by intersection or union
without care, and reporting one while implying another is the single most
common failure in uncertainty quantification. **This spec's first deliverable is
a type system that keeps them apart.**

### The slab as a common shape

All three produce, in the simplest case, an interval `[lo, hi]` at each input —
a slab in `(x, y)` space bounded by two surfaces. When the surfaces are
hyperplanes in a feature space, that is exactly the `band` role's geometry:

```
band(z) = S(z + b_hi) - S(z + b_lo)
```

with `b_lo`, `b_hi` the two boundaries. So a slab-shaped predictor is a native
omnibias object, and the width `b_hi - b_lo` is a learnable parameter with exact
derivatives.

That shared *shape* is why the three registers get confused, and it is also why
having them in one library with distinct types is valuable.

### Conformal prediction on the slab

Split conformal: hold out a calibration set, compute non-conformity scores
`s_i = |y_i - f(x_i)|`, take the `ceil((n+1)(1-alpha))`-th smallest as `q`, and
predict `[f(x) - q, f(x) + q]`. Coverage `>= 1 - alpha` marginally, under
exchangeability, with no assumption on `f`.

The omnibias-specific improvements:

1. **Learnable width.** `q` can be replaced by a learned function `w(x)` with
   the conformal calibration applied to the *normalized* score `|y - f| / w(x)`,
   giving adaptive intervals with the same guarantee. `w` is a natural OMBU
   output, positive by construction through a softplus.
2. **Exact width derivatives.** Training `w` to minimize average width subject
   to calibration is a differentiable problem with closed-form gradients.
3. **Slab geometry in feature space.** For a model whose last layer is linear in
   features, the conformal slab is literally a pair of parallel hyperplanes.

### Combining registers, correctly

There is one legitimate combination and it must be spelled out. Suppose:

- a sound enclosure says the *model output* lies in `[a, b]` for inputs in a box
  (this is about the function, not the data), and
- a conformal interval says the *residual* is within `q` with probability
  `1 - alpha`.

Then a valid combined statement is: "with probability at least `1 - alpha` over
the data, the observed `y` lies in `[a - q, b + q]`, where `[a, b]` is
guaranteed". The probability applies only to the conformal part; the enclosure
part is certain. The combined object must record **both** components separately
so the statement can be reconstructed, rather than collapsing to a single
interval whose provenance is lost.

### Calibration diagnostics

Coverage is not enough: an interval that is too wide half the time and too
narrow the other half can have exactly nominal marginal coverage. Report:

- **marginal coverage**, the headline,
- **conditional coverage** by input strata, which is where adaptive methods earn
  their keep,
- **average width**, since trivially wide intervals always cover,
- **width-coverage curves** across `alpha`.

Any calibration report missing average width is not a calibration report.

## 5. Worked example

**Split conformal, worked with numbers.**

Calibration set of `n = 19` residual magnitudes, sorted:

```
0.02, 0.05, 0.07, 0.09, 0.11, 0.14, 0.16, 0.19, 0.22, 0.25,
0.28, 0.33, 0.38, 0.44, 0.51, 0.60, 0.72, 0.88, 1.15
```

For `alpha = 0.1`, the conformal quantile index is

```
ceil( (n + 1)(1 - alpha) ) = ceil( 20 * 0.9 ) = ceil(18) = 18
```

so `q` is the 18th smallest, `q = 0.88`. The prediction interval is
`[f(x) - 0.88, f(x) + 0.88]`, with marginal coverage at least `0.9` under
exchangeability.

Note what the finite-sample correction does: the naive empirical `0.9`-quantile
of 19 points would be the 17th or 18th value depending on convention, and the
`(n+1)` correction is what makes the guarantee exact rather than approximate.
Getting that index wrong is the classic conformal bug, and it is why gate G1
tests it against the finite-sample theory rather than against a large-sample
approximation.

**Adaptive width, in the population limit.** Now suppose the residual is
Gaussian with scale `0.1` on half the domain and scale `1.0` on the other half,
and take `alpha = 0.1` with an infinite calibration set so the quantiles are
exact.

Fixed conformal solves for the marginal `0.9`-quantile of `|residual|`:

```
0.5 * P(|N(0, 0.1)| <= q) + 0.5 * P(|N(0, 1)| <= q) = 0.9
```

For any `q` above about `0.4` the first term is `1` to four decimals, so
`0.5 + 0.5 * P(|N(0,1)| <= q) = 0.9`, giving `P = 0.8` and `q = 1.2816`.

Adaptive conformal calibrates the *normalized* score `|residual| / w(x)` with
`w` tracking the local scale; that score is a standard normal magnitude, whose
`0.9`-quantile is `1.6449`, so the half-widths are `0.1645` and `1.6449`.

|  | fixed | adaptive |
|---|---|---|
| average half-width | `1.2816` | `0.9047` |
| coverage, low-noise half | `1.0000` | `0.90` |
| coverage, high-noise half | `0.80` | `0.90` |
| marginal coverage | `0.90` | `0.90` |

The width improvement is real but modest, a factor of `1.42`. **The important
number is the conditional coverage**: fixed conformal delivers only `80` percent
coverage exactly where the noise is large, while wasting width where it is
small, and it hides this behind a perfectly nominal marginal `90` percent. That
asymmetry is the argument for adaptive conformal, and it is why the acceptance
gate is written on conditional coverage rather than on average width alone.

**The register contrast, made concrete.** For the same model:

- `omnibias.verify` might certify that the model output on the input box
  `[0.4, 0.6]` lies in `[2.31, 2.47]`. That is **certain**, about the function.
- Conformal says the residual is within `0.88` with probability `0.9`. That is
  **probabilistic**, about the data, and assumes exchangeability.
- A Fisher-based Wald interval might say `+-0.4`. That is **model-dependent**
  and is wrong if the model is misspecified.

Three intervals, three meanings, and the numbers alone do not distinguish them.
Hence the types.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/verify/uncertainty.py
class GuaranteeKind(StrEnum):
    SOUND_ENCLOSURE = "sound_enclosure"     # probability 1, about the function
    CONFORMAL = "conformal"                 # marginal coverage, exchangeability
    MODEL_BASED = "model_based"             # Bayesian / Fisher; model must be right

@dataclass(frozen=True)
class UncertaintyInterval:
    lo: float
    hi: float
    kind: GuaranteeKind
    level: float | None                     # None for SOUND_ENCLOSURE
    assumptions: tuple[str, ...]

    def __add__(self, other): 
        raise TypeError(
            "Intervals of different guarantee kinds do not combine by "
            "arithmetic. Use `combine_enclosure_with_conformal`."
        )

def split_conformal(
    residuals: FloatArray, *, alpha: float,
) -> float:
    """The finite-sample corrected quantile. Index is ceil((n+1)(1-alpha))."""

def adaptive_conformal(residuals, widths, *, alpha: float) -> float: ...

def combine_enclosure_with_conformal(
    enclosure: UncertaintyInterval, q: float, *, alpha: float,
) -> CombinedStatement:
    """Returns a structured statement carrying both components separately; never
    a single interval."""

@dataclass(frozen=True)
class CalibrationReport:
    marginal_coverage: float
    conditional_coverage: Mapping[str, float]
    average_width: float                    # required, not optional
    width_coverage_curve: tuple[tuple[float, float], ...]
```

Making `__add__` raise is not a joke: silently adding a conformal interval to a
sound enclosure is exactly the error this spec exists to prevent.

## 7. Practical use cases

1. **Scientific reporting.** A result with an interval whose guarantee kind is
   explicit is publishable; one without is not.
2. **Adaptive intervals in inverse problems** (spec 05-01), where noise varies
   strongly across the domain.
3. **Safety-critical filtering.** Combining a sound reachable-set enclosure with
   a conformal residual bound, correctly.
4. **Model comparison** by average width at matched coverage, which is the right
   metric and is frequently omitted.
5. **Calibration monitoring** in deployed models, with the conditional-coverage
   breakdown as the early-warning signal.

## 8. Acceptance gates

Baselines: fixed-width conformal, quantile regression, and a Gaussian-likelihood
model interval.

- **G1 finite-sample coverage.** Empirical coverage over 10 000 resamplings is
  at least `1 - alpha` for `alpha in {0.01, 0.05, 0.1, 0.2}` and calibration
  sizes `n in {19, 99, 999}`, matching the finite-sample theory including the
  `(n+1)` correction. A deliberate off-by-one in the index must fail this gate.
- **G2 type safety.** Attempting to combine intervals of different guarantee
  kinds by arithmetic raises. Asserted by test.
- **G3 adaptive win.** On a heteroscedastic problem at matched marginal
  coverage, over five seeds: the **maximum conditional-coverage deviation** from
  nominal across strata is at most `0.03` for adaptive versus at least `0.08`
  for fixed, and adaptive average width is no worse than fixed. Conditional
  coverage is the primary criterion here; a width-only gate would reward the
  wrong behaviour.
- **G4 exchangeability honesty.** On deliberately non-exchangeable data (a
  distribution shift), coverage degrades and the report *says so* via a
  detectable diagnostic, rather than silently under-covering.
- **G5 combination correctness.** `combine_enclosure_with_conformal` produces a
  statement whose empirical validity is confirmed by simulation, and never
  returns a bare interval.
- **G6 report completeness.** A `CalibrationReport` without `average_width`
  cannot be constructed; asserted by the type.

## 9. Benchmark plan

- `benchmarks/conformal_slabs.py`: coverage sweeps across `alpha` and `n`,
  adaptive-versus-fixed width comparison, distribution-shift degradation study,
  combination validation.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/uncertainty/`.

## 10. Honesty and scope

- The slab geometry comes from the `band` role with a **finite** gap — the
  opposite of the founding `delta -> 0` bias collapse. No collapse of either
  kind appears in this spec, and in particular no temperature collapse.
- **The three registers are not interchangeable.** The table in section 4 must
  appear in any user-facing document, and the type system enforces the
  separation at runtime.
- Conformal coverage is **marginal** and assumes **exchangeability**. It is not
  conditional coverage, and it fails under distribution shift. Both facts are
  gated (G1, G4).
- Model-based intervals are only as good as the model. This is stated plainly
  rather than left implicit.
- Conformal prediction is an established field (Vovk, Shafer, Lei, Romano and
  successors). The contribution is the type-level separation of guarantee kinds,
  the slab parameterization with exact width derivatives, and the correct
  combination rule with sound enclosures.
- Certificate tier: the sound-enclosure component can be sealed in the existing
  v1 format; the conformal component **cannot** and must not be, since it is a
  statistical statement, not an enclosure.

## 11. Open questions and risks

- **Conditional coverage is impossible in general** (a known negative result:
  distribution-free conditional coverage requires uninformative intervals).
  Adaptive methods improve it in practice without guaranteeing it, and that
  distinction must survive into the documentation.
- **Exchangeability detection** is itself a statistical problem; the G4
  diagnostic will have false positives and negatives, and its operating
  characteristics must be reported.
- **Sealing temptation.** Someone will want to seal a conformal interval into
  the certificate format because it looks like a bound. The format's schema
  should make that impossible rather than merely discouraged.
- **Falsifier.** If adaptive conformal does not improve conditional coverage
  over fixed conformal (G3), the learnable-slab machinery adds nothing over a
  well-known baseline, and the remaining value is the type discipline — which is
  real but small, and should then be shipped as exactly that rather than dressed
  up as a method.

## 12. Implementation checklist

- [ ] `packages/omnibias-verify/src/omnibias/verify/uncertainty.py`
- [ ] `GuaranteeKind` enum with arithmetic between kinds raising
- [ ] Finite-sample quantile index with an off-by-one regression test
- [ ] Adaptive width via a positive OMBU output with exact derivatives
- [ ] `CalibrationReport` requiring `average_width` at the type level
- [ ] Distribution-shift diagnostic with reported operating characteristics
- [ ] Certificate schema guard preventing conformal intervals from being sealed
- [ ] Coverage sweeps across `alpha` and `n` in the benchmark
- [ ] `benchmarks/conformal_slabs.py` plus smoke JSON
- [ ] Docs page and nav entry, carrying the three-register table
- [ ] Index row in `theory/README.md`
