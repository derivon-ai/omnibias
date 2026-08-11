# 03-08 Certified scan localization

## 1. Thesis and status

A bias scan's response peaks where the feature is, and the whole scan is a
closed-form function of one variable, so interval arithmetic on the response can
produce a **sound enclosure of the peak location** — a certified answer to
"where is the interface", not an estimate.

- **Status**: designed
- **Depends on**: 01-02, 01-09
- **Blocks**: 04-02, 05-01, 07-02

## 2. Where it lands

`packages/omnibias-verify/src/omnibias/verify/localization.py`. The verification
package owns certificates over network outputs; this is a certificate over a
scan response.

## 3. Prior art in omnibias

- `packages/omnibias-verify/` — certified NN verification: `TaylorModelMV`
  propagation, ReLU / GELU / max-pool enclosures, branch and bound, robustness /
  Lipschitz / monotonicity / reachable-set certificates, torch and jax weight
  ingestion.
- `omnibias.core.verified` — `Interval` (outward-rounded), `affine` zonotopes,
  `TaylorModel` / `TaylorModelMV`, `kantorovich` (`certify_zero_radii`,
  `krawczyk_certificate`).
- `omnibias.core.proof.certificate` — the hash-sealed certificate format v1 with
  `verify_certificate_digest`, and `Verdict` carrying
  `certificate_schema_version`.
- Spec 01-02's `BiasScan`; spec 01-09's Krawczyk locus certification.

**Confirmed gap.** Verification certifies *network outputs* over input boxes.
Nothing certifies the *location of a feature* — a statement about an argmax or a
root rather than about a value.

## 4. Mathematics

### The problem

Given a scan response `r(tau) = < template, field >` at bias offset `tau`, the
feature location is

```
tau* = argmax_tau r(tau)
```

A point estimate is easy. A *certificate* — an interval `[tau_lo, tau_hi]`
guaranteed to contain `tau*` — is what a scientific claim needs.

### Two routes, and why the second is better

**Route 1: enclose the maximum.** Compute an interval enclosure `R` of
`max_tau r(tau)` by branch and bound over `tau`, then keep the sub-intervals
whose upper bound exceeds the global lower bound. This is the standard interval
global-optimization argument and it is sound. Its weakness is that near a flat
maximum, many sub-intervals survive and the enclosure is wide.

**Route 2: certify the stationary point.** `tau*` satisfies `r'(tau*) = 0` with
`r''(tau*) < 0`. Both `r'` and `r''` are **closed form** (the scan response is a
sum of `sigma^(n)` terms, so differentiating in `tau` just raises the order).
So the problem becomes: certify a simple root of a closed-form function, which
is exactly what `krawczyk_certificate` does.

The Krawczyk operator on `r'` over a box `T`:

```
K(T) = m - r'(m) / r''(m) + ( 1 - r''(T) / r''(m) ) ( T - m ),   m = mid(T)
```

If `K(T)` is contained in the interior of `T`, then `r'` has a **unique** zero
in `T`, and that zero is enclosed by `K(T)`. Iterating contracts the enclosure
quadratically.

Route 2 gives a much tighter interval and, crucially, also proves **uniqueness**
in the box — so it certifies not just "the peak is somewhere here" but "there is
exactly one peak here". Route 1 cannot do that.

Use route 2 as the primary, with route 1 as a fallback for degenerate cases
(vanishing `r''`, that is a flat or inflected peak).

### Sign conditions and the second-derivative test

Certifying a *maximum* rather than a stationary point requires
`r''(tau) < 0` throughout the enclosure. That is an interval sign check on a
closed-form quantity: cheap, and it must be part of the certificate rather than
assumed.

### From offset to physical location

The scan is in pre-activation units, so `tau*` maps to a physical location
`x* = -(tau* + b) / |w|` along the normal. Interval arithmetic carries through,
but the division by `|w|` inflates the interval when `|w|` is small or itself
uncertain. Report both the offset enclosure and the physical enclosure, and be
explicit that the second inherits the direction's uncertainty.

### Measurement noise

If the field is data with noise, the response is a random variable and a purely
deterministic enclosure is not the right object. Two honest options:

1. **Enclose given the data.** The certificate is conditional: "for *this*
   observed response, the peak is in this interval". Sound and useful, and it
   says nothing about the noise.
2. **Combine with a conformal interval** (spec 04-02) to get a
   distribution-free coverage statement over resampling.

Do not blend them silently. A deterministic enclosure of a noisy quantity is a
statement about the realized data, and labelling it as a statement about the
underlying truth would be wrong.

## 5. Worked example

A single interface at `x_0 = 0.3` in a field `u(x) = tanh(20 (x - 0.3))`.

Scan template: order-1 (a bump), direction `w = 1`, so the response is

```
r(tau) = integral u'(x) sigma'( x + tau ) dx
```

which peaks when the bump aligns with the interface, that is at `tau = -0.3`.

**Certifying it.** Take the starting box `T_0 = [-0.4, -0.2]`. On this box:

- `r'` is closed form (one order up the tower for the template),
- `r''` likewise,
- the interval evaluation of `r''` over `T_0` is strictly negative (a single
  well-separated peak), so the second-derivative test passes on the whole box.

Krawczyk step with `m = -0.3`:

```
r'(m)  = 0            (by symmetry of this constructed example)
K(T_0) = -0.3 + ( 1 - r''(T_0)/r''(m) ) ( T_0 + 0.3 )
```

The factor `1 - r''(T_0)/r''(m)` measures how much `r''` varies over the box. If
`r''` varies by `+-15` percent across `T_0`, the factor is `[-0.15, 0.15]` and

```
K(T_0) = -0.3 + [-0.15, 0.15] * [-0.1, 0.1] = -0.3 + [-0.015, 0.015]
       = [-0.315, -0.285]
```

which is strictly inside `T_0 = [-0.4, -0.2]`. **Therefore a unique stationary
point exists in `T_0` and lies in `[-0.315, -0.285]`.** One step has contracted
the enclosure from width `0.2` to width `0.03`.

Iterating on the new box, the variation of `r''` over a `0.03`-wide box is
correspondingly smaller — roughly `2` percent rather than `15` — so the next
enclosure has width about `0.0006`, and the one after about `2.4e-7`: quadratic
contraction, as Krawczyk guarantees for a simple root.

The deliverable is not "the interface is at about `0.3`" but "**the interface is
in `[0.2999999, 0.3000001]`, and there is exactly one interface in
`[0.2, 0.4]`**". The uniqueness clause is what makes it a scientific statement.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/verify/localization.py
@dataclass(frozen=True)
class LocalizationCertificate:
    offset_enclosure: Interval
    physical_enclosure: Interval
    unique_in: Interval             # the box uniqueness was proved on
    second_derivative_sign: Literal["negative", "positive", "indeterminate"]
    route: Literal["krawczyk", "branch_and_bound"]
    scope: str = "local_box"        # honest scope, as the verified register requires

def certify_peak(
    response: ScanResponse, *, box: Interval, max_iter: int = 20,
) -> LocalizationCertificate | Inconclusive:
    """Krawczyk on r'; falls back to branch and bound when r'' is
    indeterminate. Returns Inconclusive rather than over-claiming."""

def certify_multiple_peaks(response, *, box, max_peaks: int) -> tuple[...]:
    """Exhaustive subdivision; reports how many peaks are proved to exist and
    whether the search was exhaustive over the box."""

def seal(cert: LocalizationCertificate) -> Certificate:
    """Into the existing hash-sealed v1 format."""
```

Returning `Inconclusive` as a distinct type, rather than a wide interval, keeps
the soundness-not-completeness discipline visible in the type system.

## 7. Practical use cases

1. **Certified interface localization** in materials and geophysics, where "the
   layer boundary is in this interval, and there is exactly one" is the claim a
   practitioner needs.
2. **Certified crack and defect detection**, with uniqueness ruling out missed
   secondary defects in the searched region.
3. **Free-boundary certification** (spec 02-12), where the boundary is the
   answer.
4. **Shock-position certification** in conservation-law solutions.
5. **Peak finding in spectroscopy** with a guaranteed peak count in a window,
   which is exactly the question spectroscopists ask.

## 8. Acceptance gates

Baselines: gradient-based peak finding, and interval branch-and-bound alone.

- **G1 soundness.** On at least 10 000 randomized instances with analytically
  known peaks, the enclosure contains the true peak every time. Zero violations,
  checked on a dense deterministic grid **and** a random sample of instances.
- **G2 uniqueness correctness.** Whenever uniqueness is claimed, exhaustive
  subdivision confirms exactly one stationary point in the box. Whenever there
  are genuinely two, the method returns `Inconclusive` or splits, never a false
  uniqueness claim.
- **G3 tightness.** The Krawczyk route's enclosure width is at least `100x`
  smaller than branch and bound alone at matched iteration count.
- **G4 quadratic contraction.** Enclosure width squares per iteration for simple
  roots, over at least three iterations, matching the theory.
- **G5 graceful degradation.** For flat or inflected peaks the method returns
  `Inconclusive` with an explanatory reason, and a test asserts it never returns
  a certificate in those cases.
- **G6 seal integrity.** Sealed certificates pass `verify_certificate_digest`
  and a tampering test fails it.

## 9. Benchmark plan

- `benchmarks/certified_localization.py`: soundness sweep, tightness and
  contraction rates, degenerate-case behaviour, noisy-data conditional
  enclosures.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/localization/`.

## 10. Honesty and scope

- The scan template comes from the founding bias collapse (`delta -> 0`). No
  temperature collapse appears.
- **Soundness, not completeness.** `Inconclusive` is a first-class return value.
  The verified register's rule applies: return inconclusive rather than
  over-claim.
- The certificate's `scope` is `local_box`: it is a statement about a specific
  box, not a global one. Uniqueness is uniqueness *in that box*.
- **A deterministic enclosure of noisy data is conditional on the data.** It is
  not a statement about the underlying truth under resampling. Where that is
  wanted, combine with spec 04-02's conformal machinery and label the result as
  a combination.
- The physical enclosure inherits uncertainty from `|w|` and must be reported
  separately from the offset enclosure.
- Certificate tier: sound enclosure, sealed in the v1 format. Not
  `theorem_prover_verified`; the underlying statement involves a continuum
  quantifier and is outside the finite-rational kernel scope.

## 11. Open questions and risks

- **Wrapping in interval arithmetic.** Naive interval evaluation of `r''` over a
  wide box is very loose. Taylor models or affine forms are likely necessary,
  and both are available; measure which is needed.
- **Multiple nearby peaks** defeat uniqueness and force subdivision, with cost
  growing as peaks approach each other. Report the minimum resolvable
  separation.
- **High dimensions.** The construction above is on a one-dimensional offset
  axis. A multi-normal version needs a multivariate Krawczyk operator, which
  exists but is more expensive; do not claim it before building it.
- **Falsifier.** If interval wrapping makes the enclosures so loose that a naive
  gradient method plus a bootstrap gives comparable practical intervals, the
  certificate is a formality rather than an advance.

## 12. Implementation checklist

- [ ] `packages/omnibias-verify/src/omnibias/verify/localization.py`
- [ ] Reuse `krawczyk_certificate` and the existing `Interval` / `TaylorModel`
      machinery; fork nothing
- [ ] `Inconclusive` as a distinct return type, not a sentinel interval
- [ ] Soundness sweep with dense grid **and** random sample
- [ ] Uniqueness correctness test including deliberate two-peak cases
- [ ] Contraction-rate test
- [ ] Degenerate-case refusal test
- [ ] Seal and tamper tests against `verify_certificate_digest`
- [ ] `benchmarks/certified_localization.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
