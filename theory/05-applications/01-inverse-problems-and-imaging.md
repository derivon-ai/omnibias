# 05-01 Inverse problems and imaging

## 1. Thesis and status

Most imaging inverse problems are asking two questions — **where is the
interface** and **what kind of jump does it carry** — and those are exactly the
two things the scan and the multi-pack answer natively, with the localization
certifiable and the jump order explicit rather than inferred from a pixel grid.

- **Status**: gated
- **Depends on**: 01-01, 01-02, 01-06, 02-01, 02-05, 02-06, 02-11, 02-12, 03-05, 03-08, 03-09, 03-13, 04-02
- **Blocks**: none

Wave-0 falsifier G7 is recorded in
`docs/benchmarks/inverse_imaging.json` (`all_passed: true`). Gates G1–G6 and
the `omnibias.pinn.inverse` product API remain unearned.

## 2. Where it lands

`packages/omnibias-pinn/src/omnibias/pinn/inverse/` as an application submodule,
since inverse problems reuse the solver, the field ops, and the operator
(DeepONet / FNO) surface that already live there. It earns a submodule, not a
package: no distinct dependency tier, no distinct audience.

## 3. Prior art in omnibias

- `omnibias.pinn.solver` — mesh-free PDE solver; the forward model.
- `omnibias.pinn.operator` — DeepONet / FNO with conditioning; amortized
  inversion lives here.
- `omnibias.pinn.domain` — SDF and hard curved boundary conditions; the geometry
  of an unknown domain.
- `omnibias.pinn.certified` — certified residual machinery.
- `omnibias.shape` — soft occupancy fields and soft-coverage operators.
- `omnibias.verify` — Lipschitz, robustness, and reachable-set certificates.
- `omnibias.core.verified.kantorovich` — radii-polynomial existence, which is
  how a recovered solution is proven to exist near the computed one.
- `omnibias.measure` — importance reweighting, useful for non-uniform sensors.
- `benchmarks/inverse_imaging.py` — Wave-0 G7 falsifier (gated): logistic
  bias-scan localization scaling `alpha^(n - 5/2)` for `n in {3, 4}`.

**Confirmed gap.** There is a forward solver and an operator-learning surface,
but no inverse-problem module: no interface localization product API, no
layered-material inversion, no free-boundary estimation, no sensor-placement
design. The G7 falsifier measures the scaling law only.

## 4. Mathematics

### The three canonical problems, and why the primitives fit

**(a) Interface localization.** Given samples of a field with a jump across an
unknown hyperplane `w . x = tau*`, recover `tau*`. The bias scan (spec 01-02)
computes, for all `tau` in one sweep,

```
r_n(tau) = (1/N) sum_i y_i alpha^n sigma^(n)( alpha (w . x_i - tau) )
```

Integrating by parts `n - 1` times moves every derivative onto the data and
leaves the normalized mollifier `phi_alpha(y) = alpha sigma'(alpha y)`:

```
r_n(tau) = (-1)^(n-1) ( u^(n-1) * phi_alpha )(tau)
```

If `u` has a jump of size `J` in its `m`-th derivative then `u^(m+1)` contains
`J delta(x - tau*)`, so the peak appears in channel

```
n = m + 2
```

with height `J alpha sigma'(0)` and width `~ 1/alpha`. The scan therefore
answers *both* questions at once: the peak location is `tau*`, and the channel
index minus two is the jump order. Spec 03-08 then encloses the peak soundly, so
the answer is "the interface is in `[tau_lo, tau_hi]`, certified" rather than an
estimate.

**(b) Layered-material inversion.** Given boundary measurements of a stratified
medium, recover the layer positions and the contrast at each. The
transfer-matrix architecture (spec 02-11) makes the forward map a product of
`2 x 2` matrices whose derivatives with respect to layer parameters are exact,
so the inversion is a small, well-conditioned nonlinear least squares with
analytic Jacobian instead of a finite-differenced one.

**(c) Free-boundary estimation.** The unknown is a moving interface — a melting
front, a phase boundary, a shock. The equality-locus machinery (spec 01-09, and
the layer in 02-12) makes the boundary the *solution* of `f_1 = f_2`, so it is
represented exactly and its motion has a closed-form velocity from the implicit
function theorem:

```
d tau / dt = - ( d(f_1 - f_2) / dt ) / ( d(f_1 - f_2) / d tau )
```

Both derivatives come from the tower, so the front velocity is exact rather than
finite-differenced across time steps.

### Regularization, and the honest statement about it

Inverse problems are ill-posed; the answer depends on the regularizer, and no
amount of exact differentiation changes that. What the tower changes is that
the regularizer can be a **statement about derivatives at a specific order**
rather than a generic smoothness penalty:

- total variation regularizes order 1,
- curvature regularizes order 2,
- a multi-pack lets you regularize order `k` at some locations and order `j` at
  others, which is the right prior for a piecewise-smooth medium with known
  interface types.

This is a real modelling gain, and it is separate from the claim that the
solution is correct.

### Identifiability, before inversion

Spec 04-01's Fisher metric answers "which parameters can this data determine",
and for inverse problems that question should be asked **before** running the
inversion, not discovered afterwards from an unstable result. The metric's small
eigenvalues name the unidentifiable directions explicitly.

### Sensor placement as an experiment-design problem

Given a budget of `M` sensors, choose locations maximizing the smallest Fisher
eigenvalue (E-optimal) or the log-determinant (D-optimal). The log-determinant
objective is submodular in the sensor set, so `omnibias.submodular`'s continuous
greedy with its `(1 - 1/e)` guarantee applies directly. That is an unusually
clean fit: an established guarantee attaching to a real design problem with no
new theory required.

## 5. Worked example

**Interface localization with a certified answer.**

A one-dimensional field with a jump in the first derivative at `tau* = 0.37`:

```
u(x) = 0.5 x                     for x < 0.37
u(x) = 0.5 x + 2.0 (x - 0.37)    for x >= 0.37
```

so `u'` jumps from `0.5` to `2.5`: a jump of size `J = 2.0` at order `m = 1`.
Sample `u` at `N = 200` points on `[0, 1]` with Gaussian noise of standard
deviation `s = 0.01`.

By the channel rule, `n = m + 2 = 3`. The signal part of the response is

```
r_3(tau) = J phi_alpha(tau - tau*) = J alpha sigma'( alpha (tau - tau*) )
```

a clean peak at `tau*` of height `J alpha sigma'(0) = 2.0 * alpha * 0.25` and
width `~1/alpha`. At `alpha = 100` the peak height is `50`.

**Noise.** The noise term is `(1/N) sum_i eps_i K_3(x_i)`, whose standard
deviation is

```
sd( r_3 ) = s alpha^(n - 1/2) || sigma^(n) ||_2 / sqrt(N)
```

The logistic tower's `L^2` norms are exactly rational under the substitution
`t = sigma(y)`:

```
|| sigma''' ||_2^2  = int_0^1 t(1-t)(1 - 6t + 6t^2)^2 dt        = 1/42
|| sigma'''' ||_2^2 = int_0^1 t(1-t)(1 - 14t + 36t^2 - 24t^3)^2 dt = 1/30
```

so `||sigma'''||_2 = 0.154303`. At `alpha = 100`, `alpha^2.5 = 1e5`, and

```
sd( r_3 ) = 0.01 * 1e5 * 0.154303 / sqrt(200) = 10.91
```

against a peak of `50`, a signal-to-noise ratio of `4.58`.

**Localization error.** The peak estimator solves `r_3'(tau) = 0`, so its
standard deviation is `sd(r_3') / |r_3''(tau*)|`. The curvature at the peak is
`J alpha^3 sigma'''(0) = -2.5e5`, and `sd(r_3') = s alpha^3.5 ||sigma''''||_2 /
sqrt(N) = 0.01 * 1e7 * 0.182574 / 14.142 = 1291`. Hence

```
sd( tau_hat ) = 1291 / 2.5e5 = 5.2e-3
```

**The scaling, which is the useful part.** Collecting powers of `alpha`:

```
sd( tau_hat )  ~  alpha^(n - 5/2)
```

For `n = 3` that is `alpha^(1/2)`: **localization gets worse as the scan is
sharpened**. The peak narrows as `1/alpha` but the noise in an order-`n` channel
grows as `alpha^(n - 1/2)`, and for any channel above `n = 2` the noise wins.
Dropping to `alpha = 25` improves the localization standard deviation by
`sqrt(4) = 2x`, to `2.6e-3`, even though the peak is four times wider.

So the design rule is the opposite of the intuitive one: **use the smallest
`alpha` the problem allows**, bounded below only by (i) the kernel having to fit
inside the domain without touching the boundary, and (ii) the smooth background
`(u_smooth^(n-1) * phi_alpha)` staying below the peak. For this piecewise-linear
field the background in channel 3 is identically zero, so the binding constraint
is the domain: at `tau* = 0.37` the kernel must decay before `x = 0`, which
needs roughly `alpha > 20`.

Note also that the scan is *not* statistically efficient here. The Cramér-Rao
bound for this model, using every sample, is `4.5e-4` — about ten times better —
because the kink changes `u` globally and the scan deliberately looks only
locally. That is the price of locality and robustness to an unknown background,
and it should be quoted rather than hidden.

**The certified statement.** Spec 03-08's Krawczyk enclosure on `r_3'(tau) = 0`
turns the noiseless part into: the peak of the response lies in
`[0.3693, 0.3707]`, guaranteed, and is unique in that interval. Combined with a
conformal bound on the noise contribution (spec 04-02, keeping the two guarantee
kinds separate), the reportable result is an enclosure plus a coverage
statement, not a single number with an error bar of unstated provenance.

**The certified statement.** Spec 03-08's Krawczyk enclosure on `r'(tau) = 0`
turns this into: the peak of the noiseless response lies in
`[0.3693, 0.3707]`, guaranteed, and is unique in that interval. Combined with a
conformal bound on the noise contribution (spec 04-02, keeping the two
guarantee kinds separate), the reportable result is an enclosure plus a
coverage statement, not a single number with an error bar of unstated
provenance.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/pinn/inverse/_core.py
@dataclass(frozen=True)
class InterfaceEstimate:
    location: Interval              # sound enclosure, spec 03-08
    jump_order: int                 # which scan channel peaked
    jump_size: float
    unique_in_enclosure: bool

def locate_interface(
    points, values, normal, *, order: int, alpha: float,
) -> InterfaceEstimate: ...

def invert_layered(
    measurements, *, n_layers: int, init: LayerStack | None = None,
) -> LayerStack:
    """Transfer-matrix inversion with the analytic Jacobian of spec 02-11."""

def track_free_boundary(field_pair, *, t_span) -> BoundaryTrack:
    """Equality locus of f_1 = f_2 in time, with exact front velocity."""

def identifiability(model, params, data_design) -> IdentifiabilityReport:
    """Fisher eigenvalues and the named unidentifiable directions."""

def place_sensors(model, params, candidates, *, budget: int) -> SensorPlan:
    """D-optimal design via omnibias.submodular continuous greedy, carrying
    the (1 - 1/e) guarantee."""
```

## 7. Practical use cases

1. **Non-destructive testing.** Locate a delamination or crack plane in a
   layered composite from surface measurements, with a certified depth
   enclosure.
2. **Seismic layer inversion.** Recover layer depths and impedance contrasts;
   the transfer-matrix forward model is the standard one, now with exact
   derivatives.
3. **Phase-front tracking.** Solidification and melting fronts, where the
   boundary is the quantity of interest and its velocity is what physics
   models predict.
4. **Medical elastography.** Tissue stiffness interfaces from displacement
   fields, where jump order distinguishes tissue boundary types.
5. **Sensor-budget design** for any of the above, before instruments are bought
   or deployed.

## 8. Acceptance gates

Baselines, all named and all classical: total-variation deconvolution for
interface localization, Levenberg-Marquardt with finite-difference Jacobian for
layered inversion, and a level-set method for free boundaries.

- **G1 localization accuracy.** On synthetic data with known `tau*`, the
  estimate's absolute error is at most `0.5x` that of TV deconvolution at
  matched compute, over ten noise seeds and three noise levels. **Unearned.**
- **G2 enclosure soundness.** The certified enclosure contains the true `tau*`
  in `100%` of `1000` synthetic instances. **A single miss is a bug, not a
  tolerance issue.** **Unearned.**
- **G3 jump-order identification.** The correct jump order is identified in at
  least `95%` of instances at signal-to-noise ratio `10` or better, with the
  confusion matrix reported rather than only the accuracy. **Unearned.**
- **G4 layered inversion.** Recovers layer positions to within `1%` of layer
  thickness and contrasts to within `5%`, in fewer iterations than
  finite-difference Levenberg-Marquardt, on a five-layer problem. **Unearned.**
- **G5 free-boundary tracking.** Front position error against an analytic Stefan
  solution is at most `0.3x` a level-set baseline's at matched time step.
  **Unearned.**
- **G6 sensor design.** D-optimal placement beats uniform placement on posterior
  volume by at least `2x`, with the `(1 - 1/e)` guarantee reported alongside the
  achieved value. **Unearned.**
- **G7 alpha scaling.** The predicted `alpha^(n - 5/2)` scaling of localization
  standard deviation is confirmed over at least a decade of `alpha`, with the
  fitted exponent within `0.1` of the prediction, for `n in {3, 4}`. This is the
  gate that validates the design rule; if it fails, the rule is withdrawn rather
  than reworded. **Earned for the locally-seeded estimator** — see
  `docs/benchmarks/inverse_imaging.json`: five seeds, worst-seed deviations
  `0.016` (`n=3`, expected `0.5`) and `0.031` (`n=4`, expected `1.5`) over
  1.2 decades; rational `||sigma'''||_2^2 = 1/42` and
  `||sigma''''||_2^2 = 1/30`; discrete `sd(r')` within 3 sigma of Monte Carlo;
  capture rate `1.0` under the pre-registered regime. A `tau*`-free global
  argmax earns the same claim for `n=3` only; for `n=4` a spurious boundary
  maximum near `tau -> 1` dominates at large `alpha` (recorded in
  `honesty.boundary_artifact`, not softened into a pass).

## 9. Benchmark plan

- `benchmarks/inverse_imaging.py` with four families: interface localization
  (accuracy, enclosure coverage, order confusion, `alpha` sweep), layered
  inversion, Stefan free boundary, sensor design. **Landed so far:** the
  `alpha`-sweep family only (Wave-0 G7); the other three families remain open.
- Smoke JSON committed; full multi-seed under `$OMNIBIAS_SCRATCH/inverse/`.

## 10. Honesty and scope

- **Ill-posedness is not removed.** Exact derivatives improve conditioning and
  make the regularizer expressible at a chosen order; they do not make an
  ill-posed problem well-posed. Every result depends on the regularizer, and the
  regularizer must be reported with the result.
- The certified enclosure (G2) is about the **noiseless response functional** on
  the given samples. Noise is handled in the separate statistical register (spec
  04-02) and the two must not be merged into one interval.
- The scan's `alpha` is a **tempering scale**, not a collapse parameter of
  either kind. The multi-pack's order comes from the founding `delta -> 0`
  bias collapse; no temperature collapse appears in this spec.
- The `(1 - 1/e)` sensor-design guarantee is submodular-maximization theory
  applied to the log-determinant objective; it is a guarantee about the
  *optimization*, not about recovery quality.
- Interface detection, transfer-matrix inversion, and level-set free boundaries
  are all mature fields with strong classical methods. The claim is a better
  Jacobian and a certified localization, not a new subject.

## 11. Open questions and risks

- **Multidimensional interfaces.** The scan localizes a hyperplane given its
  normal. Unknown normals require a search over the sphere, and the cost and
  reliability of that search are unmeasured.
- **Curved interfaces** are only locally hyperplanes; the honest scope is a
  local tangent estimate, and the error from curvature over the scan window
  needs bounding.
- **Noise model dependence.** The signal-to-noise analysis assumes additive
  independent noise; correlated noise changes the plateau argument.
- **Baseline strength.** Classical TV deconvolution with a good solver is
  strong. If G1 fails, report it and narrow the claim to the certified-enclosure
  contribution, which the baseline does not provide at all.
- **Efficiency gap.** The scan estimator sits about an order of magnitude above
  the Cramér-Rao bound on the worked example, because it is deliberately local.
  Where a global parametric model is trustworthy, fitting it directly will beat
  the scan, and the spec should say so rather than compete on ground it loses.
- **Falsifier.** If the `alpha^(n - 5/2)` scaling (G7) does not appear, the
  noise analysis in section 5 is wrong and the design rule must be rederived
  before it is documented anywhere. **G7 passed for the locally-seeded
  estimator**; the design rule "use the smallest `alpha` the problem allows"
  is licensed for the logistic scan on this synthetic family. Global
  (unseeded) peak search remains a recorded limitation for `n=4`.

## 12. Implementation checklist

- [ ] `packages/omnibias-pinn/src/omnibias/pinn/inverse/_core.py`
- [ ] Scan-based `locate_interface` with the order channel reported
- [ ] Krawczyk enclosure wired from spec 03-08, with the `1000`-instance
      coverage test
- [ ] Transfer-matrix inversion reusing spec 02-11's analytic Jacobian
- [ ] Free-boundary tracking on the equality locus with exact front velocity
- [ ] Fisher identifiability report before inversion
- [ ] Sensor placement through `omnibias.submodular` continuous greedy
- [ ] Jump-order confusion matrix, not just accuracy
- [x] `alpha` sweep fitting the exponent against the `alpha^(n - 5/2)` prediction
      (G7)
- [x] Cramér-Rao comparison reported alongside the scan estimator (honesty note
      in the G7 artifact; full numerical CR table remains open)
- [ ] Guard preventing the enclosure and the conformal bound from merging
- [x] `benchmarks/inverse_imaging.py` plus smoke JSON (alpha-sweep family only)
- [ ] Docs page and nav entry
- [x] Index row in `theory/README.md`
