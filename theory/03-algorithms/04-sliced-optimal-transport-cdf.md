# 03-04 Sliced optimal transport with closed-form CDFs

## 1. Thesis and status

One-dimensional optimal transport is a quantile comparison, and a mixture of
tempered activations has a **closed-form CDF and closed-form quantile
derivatives**, so sliced Wasserstein distances between such mixtures are exact
rather than sample-estimated.

- **Status**: designed
- **Depends on**: 01-02, 01-05
- **Blocks**: 05-02

## 2. Where it lands

`packages/omnibias-measure/src/omnibias/measure/transport/` — the measure
package already owns pushforwards and integration; transport is its natural
neighbour.

## 3. Prior art in omnibias

- `packages/omnibias-measure/` — the `Measure` abstraction (pushforward,
  product, importance reweighting), the measure integral `int f dmu`, and
  layer-cake / simple-function primitives, with trainable torch and jax layers.
- `docs/operator-surface.md` — the `integral` role: an antiderivative window
  `S(z + b_hi) - S(z + b_lo)` with `S' = sigma`. A CDF is exactly an
  antiderivative.
- `omnibias.graph` — Gumbel-Sinkhorn and SoftSort, the differentiable
  permutation and sorting machinery.
- `omnibias.combinatorics` — entropic relaxations onto integral polytopes with
  LP-dual gap certificates, which includes matching.

**Confirmed gap.** No Wasserstein or transport distance of any kind. Sorting and
permutation relaxations exist; distances between measures do not.

## 4. Mathematics

### The 1-D reduction

For measures `mu, nu` on `R` with CDFs `F, G`, the `p`-Wasserstein distance is

```
W_p^p(mu, nu) = integral_0^1 | F^{-1}(q) - G^{-1}(q) |^p dq
```

and for `p = 1` it simplifies further to

```
W_1(mu, nu) = integral_R | F(x) - G(x) | dx
```

Both are one-dimensional integrals of CDF differences. **No optimization at
all** — that is why 1-D transport is the base case for everything sliced.

### The closed-form CDF

A density built as a normalized mixture of tempered activation derivatives,

```
rho(x) = sum_g c_g alpha_g sigma'( alpha_g (x - mu_g) ),   c_g >= 0, sum c_g = 1
```

has CDF

```
F(x) = sum_g c_g sigma( alpha_g (x - mu_g) )
```

exactly, because `sigma` is the antiderivative of `sigma'`. So **the CDF is a
mixture of the base activations themselves**, evaluated at the same points as
the density. This is the whole trick, and it is why omnibias is a natural home
for this: the `integral` role already provides antiderivatives, and here the
antiderivative of the density is a function the library computes anyway.

For the logistic base this is a mixture of logistic CDFs; for `tanh` it is a
shifted and scaled version of the same; for the gaussian base it is a mixture of
error functions, which is the classical gaussian mixture CDF.

### Quantiles

`F^{-1}` has no closed form for a mixture (it needs a root find), but its
*derivative* does:

```
d F^{-1} / d q = 1 / rho( F^{-1}(q) )
```

and the derivative with respect to a parameter `theta` follows from implicit
differentiation:

```
d F^{-1}(q) / d theta = - ( dF/dtheta ) / rho     evaluated at x = F^{-1}(q)
```

Both numerator and denominator are closed form. So a Newton solve for the
quantile converges quadratically with an exact derivative, and the gradient of
`W_p` with respect to mixture parameters is exact — no sampling noise, no
Sinkhorn iterations, no envelope-theorem approximation.

### The `W_1` route avoids inversion entirely

Since `W_1 = integral |F - G| dx`, and `F - G` is a difference of activation
mixtures, the integral splits at the sign changes of `F - G`. Between sign
changes, `integral (F - G) dx` needs the antiderivative of `sigma`, which for
`tanh` is `log cosh` and for the logistic is the softplus — both closed form.

So `W_1` between two activation mixtures is **exactly computable** given the
sign-change locations, which are roots of `F - G` found by the same Newton
machinery. That is a genuinely exact 1-D Wasserstein distance between
parametric families.

### Slicing

For measures on `R^D`, the sliced Wasserstein distance averages the 1-D distance
over projections:

```
SW_p^p(mu, nu) = E_{w ~ S^{D-1}} [ W_p^p( w_# mu, w_# nu ) ]
```

If `mu` is a mixture of *isotropic* tempered packs, the pushforward `w_# mu` is
again a 1-D mixture with the same weights and projected means, so **the
projection is closed form** too. That is the second reason this fits: the family
is closed under projection.

The projection average is estimated by sampling directions, which is the one
remaining source of sampling error, and its variance is `O(1/L)` for `L`
directions with a known constant. Say so; do not describe the whole method as
sample-free.

## 5. Worked example

Two logistic mixtures on `R`:

```
F(x) = 0.5 sigma(x + 1) + 0.5 sigma(x - 1)          (two bumps at -1 and +1)
G(x) = sigma(x)                                      (one bump at 0)
```

both with unit scale, `sigma` the logistic.

`W_1 = integral |F - G| dx`. By symmetry `F(-x) = 1 - F(x)` and
`G(-x) = 1 - G(x)`, so `F - G` is odd about `x = 0` up to sign, and the integral
is twice the value over `[0, inf)`.

At `x = 0`: `F(0) = 0.5(sigma(1) + sigma(-1)) = 0.5(0.7310586 + 0.2689414) = 0.5`
and `G(0) = 0.5`, so `F - G = 0` at the origin: the only sign change is there.

For `x > 0`, `F(x) < G(x)` (the mass is spread wider), so

```
W_1 = 2 integral_0^inf ( G(x) - F(x) ) dx
```

Using `integral_0^inf ( sigma(x - a) - [x > 0] ... )` is awkward directly; the
clean route is the antiderivative of `sigma(x - a) - 1`, which is
`-softplus(a - x)`. Since `sigma(x) - 1 = -sigma(-x)` and
`integral sigma(-x) dx = -softplus(-x)`:

```
integral_0^R ( G - F ) dx
 = integral_0^R [ sigma(x) - 0.5 sigma(x+1) - 0.5 sigma(x-1) ] dx
 = [ softplus(x) - 0.5 softplus(x+1) - 0.5 softplus(x-1) ]_0^R
```

As `R -> inf`, `softplus(R) - 0.5 softplus(R+1) - 0.5 softplus(R-1) -> R - 0.5(R+1) - 0.5(R-1) = 0`,
so the upper limit contributes nothing. At `x = 0`:

```
softplus(0)  = log 2      = 0.6931472
softplus(1)  = log(1+e)   = 1.3132617
softplus(-1) = log(1+1/e) = 0.3132617
value at 0 = 0.6931472 - 0.5(1.3132617) - 0.5(0.3132617)
           = 0.6931472 - 0.6566309 - 0.1566309 = -0.1201146
```

so `integral_0^inf (G - F) dx = 0 - (-0.1201146) = 0.1201146` and

```
W_1 = 2 * 0.1201146 = 0.2402292
```

**Exact, in closed form, with three `softplus` evaluations.** A Monte Carlo
estimate with 10 000 samples per measure would give roughly two digits with
sampling noise; this gives all of them, and its gradient with respect to the
bump separation is likewise exact.

Sanity check on the magnitude: the two mixtures differ by spreading half the
mass by `+-1`, so a distance of about `0.24` is the right order — a full
displacement of half the mass by distance `1` would be `0.5`, and the logistic
tails overlap substantially.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/measure/transport/_core.py
@dataclass(frozen=True)
class ActivationMixture:
    weights: FloatArray        # c_g, non-negative, summing to 1
    means: FloatArray          # mu_g
    scales: FloatArray         # alpha_g
    base: str = "logistic"
    def cdf(self, x): ...      # closed form: mixture of sigma
    def pdf(self, x): ...      # closed form: mixture of alpha sigma'
    def project(self, w): ...  # closed form for isotropic packs

def w1_exact(mu: ActivationMixture, nu: ActivationMixture) -> float:
    """Exact 1-D W_1 via sign-change roots plus closed-form antiderivatives."""
def wp_quantile(mu, nu, *, p: int, quadrature: QuadratureSpec) -> float:
    """W_p for p > 1; quadrature in q, exact quantile derivatives."""
def sliced_wasserstein(mu, nu, *, p: int, directions: int, key) -> SlicedResult: ...

@dataclass
class SlicedResult:
    value: float
    direction_stderr: float     # the one remaining sampling error, always reported
```

Reporting `direction_stderr` as a required field keeps the honest statement
("exact per slice, sampled over slices") visible in the output.

## 7. Practical use cases

1. **Generative model training** with an exact 1-D transport loss, removing
   sampling noise from the objective.
2. **Distribution matching in inverse problems**, where the target is a measured
   histogram and the model is a parametric mixture.
3. **Domain adaptation** with a differentiable, exactly computed discrepancy.
4. **Barycenters.** Sliced Wasserstein barycenters of activation mixtures stay
   in the family, so the barycentre is again a mixture.
5. **Measure-valued PINNs** in the score / Fokker-Planck setting, where the
   evolving density is naturally a mixture.

## 8. Acceptance gates

Baselines: Monte Carlo sliced Wasserstein with matched compute, and
Sinkhorn-based entropic OT.

- **G1 exactness.** `w1_exact` matches a high-precision numerical integral to
  `<= 1e-12` relative on a randomized suite of mixture pairs, including cases
  with several sign changes.
- **G2 gradient exactness.** Parameter gradients match high-precision
  finite differences to `<= 1e-8` relative, including through the quantile
  inversion.
- **G3 variance win.** At matched compute, the closed-form slice estimator has
  at least `100x` lower variance than the Monte Carlo estimator on the same
  problem, with the residual variance attributable solely to direction sampling.
- **G4 metric properties.** Symmetry and the triangle inequality hold to
  `<= 1e-10` on randomized triples (a real test: an implementation error usually
  breaks one of them).
- **G5 honest reporting.** `direction_stderr` is always populated and matches an
  independent bootstrap estimate.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/sliced_ot.py`: exactness, gradient checks, variance comparison,
  metric-property checks, and a generative-fitting task.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/slicedot/`.

## 10. Honesty and scope

- The mixture components come from the founding bias collapse (`delta -> 0`).
  No temperature collapse appears anywhere in this spec.
- **Exact per slice, sampled over slices.** The 1-D distances are exact; the
  average over projection directions is a Monte Carlo estimate with `O(1/L)`
  variance. Never describe the method as sample-free.
- Exactness requires **both** measures to be activation mixtures. Against
  empirical data the CDF is a step function and the distance is exact against
  *that* step function, which is a different (and still useful) statement.
- Sliced Wasserstein is a metric but is **not** the Wasserstein distance; it
  lower-bounds it and is equivalent only up to dimension-dependent constants.
  Say which one is being reported.
- No certificate tier. Interval enclosures of the closed-form expressions are
  possible and would be a natural extension.

## 11. Open questions and risks

- **Sign-change root finding** must be robust: nearly tangent CDFs produce
  clustered roots. A certified root count (via spec 01-09's Krawczyk machinery)
  would remove the risk and is worth considering.
- **Non-isotropic packs** do not project into the family, so the closed-form
  projection is lost. Either restrict to isotropic packs or state the fallback.
- **Direction sampling in high `D`** needs many slices; the variance constant
  grows with dimension and must be measured rather than assumed benign.
- **Falsifier.** If Monte Carlo sliced Wasserstein at matched compute reaches
  the same task accuracy, the exactness is not worth the restriction to
  mixtures.

## 12. Implementation checklist

- [ ] `packages/omnibias-measure/src/omnibias/measure/transport/_core.py`
- [ ] torch and jax twins with a parity test
- [ ] Reuse the `Measure` abstraction and the `integral` role antiderivatives
- [ ] Exactness test against high-precision integration, multi-root cases
- [ ] Gradient test through quantile inversion
- [ ] Metric-property tests (symmetry, triangle inequality)
- [ ] Robust sign-change root finder with a certified-count option
- [ ] `direction_stderr` always populated, bootstrap-checked
- [ ] `benchmarks/sliced_ot.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
