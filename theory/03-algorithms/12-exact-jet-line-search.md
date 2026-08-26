# 03-12 Exact jet line search

## 1. Thesis and status

A directional jet of the loss gives the exact Taylor polynomial along a search
direction in **one** forward pass, so the line-search subproblem becomes root
finding on a known polynomial instead of a sequence of trial evaluations.

- **Status**: gated (G1/G2/G3/G6 CI; G4 step-count **leftover-recorded** / unearned vs strong Wolfe, leftover #47 (`1.83x`, need `2x`), not in CI `all_passed`; G5 order×depth crossover **leftover-recorded**, leftover #48, not in CI `all_passed`)
- **Depends on**: 01-01
- **Blocks**: 03-01, 03-13

## 2. Where it lands

`omnibias.core.line_search` (shared algebra) plus
`omnibias.{torch,jax}.line_search` twins, re-exported from
`omnibias.{torch,jax}.optim`. `optim` is a module, not a package, so the
twins sit beside it rather than at `optim/line_search.py`.

## 3. Prior art in omnibias

- `packages/omnibias-torch/src/omnibias/torch/optim/` — `CubicNewton`,
  `GaussNewton`, `KFAC`, `TrustRegionNewtonCG`, `JetSubspaceTensor`,
  `NaturalGradient`.
- `omnibias.{torch,jax}.jet` — `mlp_jet`, `compose_jet`, `derivative_jet`,
  `tower_to_jet` / `jet_to_tower`: exact directional jets through depth.
- `omnibias-curvature` — closed-form Hessian, Fisher and KFAC.
- `omnibias.difference` — certified truncation and Padé remainders, for bounding
  the jet's own truncation error.

**Closed.** `omnibias.core.line_search` plus the torch/jax twins are the
directional line-search model. `JetSubspaceTensor` remains the subspace
model; `taylor_line_min` remains the order-2/3 helper without a certified
radius.

## 4. Mathematics

### The subproblem

Given parameters `theta` and a direction `d`, line search minimizes

```
phi(s) = L( theta + s d )
```

over the step `s`. Classical methods (Armijo backtracking, Wolfe conditions,
cubic interpolation) evaluate `phi` and `phi'` at a few points and fit a low
model.

### What a jet gives

`mlp_jet` computes the directional Taylor jet of the network output, and by
chaining with the loss's own tower,

```
phi(s) = sum_{k=0}^{N} ( phi^{(k)}(0) / k! ) s^k  +  R_N(s)
```

with **all** coefficients `phi^{(k)}(0)` obtained in a single pass. So instead of
fitting a cubic from four numbers, one has the exact degree-`N` polynomial.

Then:

- The stationary points are the roots of `phi'`, a degree-`N-1` polynomial:
  solve exactly (companion matrix, or a certified root isolation).
- The minimizer among them is chosen by `phi''` sign and `phi` value — all
  polynomial evaluations.
- Wolfe conditions are polynomial inequalities, checkable exactly on intervals
  rather than tested at sample points.

### Cost accounting, honestly

A jet of order `N` through an `L`-layer network costs roughly `O(L N^2)` from the
Cauchy products, against `O(L)` for a plain forward pass. So an order-6 jet costs
roughly `36x` a forward pass in the Cauchy-product terms, though the constant is
favourable because the activation evaluations are shared.

A backtracking line search typically uses 3 to 10 forward passes plus their
backward passes. So the jet route is competitive when `N` is modest and when the
extra information avoids several trial steps. **The honest claim is a
constant-factor win in a specific regime, not an asymptotic one**, and the
benchmark's job is to find the regime.

### Truncation

`R_N` is not zero. Two responses:

1. **Trust region.** Restrict `s` to a radius where the truncation is
   provably small, using the certified remainder machinery from
   `omnibias.difference`. Inside that radius the polynomial model is a
   guaranteed approximation, which is stronger than any interpolation model.
2. **Verify and fall back.** Evaluate `phi` at the proposed step; if it
   disagrees with the model beyond tolerance, shrink and retry. One extra
   evaluation, and it makes the method safe.

Do both: the trust region for the model's validity, the verification as a
backstop. A line search that can return a worse point is not acceptable in an
optimizer, so the fallback is not optional.

### The interaction with second-order methods

`CubicNewton` builds a cubic model in the *full* space and needs a step-size
rule anyway. Feeding it an exact univariate model along its computed direction
is a natural pairing: the cubic regularization handles the direction, the jet
handles the distance. Likewise `TrustRegionNewtonCG` needs a radius, and the
jet's certified truncation radius is a principled candidate.

## 5. Worked example

**A quartic along the direction, solved exactly.**

Suppose the order-4 jet at `s = 0` gives

```
phi(0)     =  1.0
phi'(0)    = -2.0
phi''(0)   =  6.0
phi'''(0)  = -12.0
phi''''(0) =  48.0
```

so the Taylor polynomial is

```
p(s) = 1 - 2 s + 3 s^2 - 2 s^3 + 2 s^4
```

(dividing by the factorials: `6/2 = 3`, `-12/6 = -2`, `48/24 = 2`).

Stationary points: `p'(s) = -2 + 6 s - 6 s^2 + 8 s^3 = 0`.

Dividing by 2: `4 s^3 - 3 s^2 + 3 s - 1 = 0`. Testing `s = 1/4`:

```
4(0.015625) - 3(0.0625) + 3(0.25) - 1 = 0.0625 - 0.1875 + 0.75 - 1 = -0.375
```

not a root. Testing `s = 0.4`:

```
4(0.064) - 3(0.16) + 1.2 - 1 = 0.256 - 0.48 + 0.2 = -0.024
```

close. Testing `s = 0.42`:

```
4(0.074088) - 3(0.1764) + 1.26 - 1 = 0.296352 - 0.5292 + 0.26 = 0.027152
```

so the root is between `0.4` and `0.42`; linear interpolation gives

```
s* = 0.4 + 0.02 * 0.024 / (0.024 + 0.027152) = 0.4 + 0.02 * 0.46915 = 0.409383
```

Refining once with Newton on `q(s) = 4s^3 - 3s^2 + 3s - 1`,
`q'(s) = 12 s^2 - 6 s + 3`:

```
q(0.409384)  = 0.274436 - 0.502785 + 1.228151 - 1 = -0.000198
q'(0.409384) = 2.011140 - 2.456302 + 3            =  2.554838
s* = 0.409384 + 0.000198 / 2.554838 = 0.409461
```

The cubic's discriminant is `-243 < 0`, so it has exactly one real root: this is
the unique stationary point. And `p''(s*) = 6 - 12 s + 24 s^2 = 5.110 > 0`,
confirming a minimum.

```
p(0.409461) = 1 - 0.818922 + 0.502975 - 0.137295 + 0.056216 = 0.602974
```

So the exact step is `s* = 0.409461`, reducing the model value from `1.0` to
`0.602974`, obtained from **one** jet evaluation plus a cubic root solve.

A backtracking search from `s = 1` with halving and the Armijo condition tries
`p(1) = 2` (reject) then `p(0.5) = 0.625` (accept). It lands on `0.625` after two
trial evaluations. The jet route achieves a decrease of `0.397026` against
backtracking's `0.375`: **5.9 percent more decrease** for one jet instead of two
forward passes. That margin is the honest picture — a real but modest per-step
win, and the benchmark's job is to decide whether it pays for the jet's cost.

## 6. Proposed API

Shipped as `omnibias.core.line_search` / `omnibias.{torch,jax}.line_search`.

```python
# omnibias/torch/line_search.py  (and jax twin)
@dataclass(frozen=True)
class JetLineSearchConfig:
    order: int = 4
    trust_radius: float | Literal["certified"] = "certified"
    verify: bool = True                # extra evaluation backstop; on by default
    wolfe_c1: float = 1e-4
    wolfe_c2: float = 0.9

@dataclass
class LineSearchResult:
    step: float
    model_value: float
    actual_value: float | None         # populated when verify=True
    model_error: float | None
    truncation_radius: float
    fell_back: bool

def jet_line_search(
    closure, params, direction, *, config: JetLineSearchConfig,
) -> LineSearchResult: ...

def polynomial_wolfe(coeffs, *, c1: float, c2: float) -> Interval:
    """The set of steps satisfying Wolfe, as an interval, exactly."""
```

`verify=True` by default and `fell_back` in the result: a line search that can
silently return a worse point is a bug factory.

## 7. Practical use cases

1. **Second-order optimizers** in `omnibias.torch.optim` that currently need a
   step rule. Recommended stack (spec 08-01): `CubicNewton` / `GaussNewton`
   pick the direction, this spec picks the length, spec 08-04 accepts or
   rejects on a unique-zero ball, spec 08-06 may set the cubic `lambda`.
2. **Memetic evolution** (spec 03-01), where the polish step's cost directly
   determines whether the hybrid is worthwhile.
3. **PINN training**, where loss landscapes are badly conditioned and step-size
   choice dominates.
4. **Trust-region radius selection** from the certified truncation radius rather
   than from an adaptive heuristic.
5. **Exact Wolfe checking**, giving a genuinely correct condition test rather
   than one sampled at trial points.

## 8. Acceptance gates

Baselines: Armijo backtracking, strong Wolfe with cubic interpolation, and a
fixed step size.

- **G1 model exactness.** The jet coefficients match high-precision finite
  differences of `phi` to `<= 1e-10` relative for orders `0 .. 6`.
- **G2 truncation radius soundness.** Within the certified radius, the model
  error never exceeds the stated bound, on a dense grid **and** a random sample.
- **G3 never worse.** With `verify=True`, the returned point is never worse than
  the starting point, on 100 percent of a randomized suite including
  deliberately pathological directions. Asserted, not measured on average.
- **G4 step-count win.** On a suite of ill-conditioned problems, the jet line
  search reaches a target loss in at least `2x` fewer *total function
  evaluations* (counting the jet at its true cost) than strong Wolfe, over five
  seeds. **Recorded unearned:** reachable `x^2 + cond y^2` trajectories give
  Wolfe/jet `1.83` (need `2`); the stiffest seed is a Wolfe miss. The previous
  Armijo single-step stub is withdrawn. Not in CI `all_passed`.
- **G5 honest regime.** The benchmark reports the crossover in `N` and in
  network depth beyond which the jet's cost exceeds its benefit, rather than
  only showing the favourable regime. **Reported:** `mlp_jet` vs a named
  four-trial Wolfe budget is favourable at `N=2` and over budget from `N=4`
  (crossover depth `1` at order `4`). Not in CI `all_passed`.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/jet_line_search.py`: coefficient exactness, truncation soundness,
  step-count comparisons on ill-conditioned problems, cost-crossover table.
- Cost accounting counts the jet at its measured wall time, not as one
  evaluation.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/linesearch/`.

## 10. Honesty and scope

- Jets come from the founding bias collapse (`delta -> 0`) tower. No temperature
  collapse appears.
- **The win is a constant factor in a specific regime.** Order-`N` jets cost
  `O(L N^2)`, so there is a depth and order beyond which backtracking is
  cheaper. G5 requires that crossover to be published.
- **The polynomial is a model, not the loss.** Truncation is real, the trust
  radius bounds it, and the verification backstop exists because the bound can
  be conservative or the direction pathological.
- Exact line search on a polynomial model is not new (it is what cubic
  interpolation approximates). The contribution is that the polynomial is exact
  to arbitrary order from one pass, and that Wolfe conditions become exact
  interval statements.
- Certificate tier: sound enclosure for the truncation radius.

## 11. Open questions and risks

- **Jet cost in practice.** The `O(L N^2)` estimate ignores memory traffic,
  which may dominate on accelerators. Measure before believing the model.
- **Root finding robustness.** High-degree polynomials with clustered roots are
  ill-conditioned; certified root isolation (available via the verified
  register) is the safe route and costs more.
- **Direction quality.** An exact line search along a bad direction is still a
  bad step. The value is bounded by the direction-generating method.
- **Falsifier.** If G4 fails — the jet route does not reduce total evaluations
  on ill-conditioned problems — then the exact model is not buying enough, and
  the honest conclusion is that cheap approximate line searches are already
  good enough.

## 12. Implementation checklist

- [x] `packages/omnibias-core/src/omnibias/core/line_search.py` (shared algebra)
- [x] `packages/omnibias-torch/src/omnibias/torch/line_search.py`
- [x] `packages/omnibias-jax/src/omnibias/jax/line_search.py`
- [x] Reuse `mlp_jet` and `compose_jet` on the input-ray path; no new jet arithmetic
- [x] Certified truncation radius via the Lagrange identity in the verified register
      (core cannot import `omnibias.difference`; same remainder as
      `certified_fd_error_general`)
- [x] `verify=True` default with a never-worse assertion test
- [x] Certified root isolation option (`isolate_roots`, interval Newton)
- [x] Cost-crossover table in the benchmark, favourable and unfavourable regimes
- [x] Integration test with `CubicNewton` and `TrustRegionNewtonCG`
- [x] `benchmarks/jet_line_search.py` plus smoke JSON
- [x] Docs page and nav entry
- [x] Index row in `theory/README.md`
