# 03-07 Scale flow and coarse-graining

## 1. Thesis and status

The tempered scale `alpha` is a renormalization-group parameter: coarse-graining
a field means integrating out packs above a cutoff, and because the tower has an
**exact scaling law** the flow of the effective parameters is computable rather
than fitted.

- **Status**: concept
- **Depends on**: 01-07, 02-07
- **Blocks**: 03-13, 05-02

## 2. Where it lands

`packages/omnibias-core/src/omnibias/core/scale.py` for the flow algebra, with
consumers in `omnibias.fields` (coarse-grained operators) and
`omnibias.symbolic` (scale-resolved discovery).

## 3. Prior art in omnibias

- `omnibias.core.spec` — the `tempered` combinator with the exact law
  `sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)`.
- `FBPINNField` — finite-basis PINN domain decomposition with overlapping
  windows, the closest thing to a multiscale construction in the repo.
- `benchmarks/spectral_bias_fbpinn.py` — the spectral-bias benchmark, with
  per-arm `wall_seconds` and `lstsq_matched` recorded.
- `omnibias.graph` — heat kernel and spectral embedding, which are diffusion
  scale objects.
- Spec 01-07's order-as-frequency analysis — the spectral content of a pack at a
  given scale.

**Confirmed gap.** Multiscale *architecture* exists (FBPINN). There is no scale
*flow*: nothing tracks how effective parameters change as a cutoff moves, and
nothing coarse-grains an operator.

## 4. Mathematics

### The exact scaling law is the starting point

```
sigma_alpha^(n)(u) = alpha^n sigma^(n)( alpha u )
```

This is exact, not asymptotic, and it is what distinguishes this from a generic
multiscale scheme: the relation between a pack at scale `alpha` and the same
pack at scale `lambda alpha` is a known algebraic factor, so "changing scale" is
a closed-form reparameterization rather than a re-fit.

### Coarse-graining

Split a field into slow and fast parts by a cutoff `alpha_c`:

```
u = u_< + u_> ,     u_< = packs with alpha <= alpha_c,   u_> = the rest
```

Coarse-graining means finding the effective dynamics of `u_<` after eliminating
`u_>`. For a **linear** operator this is exact: the fast packs simply drop out
of the slow equations up to their overlap, and the overlap integrals are closed
form (they are inner products of activation derivatives, computable by the
`integral` role).

For a **nonlinear** operator, elimination generates new terms. To one loop, the
correction to a quadratic nonlinearity is a convolution of fast modes, and in
this basis it is a sum of pack products — which are again in the dictionary,
though at shifted orders. So the flow closes approximately at each order, with
the truncation explicit.

State this honestly: **exact for linear operators, an explicit truncation for
nonlinear ones.** That is the same status as any RG scheme, and it is only
useful if the truncation error is reported.

### The flow equations

Write the effective parameters at cutoff `alpha_c` as `theta(alpha_c)`. Then

```
d theta / d log alpha_c = beta_fn( theta )
```

with `beta_fn` computable term by term from the overlap integrals. Fixed points
`beta_fn(theta*) = 0` are scale-invariant configurations, and the eigenvalues of
`d beta_fn / d theta` at a fixed point are critical exponents.

For a field built from a pack dictionary, `beta_fn` is a polynomial in the
parameters with closed-form coefficients. That means fixed points can be found
by root finding on an explicit polynomial system rather than by fitting a flow
from simulations — which is the practical difference from numerical RG.

**The caution that must accompany this**: critical exponents extracted from a
truncated flow are approximations whose error is not controlled by the
truncation order alone. Reporting an exponent to three digits from a one-loop
truncation would be overclaiming. Report the truncation order and a convergence
study across orders, or report nothing.

### Connection to spectral bias

Spec 01-07 says pack order and scale place a channel's sensitivity in a
frequency band. Coarse-graining removes high-frequency packs, so the flow is
literally a flow in frequency content. A network trained with a schedule that
lowers `alpha_c` over time learns coarse structure first — which is the standard
cure for spectral bias, here with a principled schedule rather than a
hand-tuned one.

That connection is testable against the existing spectral-bias benchmark, and it
is the most concrete deliverable in this spec.

### Multigrid

The classical multigrid V-cycle is: smooth on a fine grid, restrict to a coarse
grid, solve, prolong, correct. With scale flow, restriction and prolongation are
the exact scaling law, so a **grid-free multigrid** is available: the "grids" are
scale bands and the transfer operators are exact.

## 5. Worked example

**Exact rescaling, verified.** A pack at `alpha = 1`, order `n = 2`, evaluated
at `u = 0.4`, versus the same pack at `alpha = 2` evaluated at `u = 0.2`.

With `sigma = tanh`, `t = tanh(0.4) = 0.3799490`:

```
sigma'(0.4)  = 1 - t^2                =  0.8556388
sigma''(0.4) = -2 t (1 - t^2)         = -0.6501983
```

The scaling law says `sigma_2''(0.2) = 2^2 sigma''(2 * 0.2) = 4 sigma''(0.4)`:

```
4 * (-0.6501983) = -2.6007932
```

Direct: `sigma_2(u) = tanh(2u)`, so `sigma_2''(u) = 4 tanh''(2u)`, at `u = 0.2`
that is `4 tanh''(0.4) = -2.6007932`. Identical, as an exact law must be. No
approximation, no fitting.

**A linear coarse-graining, worked.** Take a field
`u = c_1 p_1 + c_2 p_2` with `p_1` at `alpha = 1` and `p_2` at `alpha = 8`
(a slow and a fast pack), both order 1, both centred at 0, and the linear
operator `L = d^2/dx^2`.

`L u = c_1 p_1'' + c_2 p_2''`, and by the scaling law `p_2'' = 64 * (the unit
pack's second derivative at 8x)`. Projecting `L u` back onto `p_1` requires the
overlap `< p_1, p_2'' >`, which is an integral of a product of activation
derivatives at different scales — closed form via the `integral` role.

The magnitude of that overlap is the quantitative question: if it is small, the
fast pack decouples and coarse-graining is nearly free; if it is not, the
effective slow operator is genuinely modified. For scale ratio `8` and matched
centres, the overlap is suppressed by roughly the ratio of the scales, so the
correction to the effective operator is a few percent — small but not
negligible, which is precisely why computing it rather than assuming it is
worthwhile.

**Spectral-bias schedule.** Spec 01-07 gives the peak frequency of an order-`n`
pack at scale `alpha`. A schedule that raises `alpha_c` so that the represented
band grows linearly in training time is a concrete, non-tuned curriculum, and
comparing it against the existing FBPINN arm is a direct test.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/scale.py
@dataclass(frozen=True)
class ScaleBand:
    alpha_lo: float
    alpha_hi: float

def rescale_pack(pack: PackSpec, factor: float) -> PackSpec:
    """Exact: applies the alpha^n law. Bit-exact round trip required."""

def overlap(p: PackSpec, q: PackSpec, *, derivative_order: int = 0) -> float:
    """Closed-form inner product of two packs at different scales."""

def coarse_grain_linear(operator, packs, *, cutoff: float) -> EffectiveOperator:
    """Exact for linear operators."""

def flow_coefficients(dictionary, nonlinearity, *, order: int) -> FlowSystem:
    """Truncated beta function. `order` is recorded in the result and must be
    reported with any exponent."""

@dataclass
class FlowSystem:
    coefficients: Mapping[str, float]
    truncation_order: int
    def fixed_points(self) -> tuple[Mapping[str, float], ...]: ...
    def exponents(self, fp) -> tuple[float, ...]: ...
```

```python
# omnibias/fields/scale.py
def scale_schedule(*, target_band: Callable[[float], float], steps: int) -> Sequence[float]:
    """Curriculum from the order-as-frequency analysis, not hand-tuned."""
def grid_free_vcycle(field, residual, *, bands: Sequence[ScaleBand]) -> Tensor: ...
```

## 7. Practical use cases

1. **Spectral-bias curricula** with a derived rather than tuned schedule,
   measured against the existing FBPINN benchmark.
2. **Grid-free multigrid** for meshless PDE solvers, where classical multigrid
   needs a hierarchy of meshes.
3. **Effective models.** Deriving a coarse model from a fine one with a stated
   truncation, for turbulence-adjacent and homogenization problems.
4. **Scale-resolved discovery.** Feeding `omnibias.symbolic` a field restricted
   to a scale band, so discovered laws are labelled by the scale they describe.
5. **Adaptive refinement** (spec 03-13) driven by where the flow says structure
   is being generated.

## 8. Acceptance gates

Baselines: FBPINN with hand-tuned windows, a fixed-scale network, and (for the
multigrid arm) single-level iteration.

- **G1 exact rescaling.** `rescale_pack` round-trips bit-exactly and satisfies
  the scaling law to `<= 4 ulp` for orders `0 .. 10` and factors spanning three
  decades.
- **G2 overlap correctness.** Closed-form overlaps match high-precision
  numerical integration to `<= 1e-12` relative.
- **G3 linear exactness.** For linear operators, the coarse-grained operator
  reproduces the fine operator's action on the slow subspace exactly
  (`<= 1e-12`).
- **G4 curriculum win.** The derived schedule beats the hand-tuned FBPINN
  baseline on the existing spectral-bias benchmark, or matches it without
  tuning; the artifact records which, and "matches without tuning" is an
  acceptable pass because removing a tuning burden is the claim.
- **G5 multigrid convergence.** The grid-free V-cycle reduces the residual by at
  least `5x` per cycle on a linear problem, versus single-level iteration.
- **G6 truncation honesty.** Any reported critical exponent is accompanied by a
  convergence study across at least three truncation orders; a test asserts that
  the exponent API refuses to return a value without the order recorded.

## 9. Benchmark plan

- `benchmarks/scale_flow.py`: rescaling exactness, overlap validation,
  curriculum comparison on the existing spectral-bias problem, V-cycle
  convergence, truncation convergence study.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/scaleflow/`.

## 10. Honesty and scope

- `alpha` is a **tempering scale**, and `alpha_c -> inf` is neither collapse. The
  founding bias collapse is `delta -> 0` (biases coalescing into `sigma^(K-1)`);
  temperature collapse is `beta -> inf` (gates hardening). Scale flow is a third
  axis and must not be described with either word.
- **Coarse-graining is exact for linear operators and truncated for nonlinear
  ones.** The truncation order is a required field of `FlowSystem`, and no
  exponent may be reported without it.
- Renormalization-group ideas are borrowed, not invented here. The contribution
  is that the exact scaling law makes the transfer operators closed form, so the
  flow coefficients are computed rather than fitted.
- **Critical exponents from truncated flows are approximations** whose error is
  not controlled by the order alone. G6 exists to prevent a three-digit exponent
  from a one-loop calculation.
- No certificate tier.

## 11. Open questions and risks

- **Does the flow close?** Nonlinear elimination generates terms outside the
  dictionary. If closure requires an unbounded set of new terms, the truncation
  is uncontrolled and the honest outcome is to restrict to linear
  coarse-graining.
- **Overlap suppression.** The whole scheme is attractive only if scales
  genuinely decouple. Measure the overlap decay with scale ratio before building
  on it.
- **Schedule sensitivity.** A derived curriculum may still need a rate
  constant, in which case the tuning burden has moved rather than vanished. Say
  which parameters remain.
- **Falsifier.** If the derived curriculum performs worse than the hand-tuned
  FBPINN baseline, and the overlaps do not decouple, this reduces to a
  reparameterization identity with no operational value.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/scale.py`
- [ ] `packages/omnibias-fields/src/omnibias/fields/scale.py`
- [ ] Reuse the `tempered` combinator; do not reimplement the scaling law
- [ ] Bit-exact rescaling round-trip test
- [ ] Overlap validation against high-precision integration
- [ ] Linear coarse-graining exactness test
- [ ] `FlowSystem` refuses to report exponents without a truncation order
- [ ] Curriculum arm added to the existing spectral-bias benchmark
- [ ] `benchmarks/scale_flow.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
