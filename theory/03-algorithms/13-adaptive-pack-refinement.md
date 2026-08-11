# 03-13 Adaptive pack refinement

## 1. Thesis and status

Structural adaptation with three moves — **birth** (a new pack where the
residual demands it), **growth** (raise a pack's order), and **death** (prune a
pack that stopped earning its parameters) — each of which preserves the current
fit exactly, so refinement can never undo learning.

- **Status**: designed
- **Depends on**: 01-01, 03-01, 03-06, 03-07, 03-10, 03-12
- **Blocks**: 02-03, 05-01

## 2. Where it lands

`packages/omnibias-torch/src/omnibias/torch/refine.py` and the jax twin, beside
`growable.py` which already implements one of the three moves.

## 3. Prior art in omnibias

- `packages/omnibias-torch/src/omnibias/torch/growable.py` —
  `GrowableOperatorMultiBiasUnit`: grows the bias arity `K` of a single pack
  during training. **Growth already exists** for one axis.
- `FBPINNField` — fixed overlapping windows; a static decomposition that
  adaptive refinement would replace.
- `omnibias.core.spec` — the `tempered` combinator and its exact scaling law,
  which is how a new pack's scale is chosen.
- Spec 03-10's singularity tracking and spec 03-07's scale flow — two principled
  refinement indicators.
- `omnibias.difference` — certified truncation, for bounding what a refinement
  can gain.

**Confirmed gap.** Growth exists on one axis (`K` within a pack). There is no
birth (adding packs), no death (pruning), and no indicator-driven policy.

## 4. Mathematics

### The zero-perturbation principle

The single most important design property: **every structural move must leave
the function unchanged at the moment it happens.**

- **Birth**: a new pack enters with outer weight `c = 0`. The function is
  literally unchanged; only the gradient landscape gains a direction.
- **Growth**: raising `K` from `K` to `K + 1` in a `GrowableOMBU` must
  initialize so the represented function is identical, which is what the
  existing implementation does and what a new one must preserve.
- **Death**: a pack is removed only when its contribution is below a threshold,
  and the resulting change is bounded by that threshold, which is *reported*
  rather than assumed negligible.

Birth and growth are exactly zero-perturbation. Death is bounded-perturbation.
That asymmetry is real and should be reflected in the API: death returns the
perturbation it caused.

### Refinement indicators

Where to refine is the whole question. Four indicators, in increasing order of
principle:

1. **Residual magnitude.** The classical choice. Cheap, and it conflates "hard
   here" with "under-resolved here".
2. **Local error estimate.** Compare the residual against a certified truncation
   bound (spec 03-06's Peano route, or `omnibias.difference`). Refine where the
   bound says resolution is the binding constraint.
3. **Singularity proximity** (spec 03-10). The distance to the nearest complex
   singularity is a resolution requirement: features of size `|Im x_s|` need
   packs of scale `1/|Im x_s|`. This turns refinement into a computation.
4. **Scale-flow generation** (spec 03-07). Refine where the flow says structure
   is being generated at the cutoff.

Indicators 3 and 4 are the interesting ones because they say *what scale* to
add, not merely *where* — which is the information a classical `h`-refinement
indicator does not carry.

### What to add

Given a location and a scale, the choice is between:

- **`h`-type**: a new pack at a smaller scale (higher `alpha`), resolving finer
  features.
- **`p`-type**: raise the order of an existing pack, increasing local
  approximation order.

Classical `hp`-adaptivity says: use `p` where the solution is smooth and `h`
where it is not. The smoothness indicator here is the **coefficient decay of the
local jet** — if the Taylor coefficients decay geometrically, the solution is
analytic locally and `p` wins; if they decay slowly, `h` wins. That decay is
exactly the quantity spec 03-10 computes, so the `hp` decision is made from the
same data as the singularity tracking.

### Death and the parameter budget

Without pruning, refinement only grows. A death criterion based on contribution

```
contribution(g) = || c_g p_g ||  /  || u ||
```

with a threshold, plus a minimum age (so a newly born pack is not killed before
its weight has moved off zero), keeps the model bounded. The perturbation caused
is `<= contribution(g)` in the same norm, which is reportable.

### Convergence

A refinement loop should have a stopping criterion tied to the error, not to a
step count. With a certified error bound available (spec 03-06, 03-08), the loop
can stop when the bound meets a target. Without one, it must stop on a plateau
and say so.

## 5. Worked example

**A boundary layer, refined by indicator.**

Solve `epsilon u'' + u' = 0` on `[0, 1]` with `u(0) = 0`, `u(1) = 1` and
`epsilon = 0.01`. The exact solution is

```
u(x) = ( 1 - exp(-x/epsilon) ) / ( 1 - exp(-1/epsilon) )
```

which is essentially `1 - exp(-100 x)`: a boundary layer of width `0.01` at
`x = 0` and flat elsewhere.

**Initial model**: two packs at scale `alpha = 2`, centred at `0.25` and `0.75`.
The layer is completely unresolved; the residual near `x = 0` is `O(1)`.

**Indicator 3, worked.** Compute the local jet of the current approximation near
`x = 0` and read the coefficient decay. For the true solution, the Taylor
coefficients of `exp(-100x)` about `x = 0.005` are

```
a_k = (-100)^k exp(-0.5) / k!
```

so `|a_k / a_{k-1}| = 100 / k`. The Domb-Sykes intercept as `1/k -> 0` is `0`,
meaning the singularity is at infinity — the function is entire, so `p`-type
refinement is indicated. But the *ratios* are large for small `k`
(`100, 50, 33.3, 25, ...`), which says the local scale is `1/100`: the pack scale
must reach `alpha ~ 100` before high order helps.

So the correct refinement decision is: first `h`-type, raising the scale from
`2` to `100` near the layer, then `p`-type once the scale matches. That
two-phase answer is precisely what a naive residual indicator would not give —
residual magnitude says "refine here" but not "the scale is wrong by a factor of
50".

**Birth, concretely.** Add a pack at `mu = 0.005`, `alpha = 100`, order 1, with
`c = 0`. The model output is unchanged at every point (the new term is
identically zero). After a few gradient steps `c` moves off zero and the layer
begins to resolve. Asserting the zero-perturbation property is a one-line test:
the outputs before and after birth must be bit-identical.

**Death, concretely.** After refinement, the original pack at `alpha = 2`,
`mu = 0.25` may have `contribution = 0.003`. Removing it perturbs the solution by
at most `0.3` percent in the relevant norm, and that number is what the API
returns — so the caller decides whether it is acceptable, rather than the library
deciding silently.

## 6. Proposed API

Extends `growable.py` rather than replacing it.

```python
# omnibias/torch/refine.py  (and jax twin)
class Indicator(StrEnum):
    RESIDUAL = "residual"
    CERTIFIED_ERROR = "certified_error"
    SINGULARITY = "singularity"        # spec 03-10
    SCALE_FLOW = "scale_flow"          # spec 03-07

@dataclass(frozen=True)
class RefinePolicy:
    indicator: Indicator
    birth_threshold: float
    death_threshold: float
    min_age: int = 100                 # steps before a pack may die
    max_packs: int | None = None
    hp_rule: Literal["coefficient_decay", "always_h", "always_p"] = "coefficient_decay"

@dataclass
class RefineReport:
    born: tuple[PackSpec, ...]
    grown: tuple[int, ...]
    died: tuple[int, ...]
    death_perturbation: float          # bounded, reported, never assumed zero
    indicator_values: FloatArray

def refine(model, residual_fn, policy: RefinePolicy, *, step: int) -> RefineReport: ...

def assert_zero_perturbation(model, before, after) -> None:
    """Birth and growth must be bit-identical. Called in tests and, optionally,
    at runtime in a debug mode."""
```

## 7. Practical use cases

1. **Boundary layers and shocks**, where uniform resolution is wasteful and the
   right scale is not known in advance.
2. **Replacing FBPINN's fixed windows** with an adaptive decomposition,
   comparable on the existing benchmark.
3. **Jet-KAN refinement** (spec 02-03), where both `h` and `p` axes exist
   naturally on each edge.
4. **Long-time integration**, where structure develops and the model must follow
   it, with singularity proximity as the trigger.
5. **Parameter-efficient scientific models**: death keeps the budget bounded, so
   the model size tracks the problem's complexity rather than growing
   monotonically.

## 8. Acceptance gates

Baselines: a fixed-size model with the same final parameter count, uniform
refinement, and `FBPINNField` with hand-tuned windows.

- **G1 zero perturbation.** Birth and growth leave the model output
  **bit-identical**, asserted on every move in a long randomized run. Not
  "within tolerance": bit-identical.
- **G2 death accounting.** The reported `death_perturbation` upper-bounds the
  measured output change, with zero violations.
- **G3 indicator quality.** On the boundary-layer problem, the singularity and
  scale-flow indicators place packs at the correct scale within a factor of `2`,
  while the residual indicator does not (measured, so the claim in section 5 is
  tested rather than asserted).
- **G4 efficiency win.** At matched final parameter count, adaptive refinement
  reaches at least `10x` lower error than a fixed model with the same budget, on
  a suite with localized features, over five seeds.
- **G5 budget stability.** With death enabled, the parameter count converges
  rather than growing without bound over a long run.
- **G6 parity.** torch and jax bit-identical, including the refinement decisions.

## 9. Benchmark plan

- `benchmarks/adaptive_refinement.py`: zero-perturbation assertions, indicator
  comparison on the boundary-layer and shock problems, efficiency against fixed
  and uniform baselines, budget-stability run, and an FBPINN comparison on the
  existing spectral-bias problem.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/refine/`.

## 10. Honesty and scope

- Packs collapse by the founding bias collapse (`delta -> 0`). Refinement adds
  and removes packs; it is not a collapse operation of either kind, and
  `alpha` growth is the tempering scale, a third axis.
- **Death perturbs the model.** Only birth and growth are exactly
  zero-perturbation. The API returns the perturbation so it is never invisible.
- `hp`-adaptivity is classical finite-element technology. The contributions are
  the zero-perturbation guarantee (which mesh refinement cannot offer, since
  re-meshing changes the space), and the principled scale indicators from the
  jet.
- **Refinement decisions are discrete**, so training is not a smooth
  optimization any more. Convergence guarantees from smooth optimization do not
  transfer, and the loop's stopping criterion must be stated.
- No certificate tier by itself; where a certified error bound is available it
  drives the stopping criterion, and that should be reported.

## 11. Open questions and risks

- **Indicator cost.** Singularity tracking needs a high-order jet at many
  locations. If the indicator costs more than the refinement saves, use a
  cheaper one; measure.
- **Oscillation.** Birth and death can fight, adding and removing the same pack.
  The `min_age` guard helps; hysteresis in the thresholds is likely also needed.
- **Optimizer state.** Adding parameters mid-training invalidates optimizer
  moments. The existing `GrowableOMBU` faces this already; whatever it does
  should be followed and documented rather than reinvented.
- **Falsifier.** If a fixed model at the same final parameter count matches the
  adaptive one (G4 fails), adaptivity is not paying for its complexity, and the
  right answer is to size the model well and leave it alone.

## 12. Implementation checklist

- [ ] `packages/omnibias-torch/src/omnibias/torch/refine.py`
- [ ] `packages/omnibias-jax/src/omnibias/jax/refine.py`
- [ ] Reuse `GrowableOperatorMultiBiasUnit` for the growth move; do not
      reimplement it
- [ ] Bit-identical zero-perturbation assertions for birth and growth
- [ ] `death_perturbation` bound with a soundness test
- [ ] Indicator comparison including the negative result for plain residual
- [ ] `min_age` and hysteresis to prevent oscillation, with a long-run test
- [ ] Documented optimizer-state policy on parameter addition
- [ ] `benchmarks/adaptive_refinement.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
