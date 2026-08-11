# 03-01 Soft-population evolution

## 1. Thesis and status

Four evolutionary algorithms that use omnibias structure rather than treating
the model as a black box: a **soft-population `beta`-homotopy** EA whose
selection pressure is a temperature, a **bias-geometry** EA that mutates pack
spread and offsets, a **jet-memetic** hybrid that polishes with closed-form
curvature, and a **certified discrete** EA on the `DiscreteProblem` seam.

- **Status**: designed
- **Depends on**: 01-01, 01-08, 03-12, 04-01
- **Blocks**: 03-13

## 2. Where it lands

`packages/omnibias-discrete/src/omnibias/discrete/evolution/` for the
population machinery and the certified discrete variant, since the seam and the
anneal schedule live there; the memetic polish calls into `omnibias.torch.optim`
and `omnibias-curvature`.

Not a package. An EA that only recomposes existing pieces fails the "earn
independent existence" test.

## 3. Prior art in omnibias

- `packages/omnibias-discrete/` — the `DiscreteProblem` seam (`n`, `energy`,
  `to_polynomial`, optional `flip_deltas`), `AnnealSchedule`, `anneal_descent`
  (bit-identical torch and jax twins), the rounding and `k`-flip decoder, the
  brute-force oracle, Lasserre/SOS and trivial negative-coefficient lower
  bounds, and `certify_gap`.
- `omnibias.qubo`, `omnibias.submodular`, `omnibias.nphard` — consumers of that
  seam, including an MCTS search track in `nphard`.
- `omnibias.torch.optim` — `CubicNewton`, `GaussNewton`, `KFAC`,
  `TrustRegionNewtonCG`, `JetSubspaceTensor`, `NaturalGradient`.
- `omnibias.struct` — `logsumexp_gap_bound`, the closed-form `log(N)/beta`
  sandwich.
- `GrowableOperatorMultiBiasUnit` — structural growth of `K`.

**Confirmed gap.** No population-based optimizer of any kind. Everything is
gradient descent or annealed relaxation on a single point.

## 4. Mathematics

### Variant A: soft-population homotopy

A population is usually a set of points. Replace it with a **distribution
represented by its softmax weights over candidates**, and make selection a
temperature:

```
w_i(beta) = softmax( -beta E(x_i) )_i
```

At `beta = 0` this is uniform (pure exploration); as `beta -> inf` it
concentrates on the best candidate (pure exploitation). That is **temperature
collapse**, the same feasibility-sense limit `anneal_descent` already uses, now
applied to selection pressure rather than to a relaxation variable.

Two things follow that a standard EA does not have:

1. **A closed-form gap.** The soft-selected expected energy and the best
   energy differ by at most `log(P) / beta` for population size `P`, from
   `logsumexp_gap_bound`. So "how greedy is my selection" has an exact answer.
2. **Exact derivatives of the selection operator.** `d w_i / d E_j` is closed
   form, so the whole selection step is differentiable, which makes
   meta-optimizing the schedule possible (spec 03-07).

### Variant B: bias-geometry evolution

Standard EAs mutate parameters isotropically. Here the parameters have known
geometric roles, so mutation should respect them:

| Gene | Meaning | Sensible mutation |
|---|---|---|
| pack mean `mu_g` | sample location | additive, scaled by the local feature size |
| pack spread `delta_g` | collapse tightness | multiplicative (log-normal) |
| pack order `n_g` | derivative order | discrete `+-1` step, or birth/death |
| outer weight `c_g` | amplitude | additive |
| normal `w` | direction | rotation on the sphere |

Crossover likewise: exchanging whole *packs* between individuals is meaningful
(each pack is a self-contained feature), while exchanging raw coordinate slices
is not. This is the classic "respect the representation" argument, and here the
representation actually has semantics.

### Variant C: jet-memetic

Memetic algorithms interleave global search with local refinement. omnibias
makes the local step unusually strong: `omnibias.torch.optim` provides exact
second-order steps, and spec 03-12 gives an exact line search from a directional
jet. So each individual can be polished to a local optimum in a few exact
Newton steps rather than many gradient steps.

The cost model matters and should be stated: a memetic algorithm is worth it
only when the local polish is cheap relative to the fitness evaluation. Exact
curvature makes polish cheap, which is exactly the regime where memetic methods
win.

### Variant D: certified discrete evolution

On the `DiscreteProblem` seam, an EA is a search over binary assignments. The
contribution over a plain EA is that omnibias already has:

- a relaxation (`anneal_descent`) to seed the population,
- a decoder (rounding plus `k`-flip),
- a lower bound (`Lasserre`/SOS or the trivial negative-coefficient bound),
- and `certify_gap` to sandwich the answer.

So the EA runs inside a **certify loop**: every generation reports the best
decoded value and the current certified gap, and the algorithm can stop when the
gap closes. That is a genuinely different stopping rule from "iterations
exhausted", and it is the honest way to make an EA on an NP-hard problem
publishable.

## 5. Worked example

**Soft-population selection, worked numerically.**

Population of `P = 4` with energies `E = (2.0, 2.3, 3.1, 5.0)`.

At `beta = 1`:

```
-beta E     = (-2.0, -2.3, -3.1, -5.0)
exp         = (0.1353353, 0.1002588, 0.0450492, 0.0067379)
sum         = 0.2873812
w           = (0.470926, 0.348871, 0.156758, 0.023446)
E_soft = sum w_i E_i = 0.941852 + 0.802403 + 0.485950 + 0.117230 = 2.347435
E_best = 2.0,   gap = 0.347435,   bound = log(4)/1 = 1.386294   (sound, 4x loose)
```

At `beta = 10`:

```
w      = (0.9525580, 0.0474183, 0.0000159, ~1e-13)
E_soft = 1.9051160 + 0.1090621 + 0.0000493 = 2.0142274
gap    = 0.0142274,   bound = log(4)/10 = 0.1386294   (sound, ~10x loose)
```

The selection pressure is now a single interpretable number with a certified
meaning: at `beta = 10` the population's soft mean is guaranteed to be within
`0.1386` of the best member, and is within `0.0142` in fact.

**Bias-geometry mutation, illustrated.** An individual has a pack with
`(mu, delta, n, c) = (0.5, 0.01, 2, 1.3)`. A geometry-aware mutation might give
`(0.52, 0.0083, 2, 1.28)`: the mean moved additively, the spread moved
multiplicatively by `exp(-0.19)`, the order was untouched, and the weight moved
additively. An isotropic Gaussian mutation with a single step size would either
barely move `mu` or destroy `delta` (which lives on a log scale spanning
decades). The trade is visible in one example, which is why the variant exists.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/discrete/evolution/_core.py
@dataclass(frozen=True)
class PopulationConfig:
    size: int
    selection_beta: float | AnnealSchedule
    elitism: int = 1

def soft_weights(energies, *, beta: float) -> FloatArray: ...
def selection_gap_bound(*, population: int, beta: float) -> float:
    """log(P) / beta, reusing `logsumexp_gap_bound`."""

@dataclass(frozen=True)
class GeometryMutation:
    mean_sigma: float          # additive
    spread_log_sigma: float    # multiplicative
    order_step_prob: float     # discrete
    weight_sigma: float
    normal_rotation: float     # radians on the sphere
```

```python
# omnibias/discrete/evolution/torch.py  (and jax twin)
def soft_population_evolve(problem, config, *, generations: int, key=None) -> EvolveResult: ...
def memetic_evolve(problem, config, *, polish: Optimizer, polish_steps: int = 3) -> EvolveResult: ...
def certified_discrete_evolve(problem: DiscreteProblem, config, *, bound: BoundKind) -> CertifiedEvolveResult:
    """Runs the EA inside the existing certify loop; stops when the gap closes."""

@dataclass
class CertifiedEvolveResult:
    best_value: float
    lower_bound: float
    gap: float
    gap_closed: bool
    generations_used: int
```

## 7. Practical use cases

1. **Discrete problems where gradients are unavailable or misleading**, with the
   certified gap as the stopping rule instead of a fixed budget.
2. **Architecture search over pack structure**: which orders, how many packs,
   where. The genes are discrete and geometric, which is exactly the EA's home
   ground and awkward for gradients.
3. **Multimodal fitting.** Equality loci and interface problems often have
   several valid branches (spec 01-09); a population naturally covers them
   while a single gradient run picks one.
4. **Hyperparameter and schedule search**, where the differentiable selection
   operator allows meta-gradients.
5. **Robust design.** Populations approximate a distribution over designs, which
   is what robustness under uncertainty needs.

## 8. Acceptance gates

Baselines, all standard and named: CMA-ES, differential evolution, and plain
`anneal_descent`, at matched function-evaluation budget.

- **G1 selection-gap soundness.** `selection_gap_bound` upper-bounds the
  measured soft-versus-best gap on every generation of every run, with zero
  violations.
- **G2 continuous benchmark.** On a standard multimodal suite, the soft-population
  EA is within `1.2x` of CMA-ES's final objective at matched budget. Matching a
  strong baseline is the honest target here; beating CMA-ES on its home ground
  is not the claim.
- **G3 geometry win.** On pack-structure search, the geometry-aware mutation
  beats isotropic mutation by at least `2x` in final objective at matched
  budget, over five seeds. This is the variant's actual claim.
- **G4 memetic win.** Exact-curvature polish reaches a given objective in at
  least `3x` fewer fitness evaluations than the same EA with gradient-descent
  polish.
- **G5 certified stop.** On a suite of small QUBO instances with known optima,
  `certified_discrete_evolve` terminates with `gap_closed = True` and the
  correct optimum on at least 90 percent of instances, and never reports
  `gap_closed = True` incorrectly.
- **G6 parity.** torch and jax bit-identical given the same seed policy.

## 9. Benchmark plan

- `benchmarks/soft_evolution.py`: four arms against three baselines on three
  problem families (continuous multimodal, pack-structure search, small QUBO).
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/evolution/`.
- Random seeds are part of the artifact, since EA results are seed-sensitive.

## 10. Honesty and scope

- The selection temperature `beta -> inf` is **temperature collapse**: a soft
  weighting hardening into a hard argmin, the feasibility sense. It is **not**
  the founding bias collapse (`delta -> 0` coalescence of `K` biases into
  `sigma^(K-1)`), which is what forms the packs the EA mutates. Both appear in
  this spec, in different roles, and the module must carry the cross-reference
  note and be registered in `PENALTY_FILES` in
  `packages/omnibias-core/tests/test_concept_terminology.py`.
- **Evolutionary algorithms are not new**, and neither is memetic hybridization
  or annealed selection. The contributions are narrow and specific: the
  geometry-aware mutation operators, the closed-form selection gap, exact
  curvature in the polish step, and the certify-loop stopping rule.
- G2 deliberately targets *parity* with CMA-ES rather than superiority. An EA
  that claims to beat CMA-ES on generic continuous benchmarks is almost
  certainly measuring something else.
- The certified gap is an **instance-wise sound bound**. It says nothing about
  worst-case complexity, and nothing here bears on P versus NP.

## 11. Open questions and risks

- **Seed sensitivity.** EA comparisons are notoriously fragile; five seeds is
  the minimum and the artifact must record every one, not just the mean.
- **Budget accounting.** Memetic polish costs fitness evaluations too. The
  comparison must count *all* evaluations, including those inside the polish, or
  the win is fictitious.
- **Population diversity collapse** at high `beta` is the classic failure. The
  gap bound diagnoses it (a tiny gap means a concentrated population), which is
  a useful side benefit.
- **Falsifier.** If geometry-aware mutation does not beat isotropic mutation on
  pack-structure search, the whole spec reduces to a wrapper around existing
  pieces and should not ship.

## 12. Implementation checklist

- [ ] `packages/omnibias-discrete/src/omnibias/discrete/evolution/_core.py`
- [ ] torch and jax twins with a documented seed policy and a parity test
- [ ] Reuse `logsumexp_gap_bound`, `AnnealSchedule`, `certify_gap`; fork nothing
- [ ] Selection-gap soundness test on every generation
- [ ] Geometry-versus-isotropic mutation ablation
- [ ] Full-budget accounting test for the memetic variant
- [ ] `certified_discrete_evolve` correctness test against the brute-force oracle
- [ ] Terminology cross-reference note plus `PENALTY_FILES` registration
- [ ] `benchmarks/soft_evolution.py` plus smoke JSON with per-seed records
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
