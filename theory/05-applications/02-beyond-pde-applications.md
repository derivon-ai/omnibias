# 05-02 Beyond-PDE applications

## 1. Thesis and status

The arrangement, the scan and the multi-pack are not PDE machinery — they are
general-purpose geometry, and three domains with no differential equation in
sight (tabular data, point clouds and implicit shapes, sequences) each get a
concrete construction from them, with honest baselines that are hard to beat.

- **Status**: concept
- **Depends on**: 01-02, 01-03, 02-01, 02-02, 02-08, 03-02, 03-04, 03-07, 03-09
- **Blocks**: none

## 2. Where it lands

Three separate homes, because these are three separate audiences:

- tabular → `omnibias.tab` (already exists, already benchmarked against
  LightGBM),
- shapes → `omnibias.shape` (already exists),
- sequences → a new `omnibias.torch.sequence` submodule, or nothing at all if
  gate G5 fails.

None of them earns a new package. Stating that up front is the point of the
section.

## 3. Prior art in omnibias

- `omnibias.tab` — oblique soft-split gates `sigmoid(beta (w.x - t))`, a depth-1
  additive tier and a depth `>= 2` multiplicative tier, Newton boosting,
  soft-to-hard rounding certificates, and a **LightGBM benchmark already in
  the package** (`omnibias/tab/bench.py`). The baseline is set and it is strong.
- `omnibias.partition` — soft partition of unity over `2**depth` regions, with
  `hard_assignment`, `hardened_rules`, a sound soft-to-hard gap certificate, and
  a `RegionModels` registry driving four bridges including
  `omnibias.tab.decision`.
- `omnibias.shape` — soft occupancy fields and soft-coverage (soft-OR /
  log-sum-exp union) operators with a closed-form derivative tower.
- `omnibias.struct` — certified differentiable dynamic programming over
  sequences: soft Viterbi, CTC, soft-DTW, alignment, semiring / hypergraph
  driver. **Sequences already have a serious home here.**
- `omnibias.graph` — spectral operators, Gumbel-Sinkhorn, SoftSort, soft top-k.

**Partially closed gap, and it matters.** Tabular and shape work is not a gap at
all — those packages exist and are benchmarked. What is missing is (a) the
*arrangement* view of a tabular model, which is different from the tree view,
(b) topology of a learned shape as a trainable quantity, and (c) a transverse
convolution for sequences, which `omnibias.struct` does not provide because it
solves a different problem.

## 4. Mathematics

### (a) Tabular: the arrangement view versus the tree view

An oblique soft tree of depth `d` uses `2^d - 1` split gates arranged in a
**tree**: each gate is reached only along one root-to-node path. An arrangement
of `H` hyperplanes uses all `H` gates for every input, producing up to

```
C(H, 0) + C(H, 1) + ... + C(H, D) = O(H^D)
```

cells in `D` dimensions (Zaslavsky's bound, with equality in general position).

For `D = 10` and `H = 20`: a depth-4 oblique tree has `15` gates and `16`
leaves; an arrangement of `20` hyperplanes has `20` gates and up to
`sum_{i=0}^{10} C(20, i) = 616 666` cells. The arrangement is exponentially more
expressive per gate.

That sounds like a decisive win and it is not, for a reason worth being explicit
about: **expressiveness per parameter is not the binding constraint on tabular
data**. Gradient-boosted trees win on tabular problems because of their
inductive bias toward axis-aligned, piecewise-constant, low-order interactions,
not because they have many leaves. An arrangement's bias is toward *global*
oblique structure, which is the wrong prior for most tabular data and the right
one for a minority of it.

So the honest hypothesis is narrow: **arrangements should beat trees on tabular
problems with genuinely oblique global structure, and lose elsewhere.** That is
testable, and identifying *which* datasets have that structure is the actual
contribution. A model-selection rule that predicts, from the data, whether the
arrangement or the tree will win would be worth more than either model.

### (b) Shapes: soft occupancy with trainable topology

An implicit shape is a sublevel set `{x : f(x) <= 0}`. `omnibias.shape` already
makes `f` an OMBU field with an exact tower, which gives exact normals
(`grad f / |grad f|`), exact mean curvature (`div` of the unit normal), and
exact principal curvatures from the shape operator — all without finite
differences on a grid.

What spec 03-09 adds is that **topology becomes a trainable quantity**: soft
Betti numbers and a soft Euler characteristic, with a bounded gap to the integer
truth. So a reconstruction can be regularized toward "genus 0" or "exactly two
connected components" directly, instead of by proxy penalties on curvature.

The bounded gap is what makes this honest: the soft invariant is not the
integer, and the bound says by how much it can differ.

### (c) Sequences: the causal transverse filter

For a sequence `x_1, ..., x_T`, a causal transverse filter scans a pack template
along the *time* axis with the kernel supported on the past:

```
h_t = sum_{k >= 0} c_k sigma^(n)( alpha (t - k - tau) ),   support k >= 0
```

Because `sigma^(n)` decays exponentially, truncating at `k <= K` costs a bounded
error, and the whole filter is a fixed-width causal convolution — so inference is
`O(T K)` with no recurrence, and the derivative tower gives the filter's
frequency response in closed form (spec 01-07).

**The honest comparison** is not against a recurrent network, which this clearly
beats on parallelism, but against structured state-space models (S4, Mamba and
successors), which already provide long-range causal convolution with principled
parameterizations and are extremely strong. The specific thing an OMBU filter
adds is that its kernel is an **exact derivative of a known function**, so the
filter's spectral profile is designable rather than learned, and the derivative
of the output with respect to the timescale `alpha` is exact.

Whether that is worth anything is exactly what gate G5 asks, and the honest
prior is that it probably is not, in which case this sub-application should be
dropped rather than shipped.

## 5. Worked example

**The tabular hypothesis, made falsifiable.**

Construct two synthetic datasets in `D = 10` dimensions, `n = 10 000` samples
each, binary labels:

*Oblique-structured.* The label is
`y = 1[ w_1 . x > 0 ] XOR 1[ w_2 . x > 0 ]` with `w_1, w_2` dense random
vectors. This is exactly two hyperplanes in general position, giving `4` cells,
with the label constant on each cell.

- An arrangement with `H = 2` learned hyperplanes represents this **exactly**:
  `2` gates, `20` parameters (two `10`-dimensional normals plus offsets), zero
  error at the population level.
- An axis-aligned gradient-boosted tree must approximate two dense oblique
  boundaries with axis-aligned steps. In `D = 10` with dense normals, the
  staircase approximation of a single oblique hyperplane to accuracy `eps`
  requires on the order of `eps^(-(D-1))` leaves — astronomically many. In
  practice the tree plateaus at some accuracy set by its leaf budget.

*Axis-structured.* The label is `y = 1[x_3 > 0.5] AND 1[x_7 < 0.2]`, a
two-condition axis-aligned rule.

- A depth-2 tree represents this exactly with `2` splits.
- An arrangement can also represent it exactly (axis-aligned hyperplanes are a
  special case), but must *learn* that the normals are sparse, which is a harder
  optimization problem in `10` dimensions and will need explicit `L1`
  regularization on the normals to find it reliably.

**The prediction, stated before the experiment**: the arrangement wins by a
large margin on the oblique dataset and roughly ties (or slightly loses, through
optimization difficulty) on the axis dataset. If the arrangement loses on the
*oblique* dataset, the whole tabular sub-application is dead, because that is
the case constructed to favour it.

**The selection rule.** A cheap diagnostic that predicts which regime a real
dataset is in: fit a single linear probe per feature and compare its accuracy to
a single dense linear probe. A large gap in favour of the dense probe indicates
oblique structure. Measuring whether this diagnostic actually predicts the
winner on real benchmarks is the most valuable deliverable in this spec, and it
costs almost nothing to run.

## 6. Proposed API

Tabular and shape APIs mostly exist; these are the additions.

```python
# omnibias/tab/arrangement.py
class ArrangementClassifier(Module):
    """H hyperplanes, soft cell membership, per-cell logits.
    Reuses omnibias.partition's soft partition of unity and its sound
    soft-to-hard gap certificate."""
    def __init__(self, n_features: int, n_hyperplanes: int, *, beta: float = 1.0): ...

def obliqueness_diagnostic(X, y) -> float:
    """Ratio of dense-linear-probe accuracy to best-axis-probe accuracy.
    Predicts whether an arrangement or a tree should be preferred."""

# omnibias/shape/topology.py
def soft_euler_characteristic(field, *, beta: float, grid) -> tuple[float, float]:
    """Returns (value, bound_on_gap_to_integer). Never returns the value alone."""

# omnibias/torch/sequence.py   -- only if G5 passes
class CausalTransverseFilter(Module):
    def __init__(self, *, order: int, alpha: float, width: int): ...
```

## 7. Practical use cases

1. **Tabular problems with rotational structure** — spectroscopy, chemometrics,
   sensor fusion, anything where the informative directions are linear
   combinations rather than raw features.
2. **Model selection.** The obliqueness diagnostic tells a practitioner which
   family to reach for before training either.
3. **Shape reconstruction with topological priors** — reconstructing an organ
   known to be simply connected, or a part known to have exactly three holes.
4. **Interpretable region rules.** `omnibias.partition`'s `hardened_rules`
   already produces readable region descriptions; an arrangement's cells are
   conjunctions of linear inequalities, which is a familiar and auditable form.
5. **Sequence filtering with a designed spectral profile**, if and only if G5
   passes.

## 8. Acceptance gates

Baselines, all strong and all named: LightGBM and XGBoost (tabular, using the
existing `omnibias/tab/bench.py` harness), marching cubes plus a standard
implicit-surface pipeline (shapes), and a structured state-space model
(sequences).

- **G1 oblique win.** On the constructed oblique dataset, the arrangement beats
  a tuned LightGBM by at least `10` accuracy points, over five seeds. **This is
  the constructed-to-win case; failing it kills the sub-application.**
- **G2 axis parity.** On the constructed axis-aligned dataset, the arrangement
  is within `2` accuracy points of LightGBM. Losing badly here means the
  optimization is broken, not that the prior is wrong.
- **G3 real-data honesty.** On at least eight public tabular benchmarks, report
  the full win/loss table. **No aggregate-only reporting.** The expected outcome
  is that the arrangement loses on most of them, and the deliverable is the
  characterization of which ones it wins.
- **G4 diagnostic predictiveness.** The obliqueness diagnostic predicts the
  winner on the eight benchmarks with at least `75%` accuracy, and its
  correlation with the accuracy gap is reported.
- **G5 sequence viability.** The causal transverse filter matches a structured
  state-space baseline within `2%` on a long-range benchmark at matched
  parameter count. **If this fails, the sequence submodule is not shipped** —
  it is removed from the plan, not softened into "promising future work".
- **G6 topology bound.** The soft Euler characteristic's reported gap bound
  contains the true integer in `100%` of test cases, and the API cannot return
  the value without the bound.
- **G7 shape quality.** Topologically regularized reconstruction achieves the
  correct genus in at least `90%` of cases where an unregularized baseline gets
  it wrong, without degrading surface accuracy by more than `5%`.

## 9. Benchmark plan

- Extend `packages/omnibias-tab/src/omnibias/tab/bench.py` with the arrangement
  model, the two constructed datasets, the eight public benchmarks and the
  diagnostic study. Reusing the existing harness means the LightGBM comparison
  is apples to apples.
- `benchmarks/shape_topology.py` for G6 and G7.
- `benchmarks/sequence_transverse.py` for G5, written to be **deleted** along
  with the submodule if the gate fails.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/beyond_pde/`.

## 10. Honesty and scope

- The soft-cell weighting is **temperature collapse** (`beta -> inf`, the
  feasibility sense), exactly as in `omnibias.tab` and `omnibias.partition`. The
  scan's pack order comes from the founding `delta -> 0` bias collapse. Both
  appear in this spec and must be named separately every time.
- **Gradient-boosted trees are expected to win on most tabular benchmarks.**
  Saying this before running the experiment is the difference between a result
  and a sales pitch. The contribution is the identification of the regime where
  they do not, plus the diagnostic that predicts it.
- The arrangement's expressiveness advantage per gate is real (Zaslavsky) and
  irrelevant to most tabular data. Quoting the cell count as evidence of
  practical superiority would be misleading.
- `omnibias.struct` already handles sequences seriously through certified
  differentiable dynamic programming. The transverse filter is a different and
  much smaller idea, and must not be presented as omnibias's sequence story.
- Structured state-space models are strong, actively developed, and the honest
  prior is that the transverse filter does not beat them. G5 exists to find that
  out cheaply.
- The soft topology invariants are **not** the integer invariants; the gap bound
  is the whole content of the claim.
- No certificate tier beyond what `omnibias.partition` and `omnibias.tab`
  already provide (the sound soft-to-hard gap).

## 11. Open questions and risks

- **Cell explosion.** `O(H^D)` cells cannot be enumerated for meaningful `H` and
  `D`; the implementation must work with soft memberships and never materialize
  the cell list. Any design that enumerates cells is wrong.
- **Optimization difficulty.** Learning `H` hyperplanes jointly is non-convex
  and initialization-sensitive in a way tree induction is not. This is the most
  likely cause of a G2 failure and needs a documented initialization strategy.
- **Sparsity for axis-aligned structure.** Without `L1` on the normals, the
  arrangement will not recover sparse rules; with too much, it degenerates to an
  axis-aligned tree with worse optimization. The regularization path needs to be
  swept, not tuned once.
- **Diagnostic overfitting.** A diagnostic tuned on the same eight benchmarks it
  is evaluated on proves nothing; it must be fixed before the benchmarks are
  run, or validated on a held-out set of datasets.
- **Falsifier.** G1 is the falsifier for the tabular application and G5 for the
  sequence one. Both are designed to be cheap and to be run early, before any
  implementation effort is sunk into the parts they would invalidate.

## 12. Implementation checklist

- [ ] `packages/omnibias-tab/src/omnibias/tab/arrangement.py` reusing
      `omnibias.partition`'s soft partition and gap certificate
- [ ] Soft memberships only; assert that cells are never enumerated
- [ ] Documented initialization strategy for joint hyperplane learning
- [ ] `L1` regularization path on normals, swept not tuned
- [ ] `obliqueness_diagnostic` fixed and frozen before benchmarks are run
- [ ] Two constructed datasets (oblique, axis) as the falsifier gates
- [ ] Eight public benchmarks with the **full win/loss table**, no aggregates
- [ ] `packages/omnibias-shape/src/omnibias/shape/topology.py` returning
      value and bound together, never value alone
- [ ] `benchmarks/sequence_transverse.py` run **first**, with the submodule
      built only if G5 passes
- [ ] Extend `omnibias/tab/bench.py` rather than writing a parallel harness
- [ ] Docs page and nav entry, carrying the "trees usually win" statement
- [ ] Index row in `theory/README.md`
