# 03-09 Differentiable topology

## 1. Thesis and status

Cell counts, Euler characteristics and persistence diagrams of a learned
arrangement or occupancy field are **integer-valued**, hence not differentiable;
soft cell masses and `beta`-smoothed sublevel filtrations give them
differentiable surrogates whose gap to the integer truth is bounded, so topology
can enter a loss function honestly.

- **Status**: concept
- **Depends on**: 01-03, 02-02, 03-03, 03-05
- **Blocks**: 05-01, 05-02

## 2. Where it lands

`packages/omnibias-shape/src/omnibias/shape/topology/` with torch and jax twins.
The shape package owns occupancy fields; topology is a property of them.

## 3. Prior art in omnibias

- `packages/omnibias-shape/` — differentiable soft shape and occupancy fields,
  soft-coverage / soft-OR union operators.
- `omnibias.partition` — `partition_weights`, `hard_assignment`,
  `certify_partition_gap`; the soft-to-hard machinery this reuses.
- `omnibias.struct` — `logsumexp_gap_bound`; the semiring driver includes a
  `Counting` semiring, which is directly relevant to counting components.
- `omnibias.graph` — spectral graph operators, including Laplacians whose
  kernel dimension counts connected components.
- Spec 01-03's arrangement combinatorics and spec 02-02's tope graph.

**Confirmed gap.** No topological invariants anywhere: no Betti numbers, no
Euler characteristic, no persistence.

## 4. Mathematics

### The obstruction, stated first

Topological invariants are integers. An integer-valued function of continuous
parameters is either constant or discontinuous. So there is no differentiable
function that equals the Betti number everywhere — this is not a gap in
technique, it is a fact. Every "differentiable topology" method is therefore a
surrogate, and the honest question is what the surrogate's relationship to the
truth is.

Two respectable answers:

1. **Persistence-based.** Persistence diagrams vary continuously (in the
   bottleneck metric) with the filtration function, and persistence *values*
   (birth and death times) are differentiable almost everywhere. Losses built on
   persistence values are genuinely differentiable; the *number* of features is
   still not.
2. **Soft-count-based.** A soft count with a bounded gap to the integer count.
   Differentiable everywhere, exact nowhere, but with a certified bracket.

Build both, and label which is in use.

### Soft Euler characteristic

For an arrangement, the Euler characteristic of the union of cells with a given
property is an alternating sum over faces. Using soft cell masses `m_c(beta)`
from `partition_weights`:

```
chi_soft(beta) = sum_{faces f} (-1)^{dim f} m_f(beta)
```

Each `m_f` is a product of sigmoids, so `chi_soft` is smooth. As `beta -> inf`
each `m_f -> {0, 1}` and `chi_soft -> chi`. The gap is bounded by the sum of the
individual membership gaps, each of which `certify_partition_gap` bounds.

The bound is a sum over faces, so it is **loose for large arrangements**. State
that: this is a usable surrogate for small arrangements and a weak one for
large.

### Soft persistence

Given a filtration function `f` (for example a soft distance transform from spec
03-05), the sublevel sets `{ f <= t }` change topology at critical values. A
`beta`-smoothed indicator makes the filtration smooth, and the persistence pairs
(birth, death) are then differentiable in the field parameters wherever the
pairing is stable.

Differentiability fails exactly where pairs swap, which is a measure-zero set,
so "differentiable almost everywhere" is the correct claim — the same status as
`max` or sorting, and acceptable for gradient descent.

A persistence loss such as "penalize features with persistence below `epsilon`"
is then a legitimate differentiable objective, and it is the standard way
topology enters modern learning.

### Connected components from the Laplacian

For the tope graph (spec 02-02), the number of connected components is the
dimension of the graph Laplacian's kernel. A soft version uses the spectral
machinery already in `omnibias.graph`:

```
components_soft = sum_i sigma( beta ( epsilon - lambda_i ) )
```

which counts eigenvalues below a threshold with a smooth indicator. Exact as
`beta -> inf` and `epsilon` below the spectral gap; the gap bound follows from
the eigenvalue separation, which `omnibias.core.verified.eig_operator` can
enclose.

That last point is worth noting: **a certified spectral gap turns a soft
component count into a certified integer count**, because if the enclosure of
`lambda_k` is entirely below `epsilon` and that of `lambda_{k+1}` entirely
above, the count is exactly `k`. That is a genuinely sound integer statement, and
it is the strongest topological result available here.

## 5. Worked example

**Soft component count with a certified gap.**

Take a graph with three well-separated clusters. Its Laplacian has eigenvalues

```
lambda = ( 0, 0, 0, 0.41, 0.55, 0.72, ... )
```

three zeros (one per component) and a spectral gap to `0.41`.

Soft count with `epsilon = 0.2`, `beta = 50`:

```
sigma(50 (0.2 - 0))    = sigma(10)    = 0.9999546   (x3)
sigma(50 (0.2 - 0.41)) = sigma(-10.5) = 0.0000275
sigma(50 (0.2 - 0.55)) = sigma(-17.5) = 2.5e-8
sigma(50 (0.2 - 0.72)) = sigma(-26)   = 5.1e-12
components_soft = 3 * 0.9999546 + 0.0000275 + 2.5e-8 = 2.9998913
```

so the soft count is `2.9998913` against a true count of `3`, an error of
`1.09e-4`. Rounding gives the exact answer, and the error is small because
`beta * gap = 50 * 0.21 = 10.5` is comfortably large.

**Making it certified.** Suppose `eig_operator` returns enclosures
`lambda_3 in [-1e-12, 1e-12]` and `lambda_4 in [0.4098, 0.4102]`. Then with
`epsilon = 0.2`:

- every one of `lambda_1 .. lambda_3` is *proved* below `epsilon`,
- `lambda_4` is *proved* above `epsilon`,

so the component count is **exactly 3**, as a sound statement, not an estimate.
The soft count was never needed for the conclusion; it is the differentiable
surrogate used during training, and the certificate is the statement made at the
end. That division of labour is the right pattern for this whole spec.

**Where it breaks.** If the gap were `0.02` instead of `0.21`, `beta` would need
to be `500` for the same accuracy, the gradients would be correspondingly
sharper, and the eigenvalue enclosures would need to be ten times tighter to
certify. Near-degenerate spectra are the failure mode, and the benchmark must
include them.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/shape/topology/_core.py
@dataclass(frozen=True)
class SoftCount:
    value: float
    gap_bound: float
    rounded: int
    certified: bool          # True only when an enclosure proves the integer

def soft_euler_characteristic(arrangement, *, beta: float) -> SoftCount: ...
def soft_component_count(laplacian, *, epsilon: float, beta: float) -> SoftCount: ...
def certified_component_count(laplacian, *, epsilon: float) -> int | Inconclusive:
    """Uses eigenvalue enclosures. Returns Inconclusive when the spectral gap
    does not separate; never guesses."""

@dataclass(frozen=True)
class PersistencePair:
    birth: float
    death: float
    dimension: int
    @property
    def persistence(self) -> float: ...

def soft_persistence(field, *, beta: float, max_dim: int = 1) -> tuple[PersistencePair, ...]: ...
def persistence_loss(pairs, *, threshold: float, mode: Literal["suppress", "encourage"]) -> float:
    """Differentiable almost everywhere; the pairing is piecewise constant."""
```

## 7. Practical use cases

1. **Shape priors in inverse problems** (spec 05-01): "the reconstruction should
   have exactly one connected component and no holes" as a differentiable term.
2. **Segmentation with topological constraints**, where anatomically impossible
   topologies are penalized.
3. **Material microstructure design** with target connectivity.
4. **Mode counting in dynamical systems**, using the certified spectral route.
5. **Model selection.** The number of components a learned partition actually
   uses, certified rather than assumed.

## 8. Acceptance gates

Baselines: a standard persistence library on a discretized grid, and a
connected-components labelling algorithm.

- **G1 hard-limit agreement.** As `beta -> inf`, soft counts converge to the
  exact integer counts on a suite with known topology, over at least four
  doublings, at the predicted rate.
- **G2 gap-bound soundness.** The reported `gap_bound` contains the deviation on
  every instance, dense-grid and random-sample checked, with zero violations.
- **G3 certified counts.** `certified_component_count` is correct whenever it
  returns an integer, verified against exact labelling on at least 10 000
  instances, and returns `Inconclusive` (never a wrong integer) on
  near-degenerate spectra.
- **G4 persistence agreement.** Soft persistence pairs match a standard
  persistence library's diagram in bottleneck distance to `<= 1e-6` on smooth
  test fields.
- **G5 topological-prior win.** On a segmentation task with a known topological
  constraint, adding the persistence loss reduces topological errors by at least
  `50` percent without degrading the pixel metric, over five seeds.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/differentiable_topology.py`: convergence rates, bound soundness,
  certified-count correctness including near-degenerate cases, persistence
  agreement, segmentation task.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/topology/`.

## 10. Honesty and scope

- `beta -> inf` is **temperature collapse** (feasibility sense) throughout this
  spec. The founding bias collapse (`delta -> 0`) appears only in the underlying
  field construction. Cross-reference note plus `PENALTY_FILES` registration
  required.
- **No differentiable function equals a Betti number.** This is stated first in
  section 4 and must appear in any user-facing text. Everything here is a
  surrogate with a stated relationship to the truth.
- Persistence is differentiable **almost everywhere**; it is non-differentiable
  where pairs swap. That is acceptable for gradient descent and must not be
  described as smooth.
- The soft Euler characteristic's bound is a **sum over faces** and is loose for
  large arrangements. Report the face count alongside the bound.
- Certified integer counts require **eigenvalue enclosures that separate**.
  Where they do not, the answer is `Inconclusive`.
- Persistent homology, topological losses and soft counting are all established
  research areas. The contribution here is the integration with the arrangement
  and occupancy machinery and the certified-count route.

## 11. Open questions and risks

- **Cost.** Persistence computation is superlinear and the arrangement face
  lattice is exponential. Both must have stated size ceilings.
- **Gradient quality.** Persistence gradients are sparse (only the critical
  simplices receive gradient), which makes optimization slow. Measure.
- **Near-degeneracy** is the dominant failure mode for the certified route, and
  it is common in practice (symmetric structures have degenerate spectra).
- **Falsifier.** If the certified route almost always returns `Inconclusive` on
  realistic problems, its practical value is small and the honest deliverable is
  the soft surrogate alone.

## 12. Implementation checklist

- [ ] `packages/omnibias-shape/src/omnibias/shape/topology/_core.py`
- [ ] torch and jax twins with a parity test
- [ ] Reuse `partition_weights`, `certify_partition_gap`, `eig_operator`
- [ ] `Inconclusive` return for the certified route, never a guessed integer
- [ ] Convergence-rate tests for the soft counts
- [ ] Certified-count correctness sweep including near-degenerate spectra
- [ ] Persistence agreement test against a standard library
- [ ] Terminology cross-reference note plus `PENALTY_FILES` registration
- [ ] `benchmarks/differentiable_topology.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
