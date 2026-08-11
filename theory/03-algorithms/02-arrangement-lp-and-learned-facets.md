# 03-02 Arrangement LP with learned facets

## 1. Thesis and status

A linear program's feasible region **is** a hyperplane arrangement's cell, so a
learned arrangement is a learned polytope: the constraints become trainable, the
vertex structure is the tope graph, and the existing interior-point and duality
machinery supplies both the solve and the certificate.

- **Status**: designed
- **Depends on**: 01-03, 01-08, 02-02
- **Blocks**: 05-02

## 2. Where it lands

`packages/omnibias-convex/src/omnibias/convex/arrangement/` — the convex package
owns LP solving and duality certificates; this adds the learned-geometry front
end.

## 3. Prior art in omnibias

- `packages/omnibias-convex/` — `solve_lp`, `lp_dual_lower_bound`, the
  closed-form-Hessian log-barrier interior-point solver, KKT implicit-function
  gradients (`qp_layer`), and verified optimality enclosures.
- `omnibias.routing` — `certify_tour_gap` with a Neumaier-Shcherbina LP gap
  certificate; the reference for how a rigorous LP bound is produced here.
- `omnibias.combinatorics` — entropic relaxations onto integral polytopes with a
  tight LP-dual optimality-gap certificate.
- `omnibias.partition` — `partition_weights` over oblique gates, the
  differentiable half-space machinery.
- Spec 01-03's `Arrangement`, spec 02-02's tope graph.

**Confirmed gap.** LP solving exists for *given* constraints. Nothing learns the
constraint matrix as part of an end-to-end model, and nothing connects the
solver to arrangement combinatorics.

## 4. Mathematics

### The identification

An LP in standard inequality form

```
minimize   c . x        subject to   A x <= b
```

has feasible region `P = { x : a_i . x - b_i <= 0 }`, which is exactly the cell
of the arrangement `{ h_i = 0 }` with all signs negative. So:

| LP object | Arrangement object |
|---|---|
| constraint `i` | hyperplane `h_i` |
| feasible region | one cell (a tope) |
| active set at a point | the zero coordinates of the sign vector |
| vertex | a flat of codimension `D` |
| edge | a flat of codimension `D - 1` |
| simplex pivot | a walk in the tope's vertex-edge graph |

The dictionary is exact, and it means that spec 01-03's soft cell membership and
spec 02-02's graph are LP objects in disguise.

### Learning the constraints

In predict-then-optimize problems, `A`, `b` and `c` are outputs of a model. The
gradient of the LP solution with respect to them is what makes the pipeline
trainable. Two routes, both present in the repo:

1. **KKT implicit differentiation** (`qp_layer`'s pattern). Exact where the
   active set is stable; discontinuous where it changes, which is the known
   weakness of LP layers.
2. **Soft-cell relaxation.** Replace the hard feasibility indicator with the
   product of sigmoids from `partition_weights`, giving a smooth surrogate whose
   `beta -> inf` limit is the hard polytope. This is **temperature collapse**,
   and it repairs the differentiability at active-set changes at the price of a
   `log(n)/beta` bias, which is exactly the quantity `certify_partition_gap`
   bounds.

Having both, with an explicit statement of which is in use, is the right design:
route 1 for accuracy where it applies, route 2 for gradient flow where it does
not.

### The certificate

Weak duality gives, for any dual feasible `y >= 0` with `A^T y = c`,

```
b . y  <=  c . x*  <=  c . x_feasible
```

so a feasible primal point and a feasible dual point sandwich the optimum. With
outward-rounded interval arithmetic on the dual feasibility check — the
Neumaier-Shcherbina approach `omnibias.routing` already uses — the lower bound
is **sound in floating point**, not merely computed.

This is important and worth stating precisely: a naive LP bound computed in
floats is not a proof, because the dual point may be infeasible by a rounding
error. Neumaier-Shcherbina perturbs the bound to account for that, and the
result is a rigorous enclosure.

### Vertex enumeration and the tope graph

For small problems, enumerating the tope's vertices gives the exact optimum by
inspection, and cross-checks the solver. This is exponential in general
(`O(n^{D/2})` vertices by the upper bound theorem), so it is a *verification*
tool for small instances, not a method. The benchmark must state the size cutoff.

### What this is not

It is not a new LP algorithm. Interior-point methods are mature and the repo
already wraps one. The contribution is the learned-constraint front end, the
arrangement view that makes the combinatorics explicit, and the honest sound
certificate carried through.

## 5. Worked example

A two-variable LP with four constraints:

```
minimize   -x - y
subject to   x <= 2,   y <= 2,   x + y <= 3,   -x <= 0,  -y <= 0
```

Feasible region: the pentagon with vertices `(0,0), (2,0), (2,1), (1,2), (0,2)`.

Objective `-x - y` is minimized by maximizing `x + y`, so the optimum is on
`x + y = 3`: any point on the edge from `(2,1)` to `(1,2)`, with value `-3`.

**Degenerate on purpose**: the optimum is an edge, not a vertex, so the solution
is not unique. That is the case where naive implicit differentiation is
ill-defined, and it is the reason the soft route exists.

**Soft cell value at `beta = 5`.** The soft feasibility weight at `(1.5, 1.5)`
(the edge midpoint, on the boundary of `x + y <= 3`):

```
h_1 = x - 2       = -0.5    sigma(5 * 0.5)  = sigma(2.5)  = 0.9241418
h_2 = y - 2       = -0.5    sigma(2.5)                    = 0.9241418
h_3 = x + y - 3   =  0.0    sigma(0)                      = 0.5
h_4 = -x          = -1.5    sigma(7.5)                    = 0.9994472
h_5 = -y          = -1.5    sigma(7.5)                    = 0.9994472
weight = product                                          = 0.4265470
```

The `0.5` factor from the active constraint is the signature of being exactly on
a facet, and it is what makes the soft weight differentiable there. The
associated gap bound from `certify_partition_gap` is `log(5)/5 = 0.3219`, which
is loose at this `beta` and tightens linearly.

**Sound dual bound.** For `min c.x` subject to `A x <= b`, the dual is
`max -b.y` subject to `A^T y = -c`, `y >= 0`. Here `-c = (1, 1)` and the third
row of `A` is `(1, 1)`, so

```
y = (0, 0, 1, 0, 0)        A^T y = (1, 1) = -c,   y >= 0     feasible
dual value = -b . y = -(3 * 1) = -3
primal value                    = -3
```

The sandwich is tight, which makes this a good unit test: there is no gap in
which a sign error could hide. Fixing this convention once and asserting it is
gate G5, because inverted duality signs are the second most common source of
silently wrong bounds, after floating-point dual infeasibility — and the
Neumaier-Shcherbina correction addresses only the second.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/convex/arrangement/_core.py
@dataclass(frozen=True)
class LearnedPolytope:
    normals: FloatArray        # A
    offsets: FloatArray        # b
    def as_arrangement(self) -> Arrangement: ...
    def soft_membership(self, x, *, beta: float) -> FloatArray: ...

class DiffMode(StrEnum):
    KKT = "kkt"        # exact, discontinuous at active-set changes
    SOFT = "soft"      # smooth, biased by log(n)/beta
```

```python
# omnibias/convex/arrangement/torch.py  (and jax twin)
class LPLayer(nn.Module):
    def __init__(self, mode: DiffMode, *, beta: float | None = None,
                 dtype=None) -> None: ...
    def forward(self, polytope: LearnedPolytope, c: Tensor) -> LPOutput: ...

@dataclass
class LPOutput:
    x: Tensor
    value: Tensor
    lower_bound: Tensor        # sound, Neumaier-Shcherbina
    active_set: Tensor
    degenerate: Tensor         # flagged, not hidden
    mode: DiffMode             # which route produced the gradient
```

Returning `mode` and `degenerate` in the output is deliberate: both are needed
to interpret the gradient, and both are invisible otherwise.

## 7. Practical use cases

1. **Predict-then-optimize.** A model predicts costs or constraints and is
   trained through the LP solution, with the sound bound reported alongside.
2. **Learned feasible sets.** Safety constraints inferred from data rather than
   specified, where the arrangement view makes the learned rules readable.
3. **Inverse optimization.** Recovering the constraints that explain observed
   decisions: the constraints are exactly the learnable object here.
4. **Certified decision layers**, feeding `omnibias.struct.decision` and
   `omnibias.tab.decision` with a bound rather than a point.
5. **Small-instance verification** of combinatorial relaxations against exact
   vertex enumeration.

## 8. Acceptance gates

Baselines: a standard differentiable LP layer with KKT differentiation only, and
a soft-constraint penalty formulation.

- **G1 solver agreement.** For randomized feasible LPs with `D <= 8`, `n <= 20`,
  `solve_lp` agrees with exact vertex enumeration on the optimal value to
  `<= 1e-10`.
- **G2 sound bound.** The reported `lower_bound` never exceeds the true optimum,
  verified against exact rational LP solutions on a randomized suite including
  deliberately ill-conditioned instances. Zero violations.
- **G3 degeneracy handling.** On degenerate instances the layer sets
  `degenerate = True` and the `SOFT` mode still produces a finite, useful
  gradient where `KKT` does not.
- **G4 predict-then-optimize skill.** On a shortest-path or knapsack-relaxation
  decision task, the LP layer beats a two-stage predict-then-solve baseline in
  decision regret, with skill `> 0`, over five seeds.
- **G5 convention test.** A dedicated test fixes the duality sign convention and
  asserts it on a case with a known dual, because convention errors are the
  classic silent failure here.

## 9. Benchmark plan

- `benchmarks/arrangement_lp.py`: solver agreement, bound soundness including
  ill-conditioned instances, degeneracy behaviour, decision-regret task, and a
  size table showing where vertex enumeration stops being feasible.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/arrlp/`.

## 10. Honesty and scope

- The soft route's `beta -> inf` is **temperature collapse** (feasibility sense),
  not the founding bias collapse (`delta -> 0`). The module must carry the
  cross-reference note and be registered in `PENALTY_FILES`.
- **This is not a new LP algorithm.** It wraps the existing interior-point
  solver and adds a learned-constraint front end plus the arrangement view.
- The lower bound is sound **only** with the Neumaier-Shcherbina correction. A
  float dual bound without it is not a proof, and the implementation must not
  offer an uncorrected fast path that could be mistaken for one.
- Vertex enumeration is exponential and is a verification tool for small
  instances only. The benchmark states the cutoff.
- Certificate tier: sound enclosure for the LP bound. Nothing here bears on
  complexity theory.

## 11. Open questions and risks

- **Active-set discontinuity** is intrinsic to LP layers. The soft route trades
  it for bias; neither is free, and the honest deliverable is a clear rule for
  when to use which.
- **Infeasibility.** A learned constraint set can become infeasible during
  training. The layer must detect and report it, and the training loop needs a
  policy (a feasibility restoration step, or a barrier on the constraint
  parameters).
- **Scale.** Interior-point cost is superlinear; inside a training loop with a
  solve per sample, this may dominate. Measure the practical instance-size
  ceiling.
- **Falsifier.** If the two-stage baseline matches the LP layer on decision
  regret, end-to-end training through the solver is not paying for itself.

## 12. Implementation checklist

- [ ] `packages/omnibias-convex/src/omnibias/convex/arrangement/_core.py`
- [ ] torch and jax twins with a parity test
- [ ] Reuse `solve_lp`, `lp_dual_lower_bound`, `qp_layer`; fork nothing
- [ ] Vertex-enumeration cross-check for small instances
- [ ] Bound soundness test against exact rational LP solutions
- [ ] Duality sign-convention test with a known dual
- [ ] Infeasibility detection and reporting test
- [ ] Terminology cross-reference note plus `PENALTY_FILES` registration
- [ ] `benchmarks/arrangement_lp.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
