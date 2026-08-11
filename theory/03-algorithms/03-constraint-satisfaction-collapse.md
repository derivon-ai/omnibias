# 03-03 Constraint satisfaction by collapse

## 1. Thesis and status

A CSP clause is a soft-OR over literal gates, so a whole instance is a product
of soft-ORs whose `beta -> inf` limit is exact satisfaction; annealing that
temperature turns constraint satisfaction into a differentiable descent with a
closed-form gap to the hard problem.

- **Status**: designed
- **Depends on**: 01-03, 01-08, 01-09
- **Blocks**: 03-09

## 2. Where it lands

`packages/omnibias-discrete/src/omnibias/discrete/csp/` beside the existing
`omnibias.discrete.maxsat` consumer, since it is the same seam with a different
front end.

## 3. Prior art in omnibias

- `packages/omnibias-discrete/src/omnibias/discrete/maxsat/` — weighted CNF to a
  pseudo-Boolean `MaxSATProblem`, already on the `DiscreteProblem` seam.
- `omnibias-logic` — differentiable and certified Boolean logic: weighted MaxSAT
  plus weighted `#SAT` model counting with inclusion-exclusion count enclosures.
- `omnibias-boolean` — exact ANF / Reed-Muller and Walsh spectra, Boolean
  differential calculus, reproductive equation solving (eliminant plus GF(2)),
  and a `beta`-annealed soft-gate solver.
- `omnibias.discrete` — `AnnealSchedule`, `anneal_descent`, rounding and
  `k`-flip decoder, `certify_gap`.
- `omnibias-shape` — soft-OR / log-sum-exp union operators with a closed-form
  derivative tower.

**Confirmed gap.** MaxSAT and Boolean solving exist. What is missing is the
*general finite-domain CSP* front end: variables over arbitrary finite domains,
`n`-ary constraints given as relations, global constraints (`all-different`,
cardinality), and the arc-consistency structure that CSP solvers rely on.

## 4. Mathematics

### Encoding

A CSP has variables `x_1 .. x_n` over finite domains `D_1 .. D_n` and
constraints `C_k` over subsets of variables. One-hot encode:

```
p_{i,v} in [0, 1],    sum_v p_{i,v} = 1
```

which is a point in a product of simplices. The relaxation is the interior; the
CSP solutions are vertices.

A constraint `C_k` over variables `S_k` with allowed tuple set `R_k` has
satisfaction probability under the product distribution

```
s_k(p) = sum_{t in R_k} prod_{i in S_k} p_{i, t_i}
```

which is a multilinear polynomial. **Exact, differentiable, and closed form.**

The whole instance is satisfied when every `s_k = 1`, so the natural objective is

```
E(p) = sum_k w_k ( 1 - s_k(p) )
```

and `E = 0` exactly on satisfying assignments (at vertices).

### Where the temperature enters

Two distinct places, and conflating them is the standard confusion:

1. **Simplex sharpness.** Parameterize `p_{i,.} = softmax(beta_1 z_{i,.})`. As
   `beta_1 -> inf` the distribution concentrates on a vertex: temperature
   collapse, feasibility sense.
2. **Soft-OR sharpness.** For clause-form constraints, `s_k` can instead be
   written as a soft-OR of literal satisfactions with its own `beta_2`, and
   `beta_2 -> inf` recovers the hard disjunction: also temperature collapse, but
   a different knob.

Both are `beta -> inf` limits and neither is the founding bias collapse. The
implementation should expose them separately, because annealing them together is
usually wrong: sharpening the simplex too early freezes the search.

### The gap

For the soft-OR form, `logsumexp_gap_bound` gives the standard `log(m)/beta`
sandwich for an `m`-literal clause. Summing over clauses,

```
| E_soft(p) - E_hard(round(p)) |  <=  sum_k w_k log(m_k) / beta_2  +  rounding term
```

so the relaxation's bias is a computed quantity. Combined with the `k`-flip
decoder and a lower bound, `certify_gap` produces an instance-wise sandwich.

### Global constraints

The value of a CSP framework over plain SAT is global constraints, which have
compact relaxations:

| Constraint | Relaxation | Exact at vertices? |
|---|---|---|
| `all-different` over `S` | `sum_v ( sum_{i in S} p_{i,v} - 1 )_+^2` | yes |
| cardinality `sum x_i = k` | `( sum_i p_{i,1} - k )^2` | yes |
| `element` / table | the multilinear `s_k` above | yes |

Each is differentiable with closed-form gradients, and each is exactly zero
precisely on feasible vertex assignments. That last property is what makes the
relaxation a *relaxation* rather than a heuristic surrogate.

### Arc consistency as a differentiable filter

Classical CSP solvers propagate: if a value has no support in a neighbouring
domain, remove it. The soft analogue multiplies `p_{i,v}` by a support score

```
supp_{i,v} = prod_{k : i in S_k} ( sum_{t in R_k, t_i = v} prod_{j != i} p_{j, t_j} )
```

and renormalizes. This is a differentiable message-passing step, closely related
to belief propagation, and it converges to arc consistency in the hard limit.
Worth having, and worth labelling accurately: it is soft arc consistency, a known
idea, made differentiable.

## 5. Worked example

**Graph colouring**: colour a triangle with 3 colours, all edges `!=`.

Variables `x_1, x_2, x_3`, domains `{R, G, B}`, constraints `x_1 != x_2`,
`x_2 != x_3`, `x_1 != x_3`. Solutions: the `3! = 6` proper colourings.

Start from the uniform point `p_{i,v} = 1/3` for all `i, v`.

Constraint satisfaction probability for `x_1 != x_2` under independence:

```
s = 1 - sum_v p_{1,v} p_{2,v} = 1 - 3 * (1/3)(1/3) = 1 - 1/3 = 2/3
```

so `E = 3 * (1 - 2/3) = 1` at the uniform point with unit weights.

Now take a partially decided point: `p_1 = (0.8, 0.1, 0.1)`,
`p_2 = (0.1, 0.8, 0.1)`, `p_3 = (1/3, 1/3, 1/3)`.

```
s_12 = 1 - (0.8*0.1 + 0.1*0.8 + 0.1*0.1) = 1 - (0.08 + 0.08 + 0.01) = 1 - 0.17 = 0.83
s_13 = 1 - (0.8 + 0.1 + 0.1)/3 = 1 - 1/3 = 0.666667
s_23 = 1 - 1/3 = 0.666667
E = (1 - 0.83) + (1 - 0.666667) + (1 - 0.666667) = 0.17 + 0.333333 + 0.333333 = 0.836667
```

Gradient with respect to `p_{3,R}`: only `s_13` and `s_23` involve `x_3`, and

```
dE/dp_{3,R} = p_{1,R} + p_{2,R} = 0.8 + 0.1 = 0.9
dE/dp_{3,G} = p_{1,G} + p_{2,G} = 0.1 + 0.8 = 0.9
dE/dp_{3,B} = p_{1,B} + p_{2,B} = 0.1 + 0.1 = 0.2
```

so descent pushes `x_3` toward `B`, which is exactly the arc-consistency
inference: `R` and `G` are largely taken, `B` is free. **The gradient reproduces
constraint propagation**, and it does so with closed-form derivatives on the
multilinear objective.

At the vertex `p_1 = e_R`, `p_2 = e_G`, `p_3 = e_B`: every `s_k = 1` and `E = 0`
exactly — a genuine solution, not an approximate one.

**Certified statement.** With the soft-OR form at `beta = 20` and `m = 3`
literals per clause, the per-clause bias is `log(3)/20 = 0.0549` and the
three-clause total is `0.1648`, so a soft objective below `0.1648` guarantees a
satisfying assignment exists after rounding. That is the shape of the
certificate: not "probably satisfiable", but a bound.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/discrete/csp/_core.py
@dataclass(frozen=True)
class Variable:
    name: str
    domain: tuple[object, ...]

@dataclass(frozen=True)
class Relation:
    scope: tuple[int, ...]
    allowed: frozenset[tuple[int, ...]]
    weight: float = 1.0

@dataclass(frozen=True)
class GlobalConstraint:
    kind: Literal["all_different", "cardinality", "element"]
    scope: tuple[int, ...]
    parameter: int | None = None

class CSP(DiscreteProblem):
    """Implements the existing seam: n, energy, to_polynomial, flip_deltas."""
    variables: tuple[Variable, ...]
    relations: tuple[Relation, ...]
    globals: tuple[GlobalConstraint, ...]
```

```python
# omnibias/discrete/csp/torch.py  (and jax twin)
def csp_solve(
    csp: CSP, *, simplex_schedule: AnnealSchedule, clause_schedule: AnnealSchedule,
    steps: int, arc_consistency: bool = True,
) -> CSPResult:
    """Two independent temperature schedules, never a single fused one."""

def certify_csp(csp: CSP, assignment, *, beta: float) -> GapCertificate: ...
def soft_arc_consistency(csp: CSP, p, *, iterations: int = 3): ...
```

## 7. Practical use cases

1. **Scheduling and timetabling** with global constraints, where a
   differentiable relaxation can be embedded in a larger learned pipeline.
2. **Graph colouring and frequency assignment**, the classical benchmarks.
3. **Neural-symbolic integration.** A neural front end predicts constraint
   weights or domains, and the CSP layer is differentiable end to end.
4. **Configuration problems** where feasibility must be certified, not just
   found.
5. **Puzzle-like structured prediction** (Sudoku, Latin squares) as a testbed
   with unambiguous ground truth.

## 8. Acceptance gates

Baselines: a standard CSP solver (backtracking with arc consistency), the
existing `omnibias.discrete.maxsat` path via a SAT encoding, and simulated
annealing.

- **G1 exactness at vertices.** `E(p) = 0` if and only if `p` is a vertex
  encoding a satisfying assignment, verified exhaustively for all instances with
  at most `3^6` assignments.
- **G2 gap soundness.** The certified bound never claims satisfiability where
  brute force shows none, on a randomized suite spanning the satisfiability
  phase transition. Zero violations.
- **G3 solve rate.** On random binary CSPs at the phase transition, the solve
  rate is within `10` percentage points of the backtracking solver at matched
  wall time, over five seeds. Parity is the honest target; a differentiable
  relaxation is not expected to beat a mature complete solver.
- **G4 differentiability win.** On a neural-symbolic task where constraint
  weights are learned, the differentiable path beats a two-stage
  predict-then-solve baseline in end-to-end accuracy, with skill `> 0`. **This
  is the actual claim of the spec.**
- **G5 global-constraint correctness.** Each global constraint's relaxation is
  exactly zero on feasible vertices and strictly positive otherwise, verified
  exhaustively on small scopes.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/csp_collapse.py`: exactness checks, phase-transition solve rates
  against the baselines, neural-symbolic task, global-constraint validation.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/csp/`.

## 10. Honesty and scope

- **Both temperatures are temperature collapse** (feasibility sense), and there
  are two of them: simplex sharpness and clause sharpness. Neither is the
  founding bias collapse (`delta -> 0`), and the module carries the
  cross-reference note and a `PENALTY_FILES` entry.
- **A relaxation is not a complete solver.** It can fail to find a solution that
  exists, and it never proves unsatisfiability. G3 targets parity, not
  superiority, and the honest framing is "differentiable and certifiable", not
  "faster".
- Soft arc consistency is a known idea (soft constraint propagation, belief
  propagation adjacent). The contribution is its exact differentiability on this
  substrate.
- Certified statements are **instance-wise sound gaps**. Nothing here bears on
  complexity theory or on P versus NP.

## 11. Open questions and risks

- **Local minima.** The multilinear objective has many, and the relaxation will
  get stuck. Restarts and the population methods of spec 03-01 are the obvious
  mitigations; measure how much they help.
- **Domain-size scaling.** One-hot encoding is `O(sum |D_i|)` variables, which is
  fine for small domains and bad for large ones. State the practical ceiling.
- **Schedule interaction.** Two temperatures means a two-dimensional schedule
  space. This is a real tuning burden and should be reduced to a defensible
  default by measurement, not left to the user.
- **Falsifier.** If the differentiable path loses the neural-symbolic task (G4)
  to two-stage, the spec has no unique value, since G3 only targets parity.

## 12. Implementation checklist

- [ ] `packages/omnibias-discrete/src/omnibias/discrete/csp/_core.py` on the
      existing `DiscreteProblem` seam
- [ ] torch and jax twins with a parity test
- [ ] Reuse `AnnealSchedule`, `anneal_descent`, `certify_gap`,
      `logsumexp_gap_bound`; fork nothing
- [ ] Exhaustive vertex-exactness test for small instances
- [ ] Global-constraint exactness tests
- [ ] Phase-transition benchmark against a complete solver
- [ ] Two-schedule default derived from measurement
- [ ] Terminology cross-reference note plus `PENALTY_FILES` registration
- [ ] `benchmarks/csp_collapse.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
