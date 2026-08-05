# omnibias-combinatorics

**Status: Alpha (0.1.0a1).**

Exact **differentiable** matching / flow / matroid layers for omnibias, with a
**tight** rigorous optimality-gap certificate. Unlike TSP / QUBO (NP-hard), the
assignment, transportation, min-cost-flow and matroid problems are in **P** --
solved exactly by Hungarian / LP / max-flow / greedy. The one honest limit is that
the exact combinatorial argmin is piecewise-constant, so its gradient is a.e. zero
and it is useless for learning. This package answers the well-posed question with a
**yes-if**:

> **Yes** you can put an exact combinatorial solver inside a network and train
> *through* it **if** you relax it entropically (Sinkhorn), decode to a vertex, and
> report a certified gap -- which here is **tight** (`~0`) because these polytopes
> are integral.

Concretely, each layer is a three-part object:

\[
\underbrace{c(\theta)}_{\text{cost head}}\;\longrightarrow\;
\underbrace{x^\star = \text{relax}_\beta(c)}_{\text{entropic / Sinkhorn relaxation}}\;\longrightarrow\;
\underbrace{v = \text{decode}(x^\star)}_{\text{feasible vertex (upper bound)}},\qquad
\underbrace{\ell \le \text{opt} \le c^\top v}_{\text{certified (tight) gap}}.
\]

1. **Differentiable entropic relaxation.** An entropic / Sinkhorn map onto the
   problem's polytope (Birkhoff / transportation / flow / matroid); as the inverse
   temperature `beta -> inf` it collapses onto a **polytope vertex**. Unrolled and
   batched for backprop, so a cost model trains *through* the optimizer. The Sinkhorn
   and soft-top-k kernels are reused from [omnibias-graph](../omnibias-graph) (not
   re-implemented).
2. **Decode to a vertex.** Round the annealed relaxation to a feasible solution --
   the upper bound. The exact classical algorithm (`classical_optimum`) is the
   best-in-class baseline.
3. **Tight LP-dual certificate.** The Neumaier-Shcherbina verified LP dual
   (`omnibias.convex.lp_dual_lower_bound`, outward-rounded intervals over
   `omnibias.core.verified`) is a *lower* bound on the true optimum, so
   `lower <= opt <= objective` is a **certified** gap. Because the polytope is
   integral, the LP relaxation is exact and the gap is tight.

## What's in the box

```python
import numpy as np
from omnibias.combinatorics import (
    AssignmentProblem, classical_optimum, decode, certify_gap, brute_force_min,
)
from omnibias.combinatorics.jax import assignment_relaxation  # torch twin available

cost = np.random.default_rng(0).random((6, 6))
prob = AssignmentProblem(cost)

P = assignment_relaxation(cost)              # (n, n) doubly-stochastic heatmap, grad-friendly
sol, obj = decode(prob, relaxed=np.asarray(P))  # a permutation + its cost (upper bound)
opt = classical_optimum(prob)                # Hungarian (scipy) -- the exact baseline
cert = certify_gap(prob, sol)                # lower <= opt <= obj (tight, integral polytope)
print(cert.lower_bound, cert.objective, cert.relative_gap, cert.certified)
```

Every backend function has a bit-identical `omnibias.combinatorics.torch` twin
(float64). Problem front-ends: `AssignmentProblem`, `TransportProblem`,
`MinCostFlowProblem`, `MatroidProblem` (`UniformMatroid` / `PartitionMatroid` /
`GraphicMatroid`).

## Honest scope

- The certificate is a **genuine** rigorous lower bound (LP dual, Neumaier-Shcherbina,
  outward-rounded intervals) plus the decoded upper bound. It never asserts the
  *relaxation* returns the exact argmin -- it reports a gap, which is tight here only
  because these polytopes are integral.
- These problems are in **P**: `classical_optimum` (Hungarian / LP / max-flow / greedy)
  solves them exactly and is the honest baseline. The value-add is the *differentiable*
  layer for training end to end, plus the certificate.
- `brute_force_min` is the exponential vertex-enumeration oracle for small instances
  (assignment / matroid), used only to self-check the sandwich.
- The relaxation layers need a `jax` / `torch` backend **and** `omnibias-graph`
  (Sinkhorn / soft-top-k); the certificate needs `scipy` (core) for the exact LP solve
  and the `convex` extra for the rigorous interval seal (without `convex` it returns the
  valid float LP bound with `certified=False`).
- Field-level math is torch + jax only (repo convention). Extension-tier typing (authored
  strict-clean; not on the shared strict CI gate).

## Tests

```bash
pip install -e "packages/omnibias-combinatorics[convex,jax,torch,graph,test]"
python -m pytest packages/omnibias-combinatorics/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
