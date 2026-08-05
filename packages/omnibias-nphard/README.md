# omnibias-nphard

**Status: Alpha (0.1.0a1).**

Differentiable, certified **heuristics** for named **NP-hard** families -- quadratic
assignment (QAP), generalized assignment (GAP), and parallel-machine scheduling -- on
the omnibias stack. It complements the other NP-hard packages
[omnibias-qubo](../omnibias-qubo) (QUBO / Ising) and [omnibias-routing](../omnibias-routing)
(TSP), and is the applied "named families" layer on top of them.

These problems are **NP-hard**: there is no poly-time differentiable map to the exact
optimum (that would imply `P = NP`, and the exact argmin's gradient is a.e. zero). So --
exactly like `omnibias-qubo` / `omnibias-routing`, and unlike the P-class
`omnibias-combinatorics` (whose certificate is *tight*) -- this package answers the
well-posed question with a **yes-if**:

> **Yes** you can put an NP-hard solver inside a network and train *through* it (and let
> an MCTS search tackle it) **if** you accept (1) a differentiable annealed *relaxation* +
> a structure-preserving *decoder* (a strong heuristic upper bound), and (2) a **sound but
> generally non-zero** certified optimality gap `lower <= optimum <= decoded`. The gap is
> honest -- a weaker bound only widens it; it is never asserted zero.

Each family reduces to a **quadratic pseudo-Boolean (QUBO-form)** energy, so the
relaxation, decoder and certificate are reused from `omnibias-qubo` rather than
re-implemented; this package adds the family constructors, structure-preserving decoders,
named classical baselines, decision-focused training, and a search track.

\[
\underbrace{w(\theta)}_{\text{predicted weights}}\;\longrightarrow\;
\underbrace{x^\star = \text{relax}_\beta(\text{QUBO}(w))}_{\text{annealed relaxation}}\;\longrightarrow\;
\underbrace{v = \text{decode}(x^\star)}_{\text{feasible solution (upper bound)}},\qquad
\underbrace{\ell \le \text{opt} \le E(v)}_{\text{certified (non-tight) gap}}.
\]

## What's in the box

<!-- docs-test: slow -->
```python
import numpy as np
from omnibias.nphard import qap, decode, certify_gap, brute_force_min, classical_optimum

rng = np.random.default_rng(0)
flow = rng.random((5, 5)); dist = rng.random((5, 5))
problem = qap(flow, dist)                       # a QAP as a QUBO-form problem

sol, obj = decode(problem)                      # a valid permutation + its QAP cost (upper bound)
base = classical_optimum(problem)               # scipy FAQ / 2-opt QAP -- the named baseline
cert = certify_gap(problem, sol, kind="sos")    # lower <= opt <= obj  (sound, generally non-tight)
print(cert.lower_bound, cert.energy, cert.relative_gap, cert.certified)
```

Problem front-ends: `qap` (`QAPProblem`), `gap` (`GAPProblem`), `schedule`
(`SchedulingProblem`). Differentiable relaxation layers live in `omnibias.nphard.jax` /
`omnibias.nphard.torch` (bit-identical float64 twins); the MCTS search track is in
`omnibias.nphard.search`.

## Honest scope

- These families are **NP-hard**. The package is a *differentiable heuristic + a sound
  optimality-gap certificate*, never an exact-optimality (`P = NP`) claim. The gap is
  generally **non-zero** (contrast `omnibias-combinatorics`, whose integral polytopes make
  the gap tight).
- `certify_gap` returns a rigorous **lower** bound (`kind="spectral"` box-QP, or the
  tighter `kind="sos"` Lasserre bound over the Boolean hypercube) plus the decoded
  **upper** bound. `kind="sos"` needs the `sos` extra; the spectral seal needs the
  `convex` extra; each degrades honestly (`certified=False`).
- Named classical baselines (`classical_optimum`): `scipy.optimize.quadratic_assignment`
  (FAQ / 2-opt) for QAP, LPT for scheduling, an LP-relaxation lower bound + greedy for GAP
  (optional OR-Tools CP-SAT via the `ortools` extra).
- `brute_force_min` is the **exponential** exact oracle (`n!` permutations / `m^n`
  assignments) for tiny instances only, used to self-check the sandwich.
- The MCTS search track is a **heuristic** (AlphaZero-style prior from the differentiable
  relaxation); the found solution is still handed to `certify_gap` for a sound gap. It is
  not an optimality guarantee.
- The relaxation layers need a `jax` / `torch` backend. Extension-tier typing (authored
  strict-clean; not on the shared strict CI gate).

## Tests

```bash
pip install -e "packages/omnibias-nphard[sos,convex,jax,torch,test]"
python -m pytest packages/omnibias-nphard/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
