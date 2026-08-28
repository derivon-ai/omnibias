---
name: omnibias-nphard
description: Run differentiable certified heuristics for named NP-hard families (QAP, GAP, scheduling) on omnibias-qubo, with an MCTS track. Use when inventing a new family encoding or a structure-preserving decoder with a gap certificate.
---

# omnibias-nphard

Differentiable, certified heuristics for named NP-hard families (quadratic assignment,
generalized assignment, parallel-machine scheduling) for omnibias: each family is
encoded as a quadratic pseudo-Boolean (QUBO-form) energy on top of omnibias-qubo,
relaxed by the annealed temperature-collapse (beta -> inf) penalty, decoded by a
structure-preserving heuristic, and sandwiched by a rigorous -- but honestly non-tight
-- optimality-gap certificate (spectral / SOS-Lasserre). Named classical baselines
(scipy FAQ/2-opt QAP, LPT, LP relaxation) and a from-scratch MCTS search track that uses
the differentiable relaxation as an AlphaZero-style prior. Bit-identical torch + jax
twins.

## Why nested AD fails

Nested AD cannot search a combinatorial assignment. Generic QUBO relaxations have no
named-baseline bakeoff and no honest gap object. MCTS without a differentiable prior is
a separate stack.

## What only this tower unlocks

Differentiable, certified heuristics for named NP-hard families (quadratic assignment,
generalized assignment, parallel-machine scheduling) for omnibias: each family is
encoded as a quadratic pseudo-Boolean (QUBO-form) energy on top of omnibias-qubo,
relaxed by the annealed temperature-collapse (beta -> inf) penalty, decoded by a
structure-preserving heuristic, and sandwiched by a rigorous -- but honestly non-tight
-- optimality-gap certificate (spectral / SOS-Lasserre). Named classical baselines
(scipy FAQ/2-opt QAP, LPT, LP relaxation) and a from-scratch MCTS search track that uses
the differentiable relaxation as an AlphaZero-style prior. Bit-identical torch + jax
twins.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Each family is a QUBO-form energy on `omnibias-qubo`. Temperature collapse
(`beta -> inf`) anneals the relaxation. Gaps are spectral / SOS sandwiches.

| You want | Import |
| --- | --- |
| QAP / GAP / scheduling | `omnibias.nphard` |
| MCTS track | `omnibias.nphard` search |

## Extend

- Source: [`packages/omnibias-nphard`](../../../packages/omnibias-nphard).
- Namespace: `omnibias.nphard`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-nphard/tests -q`.
- Compose with `omnibias-qubo`, `omnibias-discrete`, `omnibias-routing` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A named QAP instance whose decoded heuristic, SOS lower bound, and scipy FAQ baseline
are reported in one `gates` JSON with skill > 0.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
