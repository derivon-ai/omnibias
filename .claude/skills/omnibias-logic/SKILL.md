---
name: omnibias-logic
description: Differentiate certified Boolean logic: weighted MaxSAT plus (weighted) #SAT / model counting with inclusion-exclusion count enclosures. Use when inventing a new CNF consumer or a sealed model-count sandwich.
---

# omnibias-logic

Differentiable + certified Boolean logic for omnibias: re-exports the weighted-MaxSAT
DiscreteProblem consumer (annealed sigmoid relaxation + rounding / 1-flip decoder +
Lasserre / SOS certified optimality gap) and adds (weighted) #SAT / model counting with
rigorous inclusion-exclusion lower+upper count enclosures and a beta->inf annealed
model-finder relaxation (torch + jax twins). yes-if: a certified count enclosure, never
a P = NP / #P exactness claim.

## Why nested AD fails

Nested AD through a SAT solver is a discrete search. Approximate model counters return a
float, not an Interval sandwich. Generic MaxSAT relaxations have no SOS gap.

## What only this tower unlocks

Differentiable + certified Boolean logic for omnibias: re-exports the weighted-MaxSAT
DiscreteProblem consumer (annealed sigmoid relaxation + rounding / 1-flip decoder +
Lasserre / SOS certified optimality gap) and adds (weighted) #SAT / model counting with
rigorous inclusion-exclusion lower+upper count enclosures and a beta->inf annealed
model-finder relaxation (torch + jax twins). yes-if: a certified count enclosure, never
a P = NP / #P exactness claim.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Reuses the `omnibias.discrete.maxsat` DiscreteProblem. Temperature collapse
(`beta -> inf`) anneals the model-finder.

| You want | Import |
| --- | --- |
| Weighted MaxSAT | `omnibias.logic` (re-export of the discrete consumer) |
| #SAT enclosures | `omnibias.logic` inclusion-exclusion |

## Extend

- Source: [`packages/omnibias-logic`](../../../packages/omnibias-logic).
- Namespace: `omnibias.logic`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-logic/tests -q`.
- Compose with `omnibias-discrete`, `omnibias-boolean`, `omnibias-sos` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A weighted #SAT instance whose inclusion-exclusion enclosure contains the brute-force
count and whose annealed model-finder recovers a certified optimum.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
