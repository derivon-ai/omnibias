---
name: omnibias-qubo
description: Optimize QUBO / Ising energies with an annealed sigmoid relaxation, rounding / 1-flip decode, brute-force oracle, and certified optimality-gap sandwich. Use when inventing a new quadratic Boolean front-end (max-cut, independent set) on the discrete substrate.
---

# omnibias-qubo

Differentiable + certified quadratic Boolean optimization for omnibias: an annealed
sigmoid relaxation of QUBO / Ising energies solved differentiably by the
temperature-collapse (beta -> inf) penalty, a rounding + 1-flip local-search decoder, a
brute-force oracle, and a rigorous optimality-gap certificate (cheap spectral seal or a
headline SOS / Lasserre lower bound over the Boolean hypercube), plus max-cut and
max-independent-set front-ends (jax + torch).

## Why nested AD fails

Nested AD through a binary quadratic is a combinatorial search. Generic QUBO libraries
return a heuristic energy with no SOS / spectral seal and no bit-identical torch/jax
twins.

## What only this tower unlocks

Differentiable + certified quadratic Boolean optimization for omnibias: an annealed
sigmoid relaxation of QUBO / Ising energies solved differentiably by the
temperature-collapse (beta -> inf) penalty, a rounding + 1-flip local-search decoder, a
brute-force oracle, and a rigorous optimality-gap certificate (cheap spectral seal or a
headline SOS / Lasserre lower bound over the Boolean hypercube), plus max-cut and
max-independent-set front-ends (jax + torch).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Temperature collapse (`beta -> inf`) anneals the sigmoid relaxation.
Builds on `omnibias-discrete`.

| You want | Import |
| --- | --- |
| Anneal / decode / certify | `omnibias.qubo` |
| max_cut / max_independent_set | `omnibias.qubo` |

## Extend

- Source: [`packages/omnibias-qubo`](../../../packages/omnibias-qubo).
- Namespace: `omnibias.qubo`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-qubo/tests -q`.
- Compose with `omnibias-discrete`, `omnibias-sos`, `omnibias-convex`, `omnibias-nphard` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A max-cut instance whose SOS lower bound, spectral seal, and 1-flip decoder sandwich the
brute-force optimum with a published `gates` block.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
