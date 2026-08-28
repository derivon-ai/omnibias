---
name: omnibias-routing
description: Solve certified differentiable routing: poly-size TSP relaxations, 2-opt decode, Neumaier-Shcherbina LP gaps, and decision-focused predict-then-optimize. Use when inventing a new tour layer or a sealed routing gap.
---

# omnibias-routing

Certified + differentiable combinatorial routing for omnibias: a poly-size Held-Karp /
single-commodity-flow / assignment TSP relaxation solved differentiably by the
temperature-collapse penalty, a heatmap + 2-opt tour decoder, and a rigorous
optimality-gap certificate (Neumaier-Shcherbina LP lower bound vs a decoded tour), plus
decision-focused predict-then-optimize routing (jax + torch).

## Why nested AD fails

Nested AD through a permutation tour is zero almost everywhere. Heatmap relaxations
without an LP dual cannot seal a gap. Generic predict-then-optimize stacks have no 2-opt
certificate.

## What only this tower unlocks

Certified + differentiable combinatorial routing for omnibias: a poly-size Held-Karp /
single-commodity-flow / assignment TSP relaxation solved differentiably by the
temperature-collapse penalty, a heatmap + 2-opt tour decoder, and a rigorous
optimality-gap certificate (Neumaier-Shcherbina LP lower bound vs a decoded tour), plus
decision-focused predict-then-optimize routing (jax + torch).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Temperature collapse (`beta -> inf`) hardens the assignment.

| You want | Import |
| --- | --- |
| TSP relaxation + 2-opt | `omnibias.routing` |
| Predict-then-optimize | `omnibias.routing` |

## Extend

- Source: [`packages/omnibias-routing`](../../../packages/omnibias-routing).
- Namespace: `omnibias.routing`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-routing/tests -q`.
- Compose with `omnibias-convex`, `omnibias-combinatorics`, `omnibias-discrete` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A decision-focused TSP whose LP gap and 2-opt tour beat a named heatmap baseline on a
public instance with skill > 0.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
