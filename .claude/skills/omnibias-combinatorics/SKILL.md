---
name: omnibias-combinatorics
description: Build exact differentiable matching, flow, and matroid layers with entropic relaxations onto integral polytopes and a tight LP-dual gap certificate. Use when inventing a new polytope layer or a certified assignment / transportation / matroid front-end.
---

# omnibias-combinatorics

Exact differentiable matching / flow / matroid layers for omnibias: entropic (Sinkhorn)
relaxations onto the assignment / transportation / flow / matroid polytopes that anneal
beta -> inf to a polytope vertex, decoded to a feasible solution, and sandwiched by a
tight LP-dual optimality-gap certificate (Neumaier-Shcherbina, tight because these
polytopes are integral). Best-in-class classical baselines (Hungarian / LP / max-flow /
greedy). Bit-identical torch + jax twins.

## Why nested AD fails

Nested AD through a discrete matching is a permutation with no useful gradient.
Relaxations that are not on an integral polytope cannot seal a tight dual gap; generic
Sinkhorn has no certificate.

## What only this tower unlocks

Exact differentiable matching / flow / matroid layers for omnibias: entropic (Sinkhorn)
relaxations onto the assignment / transportation / flow / matroid polytopes that anneal
beta -> inf to a polytope vertex, decoded to a feasible solution, and sandwiched by a
tight LP-dual optimality-gap certificate (Neumaier-Shcherbina, tight because these
polytopes are integral). Best-in-class classical baselines (Hungarian / LP / max-flow /
greedy). Bit-identical torch + jax twins.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Entropic layers anneal with temperature collapse (`beta -> inf`) onto a
polytope vertex, then decode and sandwich with Neumaier-Shcherbina.

| You want | Import |
| --- | --- |
| Assignment / transport / flow | `omnibias.combinatorics` |
| Unsplittable-flow replay | `omnibias.combinatorics.unsplittable` |
| Ramsey / extremal predicates | `omnibias.combinatorics.{ramsey,extremal}` |

## Extend

- Source: [`packages/omnibias-combinatorics`](../../../packages/omnibias-combinatorics).
- Namespace: `omnibias.combinatorics`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-combinatorics/tests -q`.
- Compose with `omnibias-discrete`, `omnibias-graph`, `omnibias-convex`, `omnibias-frontier` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A new integral polytope (laminar / gammoid) whose dual gap is tight by construction and
whose torch/jax twins stay bit-identical at float64.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
