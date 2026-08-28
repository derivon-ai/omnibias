---
name: omnibias-graph
description: Differentiate spectral graph operators (Laplacians, embeddings, heat kernels) and combinatorial relaxations (Gumbel-Sinkhorn, SoftSort, soft top-k). Use when inventing a new spectral layer or a certified arrangement on a graph.
---

# omnibias-graph

Differentiable spectral graph operators (Laplacians, spectral embedding, heat kernel)
and continuous combinatorial relaxations (Gumbel-Sinkhorn, SoftSort, soft top-k) with
torch + jax bit-parity.

## Why nested AD fails

Nested AD through an eigendecomposition is fragile and backend-split. SoftSort /
Sinkhorn without a closed-form sigmoid tower cannot expose exact Hessians or a
membership gap.

## What only this tower unlocks

Differentiable spectral graph operators (Laplacians, spectral embedding, heat kernel)
and continuous combinatorial relaxations (Gumbel-Sinkhorn, SoftSort, soft top-k) with
torch + jax bit-parity.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

| You want | Import |
| --- | --- |
| Spectral Laplacian / heat kernel | `omnibias.graph` |
| Gumbel-Sinkhorn / SoftSort | `omnibias.graph` |
| Arrangement on graphs | `omnibias.graph.arrangement` |

Cell membership on an arrangement is temperature collapse (`beta -> inf`).

## Extend

- Source: [`packages/omnibias-graph`](../../../packages/omnibias-graph).
- Namespace: `omnibias.graph`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-graph/tests -q`.
- Compose with `omnibias-combinatorics`, `omnibias-partition`, `omnibias-fields` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A heat-kernel layer whose closed-form time derivative matches the spectral expansion and
whose torch/jax twins stay within 1e-12 in float64.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
