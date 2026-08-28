---
name: omnibias-partition
description: Build a certified soft partition-of-unity: oblique split gates that route into 2**depth regions, harden under temperature collapse, and drive PINN / geometry / symbolic / tab bridges from one combine() engine. Use when inventing a new region model or a sealed soft-to-hard membership gap.
---

# omnibias-partition

A light, certified soft partition-of-unity primitive: oblique / axis-aligned / sparse
soft-split gates that route an input into 2**depth regions with weights that are
non-negative and sum to one, harden to a crisp partition under temperature collapse (the
beta -> inf penalty), carry a SOUND soft->hard membership-gap certificate
(outward-rounded Interval + closed-form Gibbs bound), and expose a per-region model
registry whose one combine() engine (sum_l w_l * out_l) is shared by every downstream
bridge. Bit-identical numpy / torch / jax weights.

## Why nested AD fails

Hard trees have no gradient. Soft trees without a partition-of-unity constraint leak
mass. Nested AD cannot enclose the membership gap as an Interval plus
log(n_regions)/beta.

## What only this tower unlocks

A light, certified soft partition-of-unity primitive: oblique / axis-aligned / sparse
soft-split gates that route an input into 2**depth regions with weights that are
non-negative and sum to one, harden to a crisp partition under temperature collapse (the
beta -> inf penalty), carry a SOUND soft->hard membership-gap certificate
(outward-rounded Interval + closed-form Gibbs bound), and expose a per-region model
registry whose one combine() engine (sum_l w_l * out_l) is shared by every downstream
bridge. Bit-identical numpy / torch / jax weights.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Temperature collapse (`beta -> inf`) hardens weights to a crisp partition.
That is the feasibility axis, distinct from founding `delta -> 0`.

| You want | Import |
| --- | --- |
| Weights / hard assignment | `omnibias.partition` |
| RegionModels combine() | `omnibias.partition` |
| Arrangement | `omnibias.partition.arrangement` |

## Extend

- Source: [`packages/omnibias-partition`](../../../packages/omnibias-partition).
- Namespace: `omnibias.partition`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-partition/tests -q`.
- Compose with `omnibias-pinn`, `omnibias-geometry`, `omnibias-symbolic`, `omnibias-tab` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A depth-d arrangement whose Interval membership enclosure and log(n_regions)/beta gap
predict the observed soft-to-hard error on a named piecewise PDE.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
