---
name: omnibias-shape
description: Differentiate soft occupancy fields and soft-coverage (soft-OR / log-sum-exp union) operators with a closed-form tower, plus morphology and topology. Use when inventing a new occupancy primitive or a soft Euler characteristic.
---

# omnibias-shape

Differentiable soft shape / occupancy fields and soft-coverage (soft-OR / log-sum-exp
union) operators with a closed-form derivative tower, torch + jax bit-parity.

## Why nested AD fails

Hard occupancy has no gradient. Nested AD through min/max CSG is a kink. Soft-OR without
a closed-form logsumexp_beta tower cannot expose exact Hessians or a morphology gap.

## What only this tower unlocks

Differentiable soft shape / occupancy fields and soft-coverage (soft-OR / log-sum-exp
union) operators with a closed-form derivative tower, torch + jax bit-parity.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Temperature collapse (`beta -> inf`) hardens soft-OR / morphology.
This is not a seventh OperatorBlock role.

| You want | Import |
| --- | --- |
| Soft occupancy / coverage | `omnibias.shape` |
| Morphology | `omnibias.shape.morphology` |
| Soft Euler / Morse | `omnibias.shape.topology` |

## Extend

- Source: [`packages/omnibias-shape`](../../../packages/omnibias-shape).
- Namespace: `omnibias.shape`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-shape/tests -q`.
- Compose with `omnibias-fields`, `omnibias-pinn`, `omnibias-geometry` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A CSG occupancy whose closed-form gradient matches a finite-difference probe and whose
soft Euler count separates components on a named 1-D Morse example.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
