---
name: omnibias-binary
description: Train and extend binary, ternary, and k-bit networks with closed-form tanh-beta Riccati backward passes on torch and jax. Use when quantizing an omnibias model, wiring a STE replacement, or inventing a new bit-width with an exact surrogate tower.
---

# omnibias-binary

Closed-form quantization gradients for binary/ternary/k-bit neural-network training via
the omnibias tanh-beta Riccati derivative (torch + jax).

## Why nested AD fails

Straight-through estimators freeze the backward pass at a hand-chosen kink. Nested AD
through a hard quantizer either vanishes or explodes; there is no honest n-th derivative
of a step. High-order training and certified rounding gaps are out of reach of generic
STE graphs.

## What only this tower unlocks

Closed-form quantization gradients for binary/ternary/k-bit neural-network training via
the omnibias tanh-beta Riccati derivative (torch + jax).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Closed-form quantization gradients live in `omnibias.binary` (torch + jax
twins). The backward is the tanh-beta Riccati tower, so every order is
`sigma^(n)` rather than a clipped identity.

| You want | Import |
| --- | --- |
| Binary / ternary STE replacement | `omnibias.binary` |
| Bit-identical jax twin | `omnibias.binary.jax` |

Pair the forward clip with the Riccati backward; temperature collapse
(`beta -> inf`) hardens the gate, while the tower stays exact at every
finite beta.

## Extend

- Source: [`packages/omnibias-binary`](../../../packages/omnibias-binary).
- Namespace: `omnibias.binary`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-binary/tests -q`.
- Compose with `omnibias-boolean`, `omnibias-torch`, `omnibias-jax`, `omnibias-curvature` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A k-bit pack whose rounding gap is a sealed Interval sandwich, trained end-to-end with
CubicNewton on the exact quantization Hessian.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
