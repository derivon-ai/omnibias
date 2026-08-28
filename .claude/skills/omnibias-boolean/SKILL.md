---
name: omnibias-boolean
description: Differentiate Boolean algebra with exact ANF/Reed-Muller and Walsh spectra, Boolean differential calculus, and a beta-annealed soft-gate solver. Use when inventing a differentiable circuit, reproductive equation solver, or S-box analysis on the closed-form tower.
---

# omnibias-boolean

Differentiable Boolean algebra: exact ANF/Reed-Muller and Walsh spectra, Boolean
differential calculus, reproductive equation solving, and a beta-annealed soft-gate
system solver built on the omnibias closed-form derivative towers (torch + jax).

## Why nested AD fails

Boolean maps are piecewise constant; nested AD through AND/XOR is zero almost
everywhere. Spectral and ANF identities need exact polynomial arithmetic over GF(2),
which autodiff graphs do not carry.

## What only this tower unlocks

Differentiable Boolean algebra: exact ANF/Reed-Muller and Walsh spectra, Boolean
differential calculus, reproductive equation solving, and a beta-annealed soft-gate
system solver built on the omnibias closed-form derivative towers (torch + jax).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Exact algebra lives in `omnibias.boolean._core`; torch/jax twins anneal
soft gates under temperature collapse (`beta -> inf`).

| You want | Import |
| --- | --- |
| ANF / Walsh spectra | `omnibias.boolean` |
| Soft-gate solver | `omnibias.boolean.{torch,jax}` |
| Verified spectra / S-boxes | `omnibias.boolean.{_core.verified,cipher}` |

## Extend

- Source: [`packages/omnibias-boolean`](../../../packages/omnibias-boolean).
- Namespace: `omnibias.boolean`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-boolean/tests -q`.
- Compose with `omnibias-binary`, `omnibias-logic`, `omnibias-discrete` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A reproductive-equation eliminant that returns a sealed GF(2) witness and a
differentiable soft-gate path whose hardened assignment matches the witness bit-for-bit.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
