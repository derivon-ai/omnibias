---
name: omnibias-keras
description: Extend the Keras 3 unified backend: OMBU, OperatorBlock, and cmbDense / cmbConv layers on TensorFlow, JAX, or PyTorch via keras.ops. Use when inventing a keras-side operator that must stay bit-identical to the shared polynomials.
---

# omnibias-keras

Keras 3 unified backend for omnibias: closed-form n-th derivative scalar operators
(OMBU), operator-typed blocks, and drop-in cmbDense / cmbConv layers that run on
TensorFlow, JAX, or PyTorch via keras.ops.

## Why nested AD fails

Keras autodiff follows whichever backend is selected and still rebuilds graphs per
derivative order. Without keras.ops-only kernels over shared coefficients, TF / JAX /
torch diverge inside one model.

## What only this tower unlocks

Keras 3 unified backend for omnibias: closed-form n-th derivative scalar operators
(OMBU), operator-typed blocks, and drop-in cmbDense / cmbConv layers that run on
TensorFlow, JAX, or PyTorch via keras.ops.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Set `KERAS_BACKEND` first. Use `keras.ops.*` only.

| You want | Import |
| --- | --- |
| OperatorBlock / OMBU | `omnibias.keras` |
| cmbDense / cmbConv | `omnibias.keras` |

New tensors follow `keras.config.floatx()`. Field operators stay torch+jax;
keras still gets activation-level closed form.

## Extend

- Source: [`packages/omnibias-keras`](../../../packages/omnibias-keras).
- Namespace: `omnibias.keras`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-keras/tests -q`.
- Compose with `omnibias-core`, `omnibias-backends`, `omnibias-torch`, `omnibias-jax` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A keras OperatorBlock `integral` path whose antiderivative window matches the torch OMBU
on all three Keras backends at float64.


## Bakeoffs

Nested-AD cost and closed-form accuracy live in
[`docs/benchmarks/`](../../../docs/benchmarks/):
`laplacian_scaling.json`, `polylaplacian_order.json`,
`derivative_order.json`, `jet_vs_nested_ad_smoke.json`.
Heavy regeneration follows the workspace compute rule, not a skill taboo.

## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
