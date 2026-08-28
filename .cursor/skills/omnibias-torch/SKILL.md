---
name: omnibias-torch
description: Extend the PyTorch backend: OMBU, OperatorBlock (six roles including closed-form integral), jets, Scan-Net / Jet-KAN, and curvature-aware optimizers. Use when inventing a torch-side primitive that must stay bit-identical to jax.
---

# omnibias-torch

PyTorch backend for omnibias: trainable scalar operators (OMBU), operator-typed blocks,
closed-form activation derivative kernels, and reference PINN / CmbNet / CvxLayer
architectures.

## Why nested AD fails

torch.autograd.grad loops rebuild the graph per order and OOM on high-order PINN /
FermiNet Laplacians. There is no shared polynomial source, no OperatorBlock integral,
and no bit-identical jax twin in vanilla PyTorch.

## What only this tower unlocks

PyTorch backend for omnibias: trainable scalar operators (OMBU), operator-typed blocks,
closed-form activation derivative kernels, and reference PINN / CmbNet / CvxLayer
architectures.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Coefficients come from `omnibias.core.polynomials`. Default dtype is
`torch.get_default_dtype()`, never a hardcoded float32.

| You want | Import |
| --- | --- |
| OperatorBlock / OMBU | `omnibias.torch` (`op` in identity/grad/laplacian/derivative/band/integral) |
| Jets | `omnibias.torch.jet`, `omnibias.torch.jet_mv` |
| Optimizers | `omnibias.torch.optim` |
| Architectures | `omnibias.torch.architectures` |

`n < 0` raises `ValueError`; unimplemented orders raise `NotImplementedError`.

## Extend

- Source: [`packages/omnibias-torch`](../../../packages/omnibias-torch).
- Namespace: `omnibias.torch`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-torch/tests -q`.
- Compose with `omnibias-core`, `omnibias-jax`, `omnibias-backends`, `omnibias-derivative-tower` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A new OperatorBlock consumer whose integral and derivative roles share one sigma
evaluation and match jax bit-for-bit on derivative_order.json.


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
