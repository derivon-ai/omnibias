---
name: omnibias-jax
description: Extend the JAX backend: closed-form sigma^(n) kernels, neural-field Laplacian / Hessian, directional and multivariate jets, and Scan-Net / Jet-KAN architectures. Use when inventing a jax-side primitive that must stay bit-identical to torch.
---

# omnibias-jax

JAX backend for omnibias: closed-form n-th derivative activation kernels (sigmoid via
Eulerian polynomials, tanh via Legendre, Gaussian via Hermite), neural-field Laplacian /
Hessian primitives, and Born-Oppenheimer derivative tools for variational quantum Monte
Carlo.

## Why nested AD fails

Nested AD in JAX still scales with order and tracing constraints. Without shared
polynomials from `omnibias.core.polynomials`, jax and torch diverge. High-order field
Laplacians are the FermiNet bottleneck.

## What only this tower unlocks

JAX backend for omnibias: closed-form n-th derivative activation kernels (sigmoid via
Eulerian polynomials, tanh via Legendre, Gaussian via Hermite), neural-field Laplacian /
Hessian primitives, and Born-Oppenheimer derivative tools for variational quantum Monte
Carlo.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Import coefficients from core; never reimplement them.

| You want | Import |
| --- | --- |
| Activation kernels / registry | `omnibias.jax` |
| Field Laplacian / Hessian | `omnibias.jax` (`neural_field_laplacian`, `neural_field_hessian`) |
| Directional / MV jets | `omnibias.jax.jet`, `omnibias.jax.jet_mv` |
| Architectures | `omnibias.jax.architectures` |
| L-infinity minimax step | `omnibias.jax.optim.linf_minimax_step` (toy residual; not a CCF champion retrain) |

Default dtype follows JAX config; use float64 for parity.

## Extend

- Source: [`packages/omnibias-jax`](../../../packages/omnibias-jax).
- Namespace: `omnibias.jax`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-jax/tests -q`.
- Compose with `omnibias-core`, `omnibias-torch`, `omnibias-backends`, `omnibias-derivative-tower` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A new jax architecture cell whose model jet is exact at order N and whose torch twin
matches bit-for-bit on the jet_vs_nested_ad bakeoff.


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
