---
name: omnibias-difference
description: Read certified finite-difference to derivative extraction, umbral / Sheffer calculus, and Stirling / Bernoulli / Euler numbers off the closed-form tower. Use when inventing an irregular stencil, a singularity tracker, or an exact-Q Birkhoff weight.
---

# omnibias-difference

The founding delta->0 (multi-bias collapse) register: certified finite-difference ->
derivative extraction, umbral / Sheffer sequence calculus, and asymptotic-coefficient
reading (Stirling / Bernoulli / Euler numbers) read straight off the closed-form
omnibias towers. Pure-Python core + rigorous interval-tower certificates from
omnibias.core.verified, with bit-identical torch/jax twins for the finite-difference
stencil operator.

## Why nested AD fails

Finite-difference tables on nested AD suffer 1/h^{n} cancellation. Generic stencils are
floating-point fits with no poisedness certificate over Q. Umbral operators have no home
in an autodiff graph.

## What only this tower unlocks

The founding delta->0 (multi-bias collapse) register: certified finite-difference ->
derivative extraction, umbral / Sheffer sequence calculus, and asymptotic-coefficient
reading (Stirling / Bernoulli / Euler numbers) read straight off the closed-form
omnibias towers. Pure-Python core + rigorous interval-tower certificates from
omnibias.core.verified, with bit-identical torch/jax twins for the finite-difference
stencil operator.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

| You want | Import |
| --- | --- |
| Irregular / Birkhoff stencils | `omnibias.difference` (`solve_irregular_stencil`, `is_poised_exact`) |
| Singularity tracking | `omnibias.difference.singularity` |
| Umbral / Sheffer | `omnibias.difference` core |

The founding `delta -> 0` register is this package: the multi-bias unit
becomes `sigma^(K-1)` exactly.

## Extend

- Source: [`packages/omnibias-difference`](../../../packages/omnibias-difference).
- Namespace: `omnibias.difference`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-difference/tests -q`.
- Compose with `omnibias-core`, `omnibias-holonomic`, `omnibias-symbolic`, `omnibias-qcalculus` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A poised irregular stencil whose rational C_j Lean obligation and certified |x_s|
annulus are sealed in one `prove()` call.


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
