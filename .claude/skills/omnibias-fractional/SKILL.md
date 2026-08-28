---
name: omnibias-fractional
description: Compute fractional derivatives in two labelled classes: grid/spectral GL/RL/Caputo/FFT operators, and a closed-form analytic gamma-ratio Taylor jet exact for polynomials of degree <= N. Use when inventing a fractional PINN residual or a differentiable-order kernel.
---

# omnibias-fractional

Fractional calculus for omnibias in two honest classes: (1) grid/spectral
Grunwald-Letnikov / Riemann-Liouville / Caputo / FFT operators -- non-local NUMERICAL
approximations (accuracy set by the grid), NOT closed form; and (2) a closed-form
analytic fractional derivative on the analytic-function class (gamma-ratio Taylor-jet
series with differentiable order; an order-N truncation, exact for polynomials of degree
<= N).

## Why nested AD fails

Nested AD has no Caputo or Riemann-Liouville operator. Grid schemes without an exact
polynomial class cannot report when the answer is exact. Generic fractional libraries do
not share the omnibias jet.

## What only this tower unlocks

Fractional calculus for omnibias in two honest classes: (1) grid/spectral
Grunwald-Letnikov / Riemann-Liouville / Caputo / FFT operators -- non-local NUMERICAL
approximations (accuracy set by the grid), NOT closed form; and (2) a closed-form
analytic fractional derivative on the analytic-function class (gamma-ratio Taylor-jet
series with differentiable order; an order-N truncation, exact for polynomials of degree
<= N).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Label the path: grid/spectral schemes are numerical (accuracy set by the
grid); the analytic gamma-ratio series is closed-form on polynomials of
degree <= N.

| You want | Import |
| --- | --- |
| GL / RL / Caputo / spectral | `omnibias.fractional` |
| Analytic fractional jet | `omnibias.fractional` analytic class |

## Extend

- Source: [`packages/omnibias-fractional`](../../../packages/omnibias-fractional).
- Namespace: `omnibias.fractional`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-fractional/tests -q`.
- Compose with `omnibias-fields`, `omnibias-pinn`, `omnibias-core` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A differentiable-order Caputo residual on a neural field whose polynomial probe is
bit-identical to the closed-form gamma-ratio jet.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
