---
name: omnibias-core
description: Extend the pure-Python derivative tower: Eulerian / Legendre / Hermite polynomials, ActivationSpec, Bell / Faa di Bruno combinatorics, and the verified / proof registers. Use when adding an activation, a collapse, or a finite obligation that every backend must share bit-identically.
---

# omnibias-core

Numerically-stable closed-form n-th derivative forward-pass framework: pure-Python core
with polynomial coefficient generators (Eulerian / Legendre / Hermite) and the
backend-agnostic ActivationSpec protocol.

## Why nested AD fails

Nested AD rebuilds a new computational graph for every extra derivative order.
Coefficients forked per backend diverge. There is no shared ActivationSpec, no Bell
table, and no Interval/Taylor-model register inside a generic autodiff core.

## What only this tower unlocks

Numerically-stable closed-form n-th derivative forward-pass framework: pure-Python core
with polynomial coefficient generators (Eulerian / Legendre / Hermite) and the
backend-agnostic ActivationSpec protocol.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

`omnibias.core` never imports torch, jax, tensorflow, or keras.

| You want | Import |
| --- | --- |
| Polynomial coefficients | `omnibias.core.polynomials` (`sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`, `hermite_coeffs`) |
| Activation metadata | `omnibias.core.spec` (`ActivationSpec`, including `integral`) |
| Bell / Faa di Bruno | `omnibias.core.bell` |
| Multi-index jets | `omnibias.core.multi_index` |
| Multipack / scan | `omnibias.core.multipack`, `omnibias.core.scan` |
| Verified register | `omnibias.core.verified` |
| Proof engine | `omnibias.core.proof` |

Every backend imports these coefficients. Forking them is the most
expensive mistake in the workspace.

## Extend

- Source: [`packages/omnibias-core`](../../../packages/omnibias-core).
- Namespace: `omnibias.core`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-core/tests -q`.
- Compose with `omnibias-derivative-tower`, `omnibias-torch`, `omnibias-jax`, `omnibias-backends` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A new Riccati activation whose polynomial coefficients, antiderivative kernel, and
Interval tower land in core first, then light up every backend without a second
implementation.


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
