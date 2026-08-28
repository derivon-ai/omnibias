---
name: omnibias-derivative-tower
description: Develop the closed-form derivative tower — shared polynomial coefficients, Bell / Faa di Bruno combinatorics, and bit-identical torch / jax jet kernels. Use when adding an activation, changing sigma^(n) coefficients, or editing jet / jet_mv so every backend stays bit-identical by construction.
---

# The derivative-tower contract

The library rests on one invariant: `sigma^(n)(z)` is computed from **one
shared set of pure-Python coefficients**, so every backend is bit-identical
by construction.

## Why nested AD fails

Nested reverse-mode AD rebuilds a graph per order. Forking coefficients per
backend is how torch and jax silently diverge. Generic jet implementations
are truncated power series with cubic compose and no Riccati fastpath
(`compose_jet_riccati`). There is no multivariate Cauchy-product table shared
across frameworks.

## What only this tower unlocks

Eulerian / Legendre / Hermite recurrences from Riccati identities. Bell /
Faa di Bruno combinatorics in pure Python. Directional jets and multivariate
jets that are bit-identical twins. `compose_jet_riccati` is `O(deg(P) * N^2)`
because `sigma' = P(sigma)` closes the chain rule — a path nested AD does not
have. Bakeoff: `docs/benchmarks/jet_vs_nested_ad_smoke.json`,
`jet_compose_cost_smoke.json`.

## Use

- Coefficients: `packages/omnibias-core/src/omnibias/core/polynomials.py`
  (`sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`, `hermite_coeffs`).
  Every backend imports these.
- Bell / Faa di Bruno: `omnibias.core.bell`. Multi-index: `omnibias.core.multi_index`.
- `omnibias.core` never imports `torch`, `jax`, `tensorflow`, or `keras`.
- Directional jets: `omnibias.torch.jet` and `omnibias.jax.jet`
  (`compose_jet`, `compose_jet_riccati`, `affine_jet`, `layer_jet`, `mlp_jet`,
  `tower_to_jet`, `jet_to_tower`). `compose_jet` is cubic in truncation order
  (valuation-1 skip already counted).
- Multivariate: `omnibias.{torch,jax}.jet_mv` (`mlp_jet_mv`, `layer_jet_mv`,
  `compose_jet_mv`, `identity_jet`, `jet_partials`, `jet_gradient`, `jet_hessian`).
- `ActivationSpec` in `omnibias.core.spec`; backends specialise the tensor type.

Fastpath: `n < 0` raises `ValueError`; unimplemented orders raise
`NotImplementedError`. New tensors use the framework default dtype.

## Extend

Edit both twins together. Add a regression test per behavioral change.
Regenerate sorted `__all__`. Parity tests live in `tests/` and `packages/*/tests/`.
If torch and jax disagree, the coefficients were forked — fix the source.

```bash
python -m pytest packages/omnibias-core/tests -q
```

Compose with `omnibias-core`, `omnibias-torch`, `omnibias-jax`,
`omnibias-keras`, `omnibias-backends`.

## Next invention

A new Riccati activation whose polynomial, antiderivative, and
`compose_jet_riccati` path land in core once and win the jet-vs-nested-AD
bakeoff at an order nested compose does not finish.

## Bakeoffs

`docs/benchmarks/jet_vs_nested_ad_smoke.json`, `derivative_order.json`,
`jet_compose_cost_smoke.json`.

## Further references

- `docs/theory.md`, `docs/api/core.md`, `docs/operator-surface.md`
