---
name: omnibias-backends
description: Compute closed-form n-th derivatives of activations and exact Taylor jets with omnibias on PyTorch, JAX, or Keras 3. Use when taking high-order derivatives, building OperatorBlock / OMBU layers, propagating directional or multivariate jets, or when the user mentions sigma^(n), closed-form derivatives, jets, or bit-identical backends.
---

# Closed-form derivative tower and jets

omnibias computes `sigma^(n)(z)` in closed form for any order `n` with a single
`sigma` evaluation. Every backend imports coefficients from
`omnibias.core.polynomials`, so PyTorch, JAX, and Keras 3 are **bit-identical
by construction**.

## Why nested AD fails

Nested reverse-mode AD rebuilds a new computational graph for every extra
derivative order. At the orders PINNs, FermiNet Laplacians, and Taylor jets
need, that graph is the bottleneck: memory scales with order, mixed partials
require a combinatorial explosion of sweeps, and backends diverge as soon as
anyone forks a kernel. Generic tools give you `grad`; they do not give you
`sigma^(n)` from one evaluation, a closed-form integral window, or a jet that
is the same bit pattern on three frameworks.

## What only this tower unlocks

Riccati identities (`sigmoid' = s(1-s)`, `tanh' = 1-t^2`) plus Eulerian /
Legendre / Hermite recurrences yield every order from one `sigma`. OperatorBlock
dispatches six roles, including a **closed-form integral**. Directional jets
(`mlp_jet`) and multivariate jets (`mlp_jet_mv`) propagate exact truncated
Taylor series through deep compositions. Nested AD cannot sustain that
workload at the orders in `docs/benchmarks/derivative_order.json` and
`docs/benchmarks/jet_vs_nested_ad_smoke.json`.

## Use

| You want | Import from | Key entry points |
| --- | --- | --- |
| Trainable operator layers (PyTorch) | `omnibias.torch` | `OperatorBlock`, `OperatorMultiBiasUnit` (`OMBU`), `cmbLinear`, `cmbConv1d`, `cmbConv2d` |
| Heterogeneous Birkhoff packs | `omnibias.torch` / `omnibias.jax` / `omnibias.core` | `MultiPackUnit`, `init_multipack` / `multipack_apply`, `MultiPackSpec` |
| Transverse bias scan | `omnibias.torch` / `omnibias.jax` / `omnibias.core` | `BiasScan`, `init_bias_scan` / `bias_scan`, `BankSpec` |
| Exact-Q irregular stencils | `omnibias.difference` | `solve_irregular_stencil`, `is_poised_exact`, `certified_irregular_error` |
| Scan-Net / Jet-KAN / LadderNet | `omnibias.torch.architectures` / `omnibias.jax.architectures` | `ScanNet`, `JetKAN`, `HermiteBasis` / `LadderNet` |
| Equivariant / hierarchical scan | `omnibias.{torch,jax}.scan_equivariant`, `.hierarchy` | `EquivariantScan`, `hierarchical_scan` |
| Activation registry | `omnibias.torch` / `omnibias.jax` | `get_activation`, `list_activations`, `register_activation` |
| Closed-form field Laplacian (JAX) | `omnibias.jax` | `neural_field_value`, `neural_field_laplacian`, `neural_field_hessian` |
| Directional Taylor jets | `omnibias.torch` / `omnibias.jax` | `mlp_jet`, `layer_jet`, `compose_jet`, `tower_to_jet` |
| Multivariate jets to order N | `omnibias.torch` / `omnibias.jax` | `mlp_jet_mv`, `jet_partials`, `jet_gradient`, `jet_hessian` |
| Raw polynomial coefficients | `omnibias.core` | `sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`, `hermite_coeffs` |

`OperatorBlock` dispatches on `op="identity"|"grad"|"laplacian"|"derivative"|"band"|"integral"`:

- `identity` (K=1): `sigma(z + b)`.
- `grad` / `laplacian` / `derivative`: closed-form `sigma^(n)` (`n = 1, 2`, or arbitrary).
- `band` (K=2): `sigma(z + b_hi) - sigma(z + b_lo)`.
- `integral` (K=2): antiderivative window `S(z + b_hi) - S(z + b_lo)` with `S' = sigma`.

Three senses of "integral": (1) this antiderivative window; (2) domain quadrature
in `omnibias.fields`; (3) `int f dmu` in `omnibias.measure`. Canonical matrix:
`docs/operator-surface.md`.

Soft-argmax `gamma -> inf` on a scan is temperature collapse; founding bias
collapse is `delta -> 0` on K parallel hyperplanes.

Runnable examples: `docs/examples/quickstart_torch.py`, `quickstart_jax.py`,
`quickstart_keras.py` (set `KERAS_BACKEND` first).

New tensors follow `torch.get_default_dtype()` / `keras.config.floatx()`.
`n < 0` raises `ValueError`; unimplemented orders raise `NotImplementedError`.
A jet carries partials up to the order you request.

## Extend

- Polynomials: `omnibias.core.polynomials` only. Pair with `omnibias-derivative-tower`.
- Torch twin: `omnibias-torch`. JAX twin: `omnibias-jax`. Keras: `omnibias-keras`.
- Tests: `python -m pytest packages/omnibias-core/tests packages/omnibias-torch/tests packages/omnibias-jax/tests -q` and root `tests/` parity.
- Compose with `omnibias-fields`, `omnibias-curvature`, `omnibias-verify` by those names.

## Next invention

A new Riccati activation whose coefficients, antiderivative kernel, and jet
fastpath land in core once and light up OperatorBlock `derivative` + `integral`
on all three backends, then win `derivative_order.json` against nested AD.

## Bakeoffs

[`docs/benchmarks/`](../../../docs/benchmarks/): `laplacian_scaling.json`,
`polylaplacian_order.json`, `derivative_order.json`, `jet_vs_nested_ad_smoke.json`.

## Further references

- API: `docs/api/torch.md`, `docs/api/jax.md`, `docs/api/core.md`
- Theory: `docs/theory.md`; activations: `docs/activations.md`
