---
name: omnibias-backends
description: Compute closed-form n-th derivatives of activations and exact Taylor jets with omnibias on PyTorch, JAX, or Keras 3. Use when using omnibias to take high-order derivatives, build OperatorBlock / OMBU layers, or propagate directional / multivariate jets, or when the user mentions sigma^(n), closed-form derivatives, jets, or bit-identical backends.
---

# Using omnibias: the closed-form derivative tower and jets

omnibias computes `sigma^(n)(z)` in closed form for any order `n` with a single
`sigma` evaluation, and it is **bit-identical across backends** because every
backend imports the coefficients from `omnibias.core.polynomials`.

## Import map

| You want | Import from | Key entry points |
| --- | --- | --- |
| Trainable operator layers (PyTorch) | `omnibias.torch` | `OperatorBlock`, `OperatorMultiBiasUnit` (`OMBU`), `cmbLinear`, `cmbConv1d`, `cmbConv2d` |
| Activation registry | `omnibias.torch` / `omnibias.jax` | `get_activation`, `list_activations`, `register_activation` |
| Closed-form field value / laplacian / hessian (JAX) | `omnibias.jax` | `neural_field_value`, `neural_field_laplacian`, `neural_field_hessian`, `neural_field_value_grad_hessian` |
| Directional Taylor jets (deep composition) | `omnibias.torch` / `omnibias.jax` | `mlp_jet`, `layer_jet`, `compose_jet`, `tower_to_jet`, `jet_to_tower` |
| Multivariate jets: every mixed partial to order N | `omnibias.torch` / `omnibias.jax` | `mlp_jet_mv`, `jet_partials`, `jet_gradient`, `jet_hessian` |
| Raw polynomial coefficients (pure Python) | `omnibias.core` | `sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`, `hermite_coeffs` |

## OperatorBlock roles (six, including a closed-form integral)

`OperatorBlock` dispatches on `op="identity"|"grad"|"laplacian"|"derivative"|"band"|"integral"`:

- `identity` (K=1): `sigma(z + b)` (literal; Lemma identity).
- `grad` (K=2) / `laplacian` (K=3): closed-form `sigma'` / `sigma''` at the bias mean.
- `derivative` (K=n+1): closed-form `sigma^(n)` at arbitrary order `n` (`grad` / `laplacian` are the `n = 1, 2` aliases).
- `band` (K=2): the literal window `sigma(z + b_hi) - sigma(z + b_lo)`.
- `integral` (K=2): the **closed-form antiderivative** window `S(z + b_hi) - S(z + b_lo)` with `S' = sigma` (the `ActivationSpec.integral` kernel; e.g. `sigmoid`'s antiderivative is `softplus`). omnibias has a closed-form integral operator, not only closed-form derivatives.

`grad` / `laplacian` / `derivative` need a base with a `fastpath` kernel; `integral` needs an antiderivative kernel; `OperatorBlock` raises `TypeError` otherwise.

**"Integral" has three distinct senses -- do not conflate:** (1) the activation antiderivative window above (`OperatorBlock(op="integral")`, closed form); (2) domain quadrature `sum_q w_q u(x_q)` (`omnibias.fields` `integrate` / `l2_norm` / `sobolev_norm`, numerical); (3) the measure integral `integral f dmu` (`omnibias.measure`, numerical; certified variant in `omnibias.verify`). Ground any capability claim in the canonical operator-surface matrix (`docs/operator-surface.md`), never in memory.

## Canonical runnable examples (copy from these)

- PyTorch: `docs/examples/quickstart_torch.py`
- JAX: `docs/examples/quickstart_jax.py`
- Keras 3: `docs/examples/quickstart_keras.py` (set `KERAS_BACKEND` first)

## Gotchas that bite

- **New tensors follow the framework default dtype** (`torch.get_default_dtype()` / `keras.config.floatx()`), never a hardcoded `float32`. Use `float64` when you need bit-identical cross-backend parity.
- **`n < 0` raises `ValueError`; genuinely unimplemented orders raise `NotImplementedError`.** Catch them; do not paper over the contract.
- **A jet only carries partials up to the order you request.** Ask for `order=2` for a Hessian, `order=3` for third derivatives; higher orders cost nothing extra per point but must be requested.
- **Keras uses `keras.ops.*` only** and is backend-agnostic; select the backend with `KERAS_BACKEND=jax|tensorflow|torch`.
- **Tab heads as layers.** `as_head(z, kind)` moves the head to `z.device` / `z.dtype`; logits are `(..., k)`. Keras tab layers live in `omnibias.tab.keras` (not `omnibias-keras`) and use `keras.ops`; `learnable_beta` on `ArrangementBoosted` is member-`beta` (ensemble `learning_rate` / `base` stay frozen). Equinox wrappers are an optional `[equinox]` extra (`omnibias.tab.jax.equinox_head`); tab CI **fails** if the extra is missing when `CI` is set (local `importorskip`).

## More detail

- API: [torch](https://github.com/derivon-ai/omnibias/blob/main/docs/api/torch.md), [jax](https://github.com/derivon-ai/omnibias/blob/main/docs/api/jax.md), [core](https://github.com/derivon-ai/omnibias/blob/main/docs/api/core.md), [tab](https://github.com/derivon-ai/omnibias/blob/main/docs/api/tab.md)
- Cookbook: [tab as a layer](https://github.com/derivon-ai/omnibias/blob/main/docs/cookbook/tab-as-layer.md)
- Theory: [closed-form derivatives](https://github.com/derivon-ai/omnibias/blob/main/docs/theory.md); [activation dictionary](https://github.com/derivon-ai/omnibias/blob/main/docs/activations.md)
