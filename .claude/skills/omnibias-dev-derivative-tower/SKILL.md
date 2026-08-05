---
name: omnibias-dev-derivative-tower
description: Develop or modify omnibias's closed-form derivative tower -- the shared polynomial coefficients, Bell / Faa di Bruno combinatorics, and the bit-identical torch / jax jet kernels. Use when adding an activation, changing sigma^(n) coefficients, or editing jet / jet_mv, and every backend must stay bit-identical by construction. For contributors modifying omnibias itself, not for consumers using it.
---

# Maintaining the derivative-tower contract

The whole library rests on one invariant: `sigma^(n)(z)` is computed from
**one shared set of pure-Python coefficients**, so every backend is bit-identical
by construction. Breaking that is the most expensive mistake you can make here.

## The one source of truth

- All polynomial coefficients live in `packages/omnibias-core/src/omnibias/core/polynomials.py` (`sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`, `hermite_coeffs`). **Every backend imports these. Never fork or reimplement them per backend.**
- Bell polynomials / Faa di Bruno combinatorics: `omnibias.core.bell` (pure Python).
- Multi-index (multivariate) combinatorics: `omnibias.core.multi_index`.
- `omnibias.core` must **never** import `torch`, `jax`, `tensorflow`, or `keras`.

## Bit-identical twins (edit both, together)

- Directional jets: `omnibias.torch.jet` and `omnibias.jax.jet` (`compose_jet`, `affine_jet`, `layer_jet`, `mlp_jet`, `tower_to_jet`, `jet_to_tower`).
- Multivariate jets: `omnibias.torch.jet_mv` and `omnibias.jax.jet_mv` (`mlp_jet_mv`, `layer_jet_mv`, `compose_jet_mv`, `identity_jet`, `jet_partials`, `jet_gradient`, `jet_hessian`).
- An activation is described by an `ActivationSpec` (`omnibias.core.spec`); backends specialise the tensor type but share the metadata.

## Hard rules for a change here

- A fastpath kernel computing `sigma^(n)`: `n < 0` **must** raise `ValueError`; a genuinely unimplemented order raises `NotImplementedError`.
- New tensors default to the framework default dtype (`torch.get_default_dtype()` / `keras.config.floatx()`), never a hardcoded `float32`.
- Regenerate the sorted `__all__` block at the bottom of any `__init__.py` you touch.
- Add a **regression test for every behavioural change** (this is non-negotiable in this repo).

## Verify before you claim done

```bash
python -m pytest packages/omnibias-core/tests -q
python -m pytest tests -q            # cross-backend torch <-> jax <-> keras parity
uv run ruff check packages tests
uv run mypy --strict packages/omnibias-core/src packages/omnibias-torch/src \
  packages/omnibias-jax/src packages/omnibias-ferminet/src
```

Cross-backend parity tests live in `tests/` (repo root) and `packages/*/tests/`.
If torch and jax disagree, the coefficients were forked somewhere -- fix the
source, do not add a tolerance.
