---
name: omnibias-dev-field-op
description: Add or modify a differential field operator in omnibias-fields (grad / div / curl / laplacian / hessian / jacobian, integration, norms, tensor divergence) with bit-identical PyTorch + JAX twins. Use when extending the field-operator surface, the ops registry, or the FieldState substrate. For contributors modifying omnibias itself, not for consumers using it.
---

# Adding a field operator the omnibias way

The field substrate is foundational: everything downstream (`pinn`, `geometry`,
`score`) builds on it, so it must never import a downstream package and its two
backends must agree bit-for-bit.

## Where things live

- Substrate (`FieldState`, attribute-DSL views, `SigmaCache`, `ops_registry`, the op catalog): `omnibias.fields._core` (`.../catalog.py` lists the operator surface).
- Backend ops: `omnibias.fields.torch.ops` and `omnibias.fields.jax.ops` -- **bit-identical twins; write both.**
- `omnibias.pinn._core` and `omnibias.pinn.<backend>.ops` are transparent re-export shims of the moved substrate; keep them working, don't duplicate logic.

## The dispatch rule (do not break the layering)

- Backend ops select the closed-form sigma-tower path via the `_omnibias_dispatch` class marker (name = `omnibias.fields._core.DISPATCH_ATTR`), **not** by importing concrete field classes. This is exactly why the foundational package never imports a downstream one. Follow the same pattern for a new op.
- New field-level ops are **torch + jax only** (matching `omnibias-pinn`). Keras users still get bit-identical activation-level math through `OperatorBlock`; do not try to add field ops to the keras backend.

## Honesty labels

State what is closed-form vs numerical. E.g. Vlasov transport / BGK / the
Maxwellian and elasticity / hyperelasticity are closed-form; the full Boltzmann
collision integral and history-dependent plasticity are numerical and are
deliberately not shipped as closed-form ops. Label new ops the same way.

## Checklist

- Add the op to both backends and register it in the catalog / ops registry.
- Regenerate the sorted `__all__` where you added a public symbol.
- Add a regression test **and** a cross-backend parity test (torch vs jax, typically `rtol/atol=1e-12` in float64).
- `omnibias-fields` is extension-tier: not on the shared `mypy --strict` gate, but author new modules strict-clean anyway.

```bash
python -m pytest packages/omnibias-fields/tests -q
python -m pytest tests -q          # cross-backend parity
uv run ruff check packages tests
```
