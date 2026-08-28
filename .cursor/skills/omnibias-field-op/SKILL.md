---
name: omnibias-field-op
description: Add or modify a differential field operator in omnibias-fields (grad / div / curl / laplacian / hessian / jacobian, integration, norms, tensor divergence, Wirtinger) with bit-identical PyTorch and JAX twins. Use when extending the field-operator surface, the ops registry, or the FieldState substrate.
---

# Adding a field operator

The field substrate is foundational: `pinn`, `geometry`, and `score` build on
it. It never imports a downstream package; its two backends agree bit-for-bit.

## Why nested AD fails

A new Laplacian-like op via nested AD is a one-off graph that forks torch vs
jax and bypasses `SigmaCache`. Without `_omnibias_dispatch`, the substrate
would have to import concrete field classes and the layering collapses.
High-order mixed partials are exactly the nested-AD bottleneck
(`laplacian_scaling.json`).

## What only this tower unlocks

Ops select the closed-form `sigma^(n)` path via the `_omnibias_dispatch`
class marker (`omnibias.fields._core.DISPATCH_ATTR`). One catalog, two
bit-identical backends, one `FieldState`. That is the composition the rest
of the workspace depends on.

## Use

- Substrate (`FieldState`, views, `SigmaCache`, `ops_registry`, catalog):
  `omnibias.fields._core` (`catalog.py` lists the operator surface).
- Backend ops: `omnibias.fields.torch.ops` and `omnibias.fields.jax.ops` —
  write both.
- `omnibias.pinn._core` and `omnibias.pinn.<backend>.ops` are re-export shims;
  keep them working.

Label closed-form vs numerical (Vlasov / BGK / Maxwellian and elasticity are
closed-form; a full Boltzmann collision integral is numerical).

## Extend

1. Add the op to both backends and register it in the catalog / ops registry.
2. Regenerate sorted `__all__`.
3. Regression test **and** torch-vs-jax parity (`rtol/atol=1e-12` in float64).
4. Author new modules strict-clean.

```bash
python -m pytest packages/omnibias-fields/tests -q
```

New field-level ops are torch + jax. Keras keeps activation-level math via
OperatorBlock. Compose with `omnibias-fields`, `omnibias-pinn`,
`omnibias-geometry`.

## Next invention

A covariant / tensor operator that geometry and PINN both consume from the
catalog, with a `laplacian_scaling.json` row at an order nested AD does not
finish.

## Bakeoffs

`docs/benchmarks/laplacian_scaling.json`, `polylaplacian_order.json`.

## Further references

- `docs/api/fields.md`
