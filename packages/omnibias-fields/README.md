# omnibias-fields

**Status: Beta (v0.1.0).**

The backend-agnostic **field substrate** for omnibias: the `FieldState` value
object, the attribute-DSL views (`state.u.grad`, `state.velocity.div`, ...), the
lazy `sigma^(n)(z)` cache, the op-extension registry, and the cross-backend
(torch + jax) closed-form differential-operator surface.

## Why this is the substrate

- **One forward pass per derivative order** — the `SigmaCache` evaluates
  `sigma^(n)(z)` exactly once per `(layer, order)` pair and feeds every
  downstream op (`grad`, `div`, `curl`, `laplacian`, `hessian`, `jacobian`,
  Sobolev norms, tensor divergence, Wirtinger).
- **Cross-backend bit-identity** — the torch and JAX op surfaces are
  arithmetic twins over the sigma tower the caller supplies, and that tower
  comes from the one shared `omnibias.core.polynomials` recurrence via
  `omnibias.torch` / `omnibias.jax`. A Laplacian on torch is therefore
  ULP-equal to the same Laplacian on JAX in float64.
- **One extension surface** — `omnibias-pinn`, `omnibias-geometry`,
  `omnibias-score` (and any external package that registers ops through
  `ops_registry`) all share one `FieldState`. The same `state.u.grad` syntax
  works everywhere.

This package was extracted from `omnibias-pinn` so that every field-based
extension can build on one shared, bit-identical substrate. `omnibias-pinn`
re-exports the moved symbols through back-compat shims, so existing
`omnibias.pinn._core` and `omnibias.pinn.<backend>.ops` imports keep working
unchanged.

## Install

```bash
pip install "omnibias-fields[torch]"   # or [jax], or [all]
```

## What's here

| Layer | Module | Contents |
|---|---|---|
| Schemas (pure Python) | `omnibias.fields._core` | `FieldState`, `ComponentSpec`, `CoordinateSpec`, `ComponentView`, `VectorView`, `SigmaCache`, `FieldBase`, `ops_registry`, `quadrature` |
| Torch ops | `omnibias.fields.torch.ops` | `value`, `derivative`, `gradient`, `divergence`, `laplacian`, `hessian`, `jacobian`, `curl`, `integrate`, `inner_product`, `l2_norm`, `sobolev_norm`, `tensor_divergence`, `dz`, `dzbar`, ... |
| JAX ops | `omnibias.fields.jax.ops` | the bit-identical twin of the torch surface |

## Building on top of this

A *field* is any object implementing the `FieldBase` protocol that, when called
on a `(B, D)` coordinate tensor, returns a `FieldState`. To make the closed-form
ops dispatch correctly, set the class attribute named by
`omnibias.fields._core.DISPATCH_ATTR` (default `"_omnibias_dispatch"`) to one of
the dispatch tags: `"one_layer"` selects the closed-form sigma-tower reduction;
any other tag selects the state-method path (the field implements
`value_component`, `derivative`, `mixed_partial` taking the `FieldState`).

To add a new op without modifying this package, register it:

```python
from omnibias.fields import ops_registry

@ops_registry.register("symmetric_laplacian")
def symmetric_laplacian(state, name):
    ...
# now available as state.u.symmetric_laplacian
```

See [`FIELDS_DERIVATIONS.md`](FIELDS_DERIVATIONS.md) for the math behind each op
and the numerical-stability notes.

## Invariants

- **Pure-Python core.** `omnibias.fields._core` imports no torch / jax / numpy.
- **Cross-backend bit-identity.** The torch and jax ops are arithmetic twins
  over the same pure-Python schemas; the sigma tower they consume comes from the
  shared `omnibias-core` polynomial coefficients. They agree to
  `rtol=1e-12, atol=1e-12` on the parity tests.
- **One sigma evaluation per `(order, axis)`.** The `SigmaCache` is filled
  lazily and reused across all ops in a residual.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
