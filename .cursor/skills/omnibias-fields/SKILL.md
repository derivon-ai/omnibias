---
name: omnibias-fields
description: Evaluate closed-form field operators (grad, div, curl, laplacian, hessian, jacobian, integrate, Sobolev norms, Wirtinger) on a FieldState with bit-identical PyTorch and JAX twins. Use when assembling PDE residuals, inventing a new differential op, or when the user mentions FieldState, SigmaCache, or the field substrate.
---

# Field substrate and closed-form operators

`omnibias-fields` is the foundational substrate: a `FieldState` value object plus
closed-form differential operators that share one bit-identical implementation
across PyTorch and JAX. Physics, geometry, score, and variational packages
compose this surface; they do not reimplement it.

## Why nested AD fails

A neural-field Laplacian via nested reverse-mode AD rebuilds the graph at every
collocation point and every extra order. Mixed partials for Hessians and
Jacobians multiply that cost. Backends diverge because there is no shared
`SigmaCache`, no ops registry, and no single `FieldState` class object.
High-order PINN and FermiNet workloads die on that graph long before the PDE
residual is interesting.

## What only this tower unlocks

Operators select the closed-form `sigma^(n)` path via the `_omnibias_dispatch`
marker (`omnibias.fields._core.DISPATCH_ATTR`), so the substrate never imports a
downstream package. One forward pass yields grad / div / curl / laplacian /
hessian / jacobian at the requested jet order. `docs/benchmarks/laplacian_scaling.json`
and `polylaplacian_order.json` are the named bakeoffs nested AD does not sustain.

## Use

| You want | Import from | Notes |
| --- | --- | --- |
| Field value object + views | `omnibias.fields` | `FieldState` (one class object across the stack) |
| Differential operators (torch) | `omnibias.fields.torch.ops` | `grad`, `divergence`, `curl`, `laplacian`, `hessian`, `jacobian`, `integrate`, `inner_product`, `l2_norm`, `sobolev_norm` |
| Differential operators (JAX) | `omnibias.fields.jax.ops` | bit-identical (parity ~1e-12 in float64) |
| Weak-form VPINN | `omnibias.fields.weak` | `TestFunctionSpace`, `exact_moment`, `weak_residual` |
| Equality locus | `omnibias.fields.locus` | `EqualityLocusLayer` / `LocusOutput` |
| Scale / coarse-graining | `omnibias.fields.scale` | exact `alpha^n` rescaling |

`omnibias.pinn._core` and `omnibias.pinn.<backend>.ops` are transparent
re-export shims of this substrate. Domain quadrature here is integral sense (2);
OperatorBlock `integral` is sense (1). Request the jet order you need: a
Laplacian needs order 2.

Closed-form vs numerical is labelled on the op: Vlasov / BGK / Maxwellian and
elasticity are closed-form; a full Boltzmann collision integral is numerical.

## Extend

- Substrate: `omnibias.fields._core` (`catalog.py` lists the operator surface).
- Add an op through `omnibias-field-op`: both backends, registry, parity test.
- Tests: `python -m pytest packages/omnibias-fields/tests -q`.
- Compose with `omnibias-pinn`, `omnibias-geometry`, `omnibias-score`,
  `omnibias-variational`, `omnibias-measure` by those names.
- New field-level ops are torch + jax. Keras still gets activation-level math
  through OperatorBlock.

## Next invention

A new tensor operator (covariant divergence on a pullback metric, or a
Wirtinger pack) registered once in `_core`, implemented as torch/jax twins,
consumed by geometry and PINN without a third copy, and winning
`laplacian_scaling.json` at an order nested AD does not finish.

## Bakeoffs

`docs/benchmarks/laplacian_scaling.json`, `polylaplacian_order.json`.

## Further references

- API: `docs/api/fields.md`, `docs/api/weak.md`, `docs/api/locus.md`
- Handbook: `docs/handbook/02-vector-calculus-pde.md`
