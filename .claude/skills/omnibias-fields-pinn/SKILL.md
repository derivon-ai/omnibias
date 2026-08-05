---
name: omnibias-fields-pinn
description: Use omnibias field operators (grad / div / curl / laplacian / hessian / jacobian, integration, Sobolev norms) and build physics-informed neural networks. Use when using omnibias to evaluate differential operators on a FieldState, assemble PDE residuals, or train a PINN / QPINN, or when the user mentions fields, PINNs, PDE residuals, or the field substrate.
---

# Using omnibias: fields and physics-informed neural networks

`omnibias-fields` is the foundational substrate: a `FieldState` value object plus
closed-form differential operators that share one bit-identical implementation
across PyTorch and JAX. `omnibias-pinn` / `omnibias-qpinn` build PINNs on top.

## Import map

| You want | Import from | Notes |
| --- | --- | --- |
| Field value object + views | `omnibias.fields` | `FieldState` (a single class object across the stack) |
| Differential operators (torch) | `omnibias.fields.torch.ops` | `grad`, `divergence`, `curl`, `laplacian`, `hessian`, `jacobian`, `integrate`, `inner_product`, `l2_norm`, `sobolev_norm` |
| Differential operators (JAX twin) | `omnibias.fields.jax.ops` | bit-identical to the torch surface (parity to ~1e-12 in float64) |
| Physics-informed NNs | `omnibias.pinn` | PDE residual layers on the field substrate |
| Quantum PINNs | `omnibias.qpinn` | TISE / TDSE / Gross-Pitaevskii |

`omnibias.pinn._core` and `omnibias.pinn.<backend>.ops` are transparent
re-export shims of the moved `omnibias-fields` substrate -- do not duplicate it.

## Canonical runnable examples (copy from these)

- PINN heat equation: `docs/examples/pinn_heat.py`
- QPINN harmonic oscillator (TISE): `docs/examples/qpinn_tise_qho.py`

## Gotchas that bite

- **Field ops are torch + jax only.** The Keras backend still gets bit-identical activation-level math through `OperatorBlock`, but the field operators are not exposed there.
- **Request the derivative order you need.** A `FieldState` jet carries partials up to its `order`; a Laplacian needs order 2.
- **Closed-form vs numerical is labeled honestly.** Vlasov transport, BGK, the Maxwellian, elasticity / hyperelasticity are closed-form; the full Boltzmann collision integral and history-dependent plasticity are numerical and deliberately not shipped as closed-form ops.

## More detail

- API: [fields](https://github.com/derivon-ai/omnibias/blob/main/docs/api/fields.md), [pinn](https://github.com/derivon-ai/omnibias/blob/main/docs/api/pinn.md), [qpinn](https://github.com/derivon-ai/omnibias/blob/main/docs/api/qpinn.md)
- Handbook: [vector calculus & PDE](https://github.com/derivon-ai/omnibias/blob/main/docs/handbook/02-vector-calculus-pde.md)
