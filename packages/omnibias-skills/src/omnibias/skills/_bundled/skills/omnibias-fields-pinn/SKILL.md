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
| Causal time marching | `omnibias.pinn.train` | windowed causal residual weighting + advance gates |
| Curved hard BCs / SDF | `omnibias.pinn.domain` | negative-inside SDF + `DistanceConstrainedField` |
| Operator learning | `omnibias.pinn.operator` | DeepONet / FNO with multi-head conditioning |
| Multilevel FBPINN / NTK | `omnibias.pinn.torch.fields` / `.losses` | spectral-bias arms + diagnostics (JAX twins) |
| Quantum PINNs | `omnibias.qpinn` | TISE / TDSE / Gross-Pitaevskii |
| Weak-form VPINN (gated 02-04) | `omnibias.fields.weak` | `TestFunctionSpace`, `exact_moment`, `weak_residual`; exact integrals only for polynomial coeffs on boxes; boundary bound on by default |
| Transmission PINN (gated 02-05) | `omnibias.pinn.interface` | `Interface` / `TransmissionInterface`, `MultiInterfaceField`; `alpha -> inf` is sharpening, neither collapse; **not** XPINN `omnibias.pinn._core.interface` |
| Equality locus (gated 01-09 / 02-12) | `omnibias.fields.locus` | `EqualityLocusLayer` / `LocusOutput`; constraint manifold, not a general PDE solver; always `branch` / `condition` / `converged` |
| Tanh-method solitons (gated 02-09) | `omnibias.pinn.travelling` | `SolitonField`; tanh algebra, not a collapse; multi-kink is not the n-soliton formula |
| Layered transfer (gated 02-11) | `omnibias.pinn.layered` | `TransferStack`; 1-D ABCD; `continuum_claim=False`; not `geometry.gauge.transfer` |
| BEM-Net (gated 02-06) | `omnibias.pinn.bem` | `BEMNet`, `half_plane_dtn`; PDE exact off-surface; BC approximated; linear constant-coeff homogeneous |
| Linearizing transforms (gated 02-13) | `omnibias.pinn.transform` | `ColeHopfField`, `darboux_dress`; named maps only; 03-11 search not claimed |

`omnibias.pinn._core` and `omnibias.pinn.<backend>.ops` are transparent
re-export shims of the moved `omnibias-fields` substrate -- do not duplicate it.

## Canonical runnable examples (copy from these)

- PINN heat equation: `docs/examples/pinn_heat.py`
- Causal marching: `docs/examples/pinn_causal_marching.py`
- SDF geometry cage: `docs/examples/pinn_sdf_geometry.py`
- QPINN harmonic oscillator (TISE): `docs/examples/qpinn_tise_qho.py`

## Guarantee levels (read before claiming)

Closed-form towers and hard cages are exact **by construction** for the
quantities they encode (e.g. Dirichlet on `φ = 0`). Training success, spectral
bias mitigation via one-shot collocation, and zero-shot operator generalization
are **empirical** on the named families in `docs/benchmarks.md` -- claim them
plainly once the absolute gate in `docs/benchmarks/pinn_four_gap_matrix.md`
passes. Prefer capability + acceptance-domain language; absolute skill floors
and validity guards are what make the win undeniable.

## Gotchas that bite

- **Field ops are torch + jax only.** The Keras backend still gets bit-identical activation-level math through `OperatorBlock`, but the field operators are not exposed there.
- **Request the derivative order you need.** A `FieldState` jet carries partials up to its `order`; a Laplacian needs order 2.
- **Closed-form vs numerical is labeled honestly.** Vlasov transport, BGK, the Maxwellian, elasticity / hyperelasticity are closed-form; the full Boltzmann collision integral and history-dependent plasticity are numerical and deliberately not shipped as closed-form ops.
- **Neumann/Robin on CSG junctions** need smooth normals; non-smooth junctions fail explicitly rather than silently inventing a normal.

## More detail

- API: [fields](https://github.com/derivon-ai/omnibias/blob/main/docs/api/fields.md), [pinn](https://github.com/derivon-ai/omnibias/blob/main/docs/api/pinn.md), [qpinn](https://github.com/derivon-ai/omnibias/blob/main/docs/api/qpinn.md), [weak form](https://github.com/derivon-ai/omnibias/blob/main/docs/api/weak.md), [transmission interface](https://github.com/derivon-ai/omnibias/blob/main/docs/api/interface.md), [locus](https://github.com/derivon-ai/omnibias/blob/main/docs/api/locus.md), [travelling](https://github.com/derivon-ai/omnibias/blob/main/docs/api/travelling.md), [layered](https://github.com/derivon-ai/omnibias/blob/main/docs/api/layered.md), [BEM](https://github.com/derivon-ai/omnibias/blob/main/docs/api/bem.md), [linearizing transforms](https://github.com/derivon-ai/omnibias/blob/main/docs/api/transforms_pde.md)
- Handbook: [vector calculus & PDE](https://github.com/derivon-ai/omnibias/blob/main/docs/handbook/02-vector-calculus-pde.md)
