---
name: omnibias-pinn
description: Build physics-informed networks with closed-form residual operators, causal marching, SDF cages, DeepONet / FNO operators, and certified weak forms. Use when assembling a PDE residual, inventing a hard BC or operator learner, or when the user mentions PINNs, causal windows, or the four-gap matrix.
---

# Physics-informed networks on the closed-form tower

`omnibias-pinn` builds PINNs on `omnibias-fields`. Residuals, hard cages, causal
windows, and operator learners consume exact `sigma^(n)` instead of nested AD.

## Why nested AD fails

Collocation PINNs differentiate a network at thousands of points. Nested AD
rebuilds a graph per point per order; high-order and polyharmonic residuals
are the first thing that OOMs. Soft boundary penalties leak. Operator learners
without an exact trunk jet cannot zero-shot a new coefficient field. Spectral
bias stays an optimizer accident when the residual operator itself is approximate.

## What only this tower unlocks

Closed-form towers plus hard cages are exact for the quantities they encode
(Dirichlet on `φ = 0`, conservation identities). Causal marching, SDF geometry,
conditioned DeepONet / FNO, and multilevel FBPINN are the constructive routes
that dissolve named PINN gaps. The four-gap suite
(`docs/benchmarks/pinn_four_gap_matrix.md`) is the absolute gate nested AD
stacks do not clear at these orders.

## Use

| You want | Import from | Notes |
| --- | --- | --- |
| Physics-informed NNs | `omnibias.pinn` | PDE residual layers on the field substrate |
| Causal time marching | `omnibias.pinn.train` | windowed causal residual weighting + advance gates |
| Curved hard BCs / SDF | `omnibias.pinn.domain` | negative-inside SDF + `DistanceConstrainedField` |
| Operator learning | `omnibias.pinn.operator` | DeepONet / FNO with multi-head conditioning |
| Multilevel FBPINN / NTK | `omnibias.pinn.torch.fields` / `.losses` | spectral-bias arms + diagnostics (JAX twins) |
| Transmission PINN | `omnibias.pinn.interface` | `Interface` / `MultiInterfaceField`; `alpha -> inf` is sharpening |
| Tanh-method solitons | `omnibias.pinn.travelling` | `SolitonField` |
| Layered transfer | `omnibias.pinn.layered` | `TransferStack`; 1-D ABCD |
| BEM-Net | `omnibias.pinn.bem` | `BEMNet`, `half_plane_dtn` |
| Linearizing transforms | `omnibias.pinn.transform` | `ColeHopfField`, `darboux_dress` |
| Characteristic transport | `omnibias.pinn.characteristic` | transport along learned `v` |
| Certified weak form | `omnibias.pinn.certified.weak_form` | width split + exact-jet Lohner |
| Inverse / coefficient recovery | `omnibias.pinn.inverse` | locally-seeded `sd ~ alpha^(n-5/2)` |
| IPM radii / toy CAP | `omnibias.pinn.certified.ipm` | `ipm_banded_toy_radii` |

Examples: `docs/examples/pinn_heat.py`, `pinn_causal_marching.py`,
`pinn_sdf_geometry.py`. Quantum residuals live in `omnibias-qpinn`.

Request the derivative order the residual needs. Neumann/Robin on CSG
junctions need smooth normals; non-smooth junctions fail explicitly.

## Extend

- Source: `packages/omnibias-pinn`. Substrate stays in `omnibias-fields`.
- Tests: `python -m pytest packages/omnibias-pinn/tests -q`.
- Research doctrine: `omnibias-pinn-research`. Compose with `omnibias-fields`,
  `omnibias-geometry`, `omnibias-qpinn`, `omnibias-symbolic`, `omnibias-verify`.
- Alpha submodules (`train`, `domain`, `operator`) are the place for aggressive
  prototypes. New top-level packages still earn independence via `omnibias-new-package`.

## Next invention

A named PDE family whose hard cage, causal window, and closed-form residual
clear the four-gap absolute gate with a `gates` JSON, then a one-shot operator
that zero-shots a new coefficient on the same family.

## Bakeoffs

`docs/benchmarks/laplacian_scaling.json`, `polylaplacian_order.json`,
plus the four-gap suite (`causal_marching`, `geometry_sdf`,
`operator_zero_shot`, `spectral_bias_fbpinn`).

## Further references

- API: `docs/api/pinn.md`, `docs/api/ipm_radii.md`, `docs/api/interface.md`, `docs/api/travelling.md`,
  `docs/api/layered.md`, `docs/api/bem.md`, `docs/api/transforms_pde.md`
- Handbook: `docs/handbook/02-vector-calculus-pde.md`
