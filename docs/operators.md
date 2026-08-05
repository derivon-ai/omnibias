# Operator catalog

`omnibias-fields` exposes a single, backend-agnostic **differential-operator
surface** that every field package (`omnibias-pinn`, `omnibias-qpinn`,
`omnibias-geometry`, `omnibias-score`, ...) builds on. Each operator is a
**closed-form** composition of the activation-derivative tower through a
`FieldState` -- never autodiff and never finite differences -- and the PyTorch
and JAX implementations are **bit-identical twins** (cross-backend parity tested
to `rtol = atol = 1e-12` in float64).

Every operator is reachable three equivalent ways:

```python
import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField

field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("x", "y")),
    components=ComponentSpec(("u",)), hidden=16, base="tanh",
)
state = field(torch.randn(8, 2, dtype=torch.float64))

state.ops.laplacian(state, "u")       # functional dispatch
state.u.lap                           # attribute DSL (ComponentView / VectorView)
from omnibias.fields.torch import ops # direct import (or .jax)
ops.laplacian(state, "u")
```

## Discovering operators

The catalog is queryable at runtime, grouped by **domain**:

```python
from omnibias.fields import list_operators, get_operator, DOMAINS

DOMAINS
# ('calculus', 'vector_calculus', 'conservation', 'fluids', 'mechanics',
#  'chemistry', 'electromagnetism', 'magnetohydrodynamics', 'kinetic',
#  'complex', 'integral')

[op.name for op in list_operators(domain="electromagnetism")]
# ['ampere_residual', 'electric_field_from_potentials', 'faraday_residual', ...]

get_operator("nernst_planck_flux")
# OperatorInfo(name='nernst_planck_flux', domain='chemistry',
#              formula='J = -D grad c - z mu F c grad phi',
#              closed_form=True, backends=('torch', 'jax'))
```

`closed_form` is `True` for the entire `omnibias-fields` surface; the flag exists
so that non-local / approximate operators contributed by downstream packages
(for example the grid-based fractional calculus in `omnibias-fractional`) are
catalogued honestly alongside the exact ones.

## Calculus

The primitive differential operators and accessors.

| operator | computes |
| --- | --- |
| `value` | `u` |
| `stack_components` | `(u_1, ..., u_C) -> (B, C)` |
| `derivative` | `d^k u / dx_a^k` |
| `mixed_partial` | `d^|orders| u / prod dx_a^{o_a}` |
| `gradient` | `grad u` |
| `divergence` (`div`) | `div u` |
| `laplacian` | `Delta u` |
| `biharmonic` | `Delta^2 u` |
| `polylaplacian` | `Delta^k u` |
| `hessian` / `spatial_hessian` | `Hess u` (all / spatial axes) |
| `jacobian` / `spatial_jacobian` | `J_ij = d u_i / dx_j` |
| `vector_derivative` | `d^k u_i / dx_a^k` |
| `vector_laplacian` | `(Delta u_1, ..., Delta u_C)` |
| `vector_biharmonic` | `(Delta^2 u_1, ..., Delta^2 u_C)` |
| `vector_hessian` | stacked component Hessians |
| `vector_polylaplacian` | `(Delta^k u_1, ..., Delta^k u_C)` |
| `gradient_of_derivative` | `grad(d u / dx_a)` |
| `directional_derivative` | `grad u . n` |

## Vector calculus

| operator | computes |
| --- | --- |
| `curl` (`rot`) / `vorticity` | `curl u` (2-D scalar, 3-D vector) |
| `curl_of_curl` | `curl(curl u)` |
| `gradient_of_divergence` | `grad(div u)` |
| `grad_squared_norm` | `|grad u|^2` |
| `deformation_gradient` | `F = grad u` |
| `strain_rate` | `eps = 0.5 (J + J^T)` |
| `rate_of_rotation_tensor` | `W = 0.5 (J - J^T)` |
| `p_laplacian` | `div(|grad u|^{p-2} grad u)` |

The identity `curl_of_curl == gradient_of_divergence - vector_laplacian` is
asserted by the test-suite rather than used to define `curl_of_curl`.

## Conservation, flux & waves

| operator | computes |
| --- | --- |
| `diffusive_flux` | `F = -D grad u` |
| `flux_divergence` | `div F` |
| `variable_coefficient_diffusion` | `div(D grad u)` |
| `conservation_residual` | `d_t rho + div F - s` |
| `advection_diffusion_residual` | `d_t c + (u.grad)c - div(D grad c) - s` |
| `gradient_of_composition` | `grad f(u) = f'(u) grad u` |
| `laplacian_of_composition` | `Delta f(u) = f'(u) Delta u + f''(u) |grad u|^2` |
| `dalembertian` (`wave_operator`) | `box u = Delta u - c^-2 d_tt u` |

## Fluids & continuum kinematics

| operator | computes |
| --- | --- |
| `advection` | `(u.grad) u` |
| `material_derivative` | `D/Dt = d_t + (u.grad)` |
| `skew_symmetric_advection` | `0.5[(u.grad)c + div(u c)]` |
| `velocity_from_streamfunction` | `(d_y psi, -d_x psi)` |
| `vorticity_from_streamfunction` | `omega = -Delta psi` |

See the [advection-diffusion cookbook](cookbook/pinn-advection-diffusion.md).

## Continuum mechanics

| operator | computes |
| --- | --- |
| `newtonian_stress` | `sigma = -p I + 2 mu eps` |
| `linear_elastic_stress` | `sigma = lam tr(eps) I + 2 mu eps` |
| `viscous_dissipation` | `Phi = 2 mu eps:eps` |
| `stokes_residual` | `mu Delta u - grad p + f` |
| `navier_cauchy_residual` | `(lam+mu) grad(div u) + mu Delta u + f` |
| `stress_divergence` / `tensor_divergence` | `(div sigma)_i = d_j sigma_ij` |
| `tensor_double_dot` | `A:B = A_ij B_ij` |

See the [linear-elasticity cookbook](cookbook/pinn-linear-elasticity.md).

## Chemistry & transport

| operator | computes |
| --- | --- |
| `fickian_flux` | `J = -D grad c` |
| `nernst_planck_flux` | `J = -D grad c - z mu F c grad phi` |
| `nernst_planck_residual` | `d_t c + div J_NP - s` |
| `darcy_flux` | `q = -(k/mu) grad p` |
| `reaction_diffusion_residual` | `d_t c - div(D grad c) - R(c) - s` |
| `poisson_residual` | `div(eps grad phi) + rho` |

Poisson-Nernst-Planck is the coupled system
`nernst_planck_residual + poisson_residual`. See the
[reaction-diffusion cookbook](cookbook/pinn-reaction-diffusion.md).

## Electromagnetism (3-D Maxwell, natural units)

| operator | computes |
| --- | --- |
| `faraday_residual` | `d_t B + curl E` |
| `ampere_residual` | `d_t E - curl B + J` |
| `gauss_residual` | `div E - rho` |
| `gauss_magnetic_residual` | `div B` |
| `poynting_vector` | `S = E x B` |
| `magnetic_field_from_potential` | `B = curl A` |
| `electric_field_from_potentials` | `E = -grad phi - d_t A` |
| `lorenz_gauge_residual` | `d_t phi + div A` |
| `vector_dalembertian` | `(box u_1, ..., box u_C)` |

See the [Maxwell cookbook](cookbook/pinn-maxwell.md).

## Magnetohydrodynamics (Alfven units, `mu_0 = rho_0 = 1`)

Single-fluid resistive/ideal MHD, built as closed-form compositions of the
`curl` / `divergence` / `gradient` / `advection` / `vector_laplacian`
primitives. The ideal limit is `resistivity = viscosity = 0`.

| operator | computes |
| --- | --- |
| `current_density` | `J = curl B` |
| `lorentz_force` | `J x B` |
| `magnetic_pressure` | `p_B = |B|^2 / 2` |
| `maxwell_stress_tensor` | `T_ij = E_i E_j + B_i B_j - 0.5 d_ij(|E|^2 + |B|^2)` |
| `magnetic_divergence` | `div B` (solenoidal constraint) |
| `induction_residual` | `d_t B - curl(u x B) - eta Delta B` |
| `ideal_mhd_momentum_residual` | `rho(d_t u + (u.grad)u) + grad p - J x B - nu Delta u - f` |

`induction_residual` expands `curl(u x B)` through the exact identity
`u(div B) - B(div u) + (B.grad)u - (u.grad)B`; a finite-amplitude Elsasser /
Alfven wave (`u = 1/2(F+C)`, `B = 1/2(F-C)`, `F = F(z - t)` divergence-free)
drives both residuals to zero. `B = 0` recovers Navier-Stokes; `u = 0` recovers
resistive magnetic diffusion.

## Kinetic theory (phase-space transport, natural units)

Collisionless transport plus the BGK relaxation closure on a phase-space field
`f(t, x, v)`. Vlasov transport, BGK relaxation and the Maxwellian are
closed-form; the full quadratic, non-local **Boltzmann collision integral is
numerical** (velocity-space quadrature) and is deliberately *not* provided as a
closed-form op.

| operator | computes |
| --- | --- |
| `vlasov_residual` | `d_t f + v.grad_x f + (F/m).grad_v f` |
| `bgk_collision` | `-(f - f_eq)/tau` |
| `bgk_vlasov_residual` | `L f + (f - f_eq)/tau` |
| `maxwellian` | `f_eq = n (m/2piT)^{d/2} exp(-m|v-u|^2/2T)` |
| `number_density` | `n = int f dv` |
| `momentum_density` | `int v f dv` |
| `kinetic_energy_density` | `int 0.5 |v|^2 f dv` |

The velocity moments contract the distribution against a supplied
velocity-space quadrature rule (evaluate the field at `quadrature_nodes` mapped
into the velocity axes first); the moments of a Maxwellian recover
`n`, `n u` and `n(0.5|u|^2 + d T / 2m)`.

## Complex / Wirtinger

| operator | computes |
| --- | --- |
| `dz` | `d/dz = 0.5(d_x - i d_y)` |
| `dzbar` | `d/dzbar = 0.5(d_x + i d_y)` |

## Integration & norms

| operator | computes |
| --- | --- |
| `integrate` | definite integral over the domain |
| `line_integral` | `int_C grad u . dr = u(end) - u(start)` (gradient theorem) |
| `inner_product` | `<u, v>` over the domain |
| `l2_norm` | `||u||_{L2}` |
| `sobolev_norm` | `||u||_{H^k}` |

## API

::: omnibias.fields._core.catalog
    options:
      show_root_heading: false
      heading_level: 3
