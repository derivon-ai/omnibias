# omnibias-fields

The backend-agnostic **field substrate**: the `FieldState` value object, the
attribute-DSL views, the lazy `sigma^(n)(z)` cache, the op-extension registry,
and the cross-backend (PyTorch + JAX) closed-form differential-operator surface
(plus integration, inner products, Sobolev norms, tensor divergence, and
Wirtinger calculus).

This package was extracted from `omnibias-pinn` so that every field-based
extension (`omnibias-pinn`, `omnibias-geometry`, `omnibias-score`) builds on one
shared, bit-identical substrate. `omnibias-pinn` re-exports the moved symbols
through transparent back-compat shims, so existing `omnibias.pinn._core` and
`omnibias.pinn.<backend>.ops` imports keep working unchanged.

## Core schemas

::: omnibias.fields._core
    options:
      show_root_heading: false
      heading_level: 3

## Quadrature

::: omnibias.fields._core.quadrature
    options:
      show_root_heading: false
      heading_level: 3

## Finite-strain solid mechanics

Alongside the small-strain fluid / linear-elastic ops, the surface carries
**finite-deformation** solid mechanics: batched tensor algebra
(`tensor_determinant`, `tensor_inverse`, `tensor_cofactor`, `tensor_matmul`,
`tensor_trace`, `tensor_transpose`), kinematics (`deformation_gradient_finite`
`F = I + ∇u`, `right_cauchy_green`, `green_lagrange_strain`, `jacobian_det`),
the hyperelastic energies (`st_venant_kirchhoff_energy`, `neo_hookean_energy`,
`mooney_rivlin_energy`), the stresses (`pk1_stress`/`pk2_stress`/`cauchy_stress`
as the exact autodiff gradient `∂W/∂F`, plus the validated closed forms
`st_venant_kirchhoff_pk2` / `neo_hookean_pk2` and the anisotropic
`hooke_stress_general`), and the balance laws (`finite_strain_residual`,
`elastodynamic_residual`). The stress divergence combines an autodiff-exact
constitutive tangent with the closed-form second spatial derivatives of the
displacement; elasticity/hyperelasticity/elastodynamics are exact, while
history-dependent plasticity/viscoelasticity is out of the closed-form scope.

## Magnetohydrodynamics & kinetic theory

Single-fluid **MHD** in Alfven units (`mu_0 = rho_0 = 1`): `current_density`
(`J = curl B`), `lorentz_force` (`J x B`), `magnetic_pressure`,
`maxwell_stress_tensor`, `magnetic_divergence`, `induction_residual`
(`d_t B - curl(u x B) - eta lap B`, with `curl(u x B)` expanded through the exact
vector identity), and `ideal_mhd_momentum_residual` (Navier-Stokes plus the
Lorentz force). A finite-amplitude Elsasser/Alfven wave drives both residuals to
zero; `B = 0` recovers Navier-Stokes and `u = 0` recovers resistive diffusion.

**Kinetic theory** on a phase-space `f(t, x, v)`: `vlasov_residual`
(`d_t f + v.grad_x f + (F/m).grad_v f`), `bgk_collision` / `bgk_vlasov_residual`,
the closed-form `maxwellian`, and the velocity moments `number_density`,
`momentum_density`, `kinetic_energy_density`. Vlasov transport, BGK and the
Maxwellian are closed-form; the full non-local Boltzmann collision integral is
numerical (quadrature) and is deliberately not shipped as a closed-form op.

## Line integral & the gradient theorem

`line_integral(state, name, curve, rule)` computes `int_C grad u . dr` for a
scalar potential `u` along a curve `r`, which by the multivariate Fundamental
Theorem of Calculus (the gradient theorem) equals `u(curve(t1)) - u(curve(t0))`
for **any** path. The `curve` is a bare callable mapping a `(1,)` parameter to an
ambient point — `omnibias-fields` never imports `omnibias-geometry`, so a curve is
not a `ChartSpec`; `state` must be pre-evaluated at `curve(quadrature_nodes(rule))`
(the same convention as the geometry surface integrals).

!!! note "Honesty label"
    The **integrand** is exact — the field gradient `grad u` is the closed-form
    sigma-tower op and the curve tangent `r'(t)` is exact forward-mode autodiff —
    but the **integral** is a numerical **Gauss-Legendre quadrature** (exact for
    polynomials up to the rule degree, convergent otherwise), matching the
    `integrate` op and the geometry surface integrals.

## Ops (torch)

::: omnibias.fields.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend has the same module layout under `omnibias.fields.jax`. All
cross-backend tests assert *bit-identical* results between the two backends
(typical tolerances: rtol/atol=1e-12 in float64).
