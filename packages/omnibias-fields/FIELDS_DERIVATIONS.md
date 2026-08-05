# omnibias-fields derivations

Math, assumptions, and numerical notes for the field-level operators shipped by
`omnibias-fields`. Every operator consumes a `FieldState` and (for the closed
sigma-tower path) one cached `sigma^(n)(z)` evaluation per order. The torch and
jax backends are bit-identical by construction because they share the
`omnibias-core` polynomial coefficients and the pure-Python schemas.

## Conventions

A field maps a collocation batch `x` of shape `(B, D)` to one or more scalar
components. For the one-layer field, component `f` is

\[
    f(x) = \sum_{h} c_{fh}\,\sigma(z_h) + b_f, \qquad z = W x + \beta,
\]

so the `n`-th partial along axis `i` is a single closed-form contraction of the
cached `sigma^{(n)}(z)` against the appropriate product of weight rows
(`W[:, i]`). This is the same derivative-tower contract documented in the repo
root `AGENTS.md`; the field ops never call autodiff for these.

## Differential operators (extracted from omnibias-pinn, unchanged)

These were moved verbatim from `omnibias-pinn`; their derivations live in
[`docs/pinn-derivations.md`](../../docs/pinn-derivations.md):

- `value`, `derivative`, `mixed_partial`, `gradient`, `divergence`, `laplacian`
- `hessian`, `spatial_hessian`, `biharmonic`, `polylaplacian`, `jacobian`
- `curl` / `vorticity`, `strain_rate`, `deformation_gradient`, `spatial_jacobian`
- `advection`, `material_derivative`, `p_laplacian`, `directional_derivative`

## New operators (Phase 2+)

### Integration over a domain

For a quadrature rule with nodes \(x_q\) and weights \(w_q\),

\[
    \int_\Omega f\,dx \;\approx\; \sum_q w_q\, f(x_q).
\]

When `D == 1` and the field's activation exposes an exact antiderivative
\(S\) with \(S' = \sigma\), the band integral is computed in closed form as
\(S(z_\text{hi}) - S(z_\text{lo})\) (one sigma-tower evaluation), and the result
is exact rather than a quadrature approximation. Otherwise the quadrature sum is
used and the docstring labels it as such.

Nodes and weights are generated once in pure Python / numpy
(`omnibias.fields._core.quadrature`) and converted to the backend tensor type at
call time, so torch and jax integrate bit-identically.

### Inner products and norms

\[
    \langle a, b\rangle_w = \int_\Omega w\,a\,b\,dx, \qquad
    \lVert u\rVert_{L^2} = \sqrt{\langle u, u\rangle},
\]
\[
    \lVert u\rVert_{H^k}^2 = \sum_{|\alpha|\le k} c_\alpha
        \int_\Omega (\partial^\alpha u)^2\,dx,
\]
with each \(\partial^\alpha u\) supplied by the closed-form derivative ops.

### Tensor-field divergence

For a 2-tensor field \(\sigma_{ij}\),
\[
    (\nabla\cdot\sigma)_i = \partial_j \sigma_{ij},
\]
summed over the spatial axes (Cartesian / flat space). The covariant version
with Christoffel corrections lives in `omnibias-geometry`.

### Wirtinger calculus

With \(z = x + i y\) and a complex field \(f = f_R + i f_I\),
\[
    \partial_z = \tfrac12(\partial_x - i\,\partial_y), \qquad
    \partial_{\bar z} = \tfrac12(\partial_x + i\,\partial_y),
\]
each returned as a `(real, imag)` pair built from the closed-form first
derivatives of \(f_R, f_I\). A holomorphic field satisfies
\(\partial_{\bar z} f = 0\) (Cauchy-Riemann).

### Finite-strain solid mechanics

Batched tensor algebra on the trailing \((d,d)\) block (`tensor_determinant`,
`tensor_inverse`, `tensor_cofactor` \(=(\det A)A^{-\top}\), `tensor_matmul`,
`tensor_trace`, `tensor_transpose`) underpins the finite-deformation kinematics:
the deformation gradient \(F = I + \nabla u\) (`deformation_gradient_finite`; the
displacement gradient \(\nabla u\) itself is the closed-form field Jacobian), the
right Cauchy-Green tensor \(C = F^\top F\), the Green-Lagrange strain
\(E = \tfrac12(C - I)\) (which vanishes for any rigid rotation \(F^\top F = I\)),
and the volume ratio \(J = \det F\).

The hyperelastic **stored energies** are exact algebraic functions of \(F\):

\[
    W_{\mathrm{StVK}} = \tfrac{\lambda}{2}(\operatorname{tr}E)^2 + \mu\operatorname{tr}(E^2),
    \quad
    W_{\mathrm{nH}} = \tfrac{\mu}{2}(I_1 - d) - \mu\ln J + \tfrac{\lambda}{2}(\ln J)^2,
\]

with a compressible (3-D) Mooney-Rivlin variant using the isochoric invariants
\(\bar I_1 = J^{-2/3}I_1\), \(\bar I_2 = J^{-4/3}I_2\). The **first
Piola-Kirchhoff stress** is the energy gradient \(P = \partial W/\partial F\),
computed by *reverse-mode* autodiff of the algebraic energy (exact to machine
precision; reverse mode is natural for a scalar energy and sidesteps a
`vmap`-of-forward-mode `linalg.det` batching defect). The second Piola-Kirchhoff
and Cauchy stresses follow as \(S = F^{-1}P\) and \(\sigma = J^{-1}PF^\top\).
Hand-derived closed forms are shipped and tested against the autodiff path:
\(S_{\mathrm{StVK}} = \lambda(\operatorname{tr}E)I + 2\mu E\) and
\(S_{\mathrm{nH}} = \mu(I - C^{-1}) + \lambda\ln J\,C^{-1}\). The general
anisotropic Hooke law is \(\sigma_{ij} = C_{ijkl}\varepsilon_{kl}\).

The **balance laws** assemble the stress divergence from the tangent modulus and
the closed-form second spatial derivatives of the displacement,

\[
    (\operatorname{Div}P)_i = \partial_J P_{iJ}
        = \underbrace{\frac{\partial P_{iJ}}{\partial F_{kL}}}_{A_{iJkL}\ \text{(autodiff-exact)}}
          \underbrace{\frac{\partial^2 u_k}{\partial x_J\,\partial x_L}}_{\text{closed-form (sigma-tower)}},
\]

giving `finite_strain_residual` \(=\operatorname{Div}P + f\) and
`elastodynamic_residual` \(=\rho\,u_{tt} - \operatorname{Div}P - f\) (with the
closed-form second time derivative \(u_{tt}\)). Validated by: rigid rotation
\(\Rightarrow E = 0,\ W = 0\); incompressible \(\Rightarrow J = 1\); a hand
uniaxial neo-Hookean stress; autodiff-vs-closed-form stress agreement; the StVK
tangent at \(F = I\) equal to the isotropic elasticity tensor; the small-strain
reduction to `navier_cauchy_residual`; and a finite-difference cross-check of
\(\operatorname{Div}P\) at finite strain. **Honesty:** elasticity /
hyperelasticity / elastodynamics are exact; history-dependent inelasticity
(plasticity return maps, viscoelastic internal variables) is iterative /
`numerical` and out of scope for the closed-form path.

### Magnetohydrodynamics

Single-fluid MHD in Alfven units (\(\mu_0 = \rho_0 = 1\)). The current closes from
Ampere's law (displacement current dropped in the MHD ordering)
\(J = \nabla\times B\) (`current_density`), giving the Lorentz force density
\(J\times B\) (`lorentz_force`), the magnetic pressure \(p_B = |B|^2/2\)
(`magnetic_pressure`), the Maxwell stress
\(T_{ij} = E_iE_j + B_iB_j - \tfrac12\delta_{ij}(|E|^2+|B|^2)\)
(`maxwell_stress_tensor`), and the solenoidal constraint \(\nabla\cdot B\)
(`magnetic_divergence`).

The **induction** residual is \(\partial_t B - \nabla\times(u\times B) - \eta\nabla^2B\)
(`induction_residual`). The ideal advection is expanded through the exact vector
identity

\[
    \nabla\times(u\times B) = u(\nabla\cdot B) - B(\nabla\cdot u)
        + (B\cdot\nabla)u - (u\cdot\nabla)B,
\]

so it is a pure composition of `divergence` and `advection` (no bespoke kernel);
a test cross-checks it against a fully independent symbolic \(\nabla\times(u\times B)\).
The **momentum** residual (`ideal_mhd_momentum_residual`) adds \(-J\times B\) to
the incompressible Navier-Stokes balance,
\(\rho(\partial_t u + (u\cdot\nabla)u) + \nabla p - J\times B - \nu\nabla^2u - f\).
Because \((\nabla\times B)\times B = (B\cdot\nabla)B - \nabla(|B|^2/2)\), the
magnetic pressure is carried inside \(J\times B\) and \(p\) is the thermal
pressure.

Validated by: a finite-amplitude nonlinear **Elsasser / Alfven wave**
\(u = \tfrac12(F+C)\), \(B = \tfrac12(F-C)\) with \(F = F(z - t)\) divergence-free
and \(C\) uniform, which makes both residuals vanish (checked symbolically with
`sympy.simplify` and numerically); the reductions \(B = 0 \Rightarrow\)
Navier-Stokes and \(u = 0 \Rightarrow\) resistive diffusion \(-\eta\nabla^2B\);
and torch/jax parity. **Honesty:** all MHD ops are closed-form / autodiff-exact.

### Kinetic theory

Phase-space transport of \(f(t, x, v)\). The **Vlasov** residual
\(\partial_t f + v\cdot\nabla_x f + (F/m)\cdot\nabla_v f\) (`vlasov_residual`)
weights the closed-form position derivatives by the *coordinate* velocity \(v\)
(read off `state.coords`) and the velocity derivatives by \(F/m\). The **BGK**
relaxation source is \(-(f - f_{eq})/\tau\) (`bgk_collision`), with
`bgk_vlasov_residual` \(= \mathcal L f + (f - f_{eq})/\tau\). The local
Maxwellian \(f_{eq} = n(m/2\pi T)^{d/2}\exp(-m|v-u|^2/2T)\) (`maxwellian`) is a
closed-form Gaussian of the velocity coordinates. Velocity moments
\(n = \int f\,dv\), \(\int v f\,dv\), \(\int\tfrac12|v|^2 f\,dv\)
(`number_density` / `momentum_density` / `kinetic_energy_density`) contract the
distribution against a supplied velocity-space quadrature rule.

Validated by: force-free transport \(f = g(x - vt)\) having identically zero
residual (1D-1V and 2D-2V); a symbolic cross-check of the forced residual; a
Maxwellian recovering \(n\), \(n u\), \(n(\tfrac12|u|^2 + dT/2m)\) under
Gauss-Legendre quadrature; and BGK moment-linearity (mass / momentum
conservation structure). **Honesty:** Vlasov transport, BGK and the Maxwellian
are closed-form; the full quadratic, non-local **Boltzmann collision integral is
`numerical`** (velocity-space quadrature) and is deliberately not shipped as a
closed-form op (enforced by a test).

## Numerical notes

- All tests run in float64. Cross-backend parity is asserted at
  `rtol=1e-12, atol=1e-12`; pinned regressions at `rtol=1e-12, atol=1e-14`.
- Gauss-Legendre and Gauss-Hermite nodes/weights are computed with numpy in
  double precision; Monte-Carlo integration uses a seeded generator for
  reproducibility and reports the standard deviation as an error estimate.
