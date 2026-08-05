# omnibias-geometry

**Status: Beta (v0.2.0).**

Differential geometry on manifolds, built on the `omnibias-fields` substrate
with cross-backend (PyTorch + JAX) parity: metric, Christoffel symbols,
covariant derivative, the Laplace-Beltrami operator, Riemann / Ricci / scalar
curvature, geodesics, and exterior calculus.

## Two exact mechanisms

`omnibias-geometry` is **exact** end-to-end. **Field-function** derivatives
(`grad f`, `hess f`, the field part of `Δ_g f`) use the **closed-form
sigma-tower** in `omnibias-fields` — one forward pass per derivative order,
bit-identical across torch / JAX, `O(1)` in input dim. **Metric** derivatives
inside Christoffel / Riemann / Ricci / scalar curvature use **forward-mode
autodiff of the analytic per-point metric `g_point(x)`** — exact to machine
precision (not a finite-difference approximation). Both paths agree with a
sympy symbolic reference within float64 ULPs.

See [`docs/scope-and-guarantees.md`](../scope-and-guarantees.md) for the
project-wide definition of "closed-form" vs "autodiff-exact".

## Learned manifolds (pullback metric)

A `ChartSpec` describes an immersion `phi: R^d -> R^n` (an analytic or
neural-network "chart"). `metric_spec_from_chart` turns it into the pullback
metric `g = J^T h J` (`J = d phi / dx` by forward-mode autodiff, `h` the ambient
metric, Euclidean by default), and `pullback_metric` evaluates it batched. Because
every connection / curvature / field operator only reads `manifold.metric.g_point`,
the entire stack (Christoffel, Riemann/Ricci/scalar curvature, Laplace-Beltrami,
geodesics) works on *learned* manifolds with no further changes. Validated by
recovering the round-sphere metric and `R = 2` from the standard `S^2` embedding.

See [`GEOMETRY_DERIVATIONS.md`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-geometry/GEOMETRY_DERIVATIONS.md)
for the index conventions and the validation against the round sphere.

## General relativity: Einstein tensor and curvature invariants

On top of Riemann / Ricci / scalar curvature the package exposes the
general-relativity layer: `einstein_tensor` (`G = Ric - ½R g`),
`einstein_equation_residual` (`G + Λg - κT`, `stress_energy=None` is vacuum),
`lowered_riemann`, the `kretschmann_scalar` invariant, the `weyl_tensor`
(conformal, `d ≥ 3`), and `geodesic_deviation` (the Jacobi/tidal acceleration).
These are the same autodiff-exact-metric path as the rest of the package (no
numerical-relativity evolution). Validated on Schwarzschild vacuum (`G = 0`,
`K = 48 M²/r⁶`), de Sitter FRW (`G₀₀ = 3H²`), and the round `S³`
(`G = -(1/R²)g`, `Weyl = 0`, `K = 12/R⁴`), plus the contracted Bianchi identity
`∇^μ G_{μν} = 0`.

## de Rham topology: Betti numbers, degree, Gauss-Bonnet

The de Rham slice of algebraic topology lives on the closed-form substrate.
`hodge_laplacian` generalises the scalar Hodge Laplacian to a `k`-form
`Δ = dδ + δd`: on 0-forms it is the full curved-manifold Laplacian, and on
`k ≥ 1` forms it is the constant-metric componentwise Laplacian
`(Δω)_I = -g^{mn}∂_m∂_n ω_I` (a genuinely curved `k`-form Laplacian, which
carries the Weitzenböck curvature term, honestly raises `NotImplementedError`).
`hodge_laplacian_matrix` assembles `Δ` in a finite Fourier form basis and
`betti_number` reads off the harmonic (kernel) dimension — the de Rham Betti
number — with `harmonic_projection` returning the kernel component of a
coefficient vector. `winding_number` is the degree of a circle map `S¹ → S¹`;
`map_degree` is the degree of a map `M² → S²` (the normalised pullback of the
target area form); and `gauss_bonnet_euler` is the Euler characteristic of a
closed surface `χ = (1/2π)∫ K dA` (reusing `scalar_curvature`). The Betti/degree
integrals are numerical (quadrature + a nullity count) and certifiable by an
`omnibias.core.verified.Interval` enclosure. Validated on the flat torus
(`b = (1, 2, 1)`), the identity `S² → S²` (degree 1), circle windings, and
`χ(S²) = 2`. Combinatorial topology (`π_n`, persistent homology, simplicial
`ℤ`-homology, Smith normal form) is **out of thesis** — see
[`docs/scope-and-guarantees.md`](../scope-and-guarantees.md).

## Surface integration of differential forms

A `k`-form integrates over the image of a `k`-dimensional `ChartSpec` immersion
`phi: R^d -> R^n` by the change-of-variables identity `∫_M ω = ∫_box φ*ω`, where
the pullback `φ*ω` is the generalised Jacobian determinant
(`pullback_form_components`, expanded by the Leibniz permutation sum in pure
Python so the two backends stay bit-identical). `integrate_form` takes a name-form
plus a `FieldState` evaluated at the chart image points; `integrate_form_values`
is the low-level twin that takes already-evaluated component tensors and the
Jacobian (handy for an analytic `dω` or a standalone check). The metric companions
integrate against the Riemannian volume element `sqrt(|det g|)` of the pullback
metric `g = J^T h J`: `volume_element` (per node), `surface_area` (its integral),
and `surface_integral` (a scalar field `∫_M f dA`).

**Honesty label.** The *integrand* is **exact** — form components are closed-form
field values / derivatives and the chart Jacobian `J = d phi / dx` is exact
forward-mode autodiff of the (analytic or neural) chart. The *integral* is a
**numerical quadrature** (`QuadratureSpec`, Gauss-Legendre): exact for polynomials
up to the rule degree and convergent otherwise — the same framing as the
Betti / degree / Gauss-Bonnet integrals above. Validated against the unit-square
`dx^dy = 1`, the closed 1-form circulation `∮(-y dx + x dy) = 2πR²`, the unit
sphere `surface_area = 4π`, and Green's theorem as a Stokes self-test (a boundary
line integral equal to the interior area integral of `dω` via `exterior_derivative`).

## Piecewise geometry: region-wise Riemannian atlas

`omnibias.geometry.atlas` (a bridge on the [`omnibias-partition`](partition.md)
keystone) blends per-region SPD metrics with a certified soft
partition-of-unity: `g(x) = Σ_l w_l(x) G_l(x)`. A convex combination of SPD
matrices is SPD, so the blended metric is a **provably valid Riemannian metric
everywhere** (unit-tested) — each leaf carries its own geometry, scalar
curvature differs by region, and a geodesic bends across the (annealed)
interface, while the whole connection / curvature / geodesic stack is reused
unchanged. `AtlasSpec` is backend-neutral; `blended_metric` / `atlas_manifold`
build a `MetricSpec` / `ManifoldSpec` on torch or JAX (parity `rtol=1e-9`,
float64). The `beta -> inf` gate hardening is the feasibility / temperature
sense of "collapse", never the founding `delta -> 0` bias collapse; metric
derivatives stay autodiff-exact.

::: omnibias.geometry.atlas
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.atlas.torch
    options:
      show_root_heading: false
      heading_level: 3

## Schemas

::: omnibias.geometry._core
    options:
      show_root_heading: false
      heading_level: 3

## Ops (torch)

::: omnibias.geometry.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.geometry.jax.ops`) is the bit-identical twin;
cross-backend tests assert agreement to `rtol=1e-9` in float64.
