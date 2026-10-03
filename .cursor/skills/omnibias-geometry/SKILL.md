---
name: omnibias-geometry
description: Compute metric, Christoffel symbols, covariant derivative, Laplace-Beltrami, Riemann / Ricci / scalar curvature, geodesics, exterior calculus, Einstein tensor, and pullback metrics of learned charts. Use when inventing a manifold operator, a gauge/holonomy primitive, or when the user mentions curvature, geodesics, or differential forms.
---

# Differential geometry on the field substrate

`omnibias-geometry` builds on `omnibias-fields` with PyTorch + JAX parity:
metric, connection, curvature, geodesics, exterior calculus, and the
general-relativity layer. Field-function derivatives ride the closed-form
tower; metric derivatives are exact forward-mode autodiff of the analytic
per-point metric.

## Why nested AD fails

Riemann curvature needs second derivatives of the metric and third derivatives
of a learned chart. Nested AD through `g = J^T h J` is a deep graph that
forks per backend and rarely exposes Christoffel or Laplace-Beltrami as first-class
ops. Generic geometry libraries do not pull back a jet from one forward pass,
and they have no holonomy-band or atlas-cocycle consumer on a partition chart.

## What only this tower unlocks

`pullback_metric` consumes a supplied chart Jacobian (`g = J^T h J`) when
provided, and otherwise uses its labeled forward-mode autodiff fallback.
`grad f`, `hess f`, and the field part of `Delta_g f` are exact `sigma^(n)`.
Gauge transfer and holonomy-band primitives compose the same substrate.
That combination — exact field jets plus an analytic metric — is the workload
nested AD does not sustain at curvature order.

## Use

| You want | Import from | Key entry points |
| --- | --- | --- |
| Metric / connection / curvature (torch) | `omnibias.geometry.torch.ops` | Christoffel, covariant derivative, `laplace_beltrami`, Riemann / Ricci / `scalar_curvature`, geodesics |
| JAX twin | `omnibias.geometry.jax.ops` | bit-identical (parity ~1e-9 in float64) |
| Learned-manifold pullback | `omnibias.geometry` | `ChartSpec`, `metric_spec_from_chart`, `pullback_metric` |
| General relativity | `omnibias.geometry` | `einstein_tensor`, `einstein_equation_residual`, `kretschmann_scalar`, `weyl_tensor` |
| Exterior / de Rham | `omnibias.geometry` | `exterior_derivative`, `hodge_star`, `hodge_laplacian`, `betti_number`, `gauss_bonnet_euler` |
| Chart scan | `omnibias.geometry.scan` | `chart_scan` via pullback metric |
| Holonomy band | `omnibias.geometry.gauge.band` | `HolonomyBand`, `band_holonomy` |
| Gauge transfer | `omnibias.geometry.gauge.transfer` | holonomy trials on a fixed matrix |
| Volume-uniform strong-coupling family | `omnibias.geometry.gauge.transfer.strong_coupling` | `volume_uniform_strong_coupling_family` |
| Atlas cocycle | `omnibias.geometry.atlas.cocycle` | jet cocycle on partition charts |

`pullback_metric` expects batched coordinates `(B, d)` and returns `(B, d, d)`.
Reshape a single point to `(1, -1)` and index `[0]`. Label field-function
derivatives as closed form and metric derivatives as exact forward-mode of
the analytic metric.

Cookbook: `docs/cookbook/geometry-sphere.md`,
`docs/cookbook/pullback-learned-manifolds.md`.

## Extend

Neuromanifold APIs live in `omnibias.geometry.neuromanifold`: explicit
realizations with live JVP/VJP and parameter jets, observation metrics, symmetry
actions, weighted tangent/normal geometry, and exact affine quotient charts.
Numerical rank is a diagnostic; a certified lower/upper rank needs a checked
minor/factorization. A sampled Jacobian kernel never establishes a nonlinear
symmetry. Monomial curves have a supported exact stratum classifier; general
singularities remain finite-order diagnostics. Collision banks live in the
backend packages; their certified acceptance lives downstream in verify.
See `docs/api/neuromanifold.md` and its 16-gap acceptance map. The geometry
package must not import the copyleft certified consumers.

`geometry.continuation` follows finite residual branches and localizes fold/Hopf
systems. `continuation_directional.directional_residual_family` adapts first
through third directional callbacks under an explicit dense-coefficient budget.
`continuation_switch.switch_equilibrium_branch` corrects transverse seeds at
supported simple equilibrium branch points and refuses deficient/degenerate
cases. It is a numerical producer; validated segments/events live downstream
in dynamics. An analytic Bratu branch supplies the PDE reference. None of these
finite systems silently establishes continuum discretization error or coverage.

For Hilbert-XVI Part A, `omnibias.geometry.algebraic` certifies rational
polygonal oval barriers and `omnibias.geometry.part_a_obstruction` audits the
22-annulus wide/deep layout against the fixed-layout/SOS obstruction boundary.
A fixed polygon layout or symmetry ansatz never excludes an isotopy scheme
without a proved reduction covering every realization.

- Source: `packages/omnibias-geometry`. Field ops stay in `omnibias-fields`.
- Tests: `python -m pytest packages/omnibias-geometry/tests -q`.
- Compose with `omnibias-fields`, `omnibias-pinn`, `omnibias-variational`,
  `omnibias-partition`, `omnibias-verify`, `omnibias-frontier`.
- Gauge lives here (`omnibias.geometry.gauge`), not a separate distribution.

## Next invention

A learned atlas whose jet cocycle and pullback Laplace-Beltrami are
bit-identical across torch/jax, and whose scalar-curvature residual is a
sealed Interval on a named chart.

## Bakeoffs

Field-side Laplacian cost: `docs/benchmarks/laplacian_scaling.json`.
Holonomy: `docs/benchmarks/holonomy_band_smoke.json`.

## Further references

- API: `docs/api/geometry.md`, `docs/api/volume_uniform_ym.md`, `docs/api/holonomy_band.md`, `docs/api/equivariant_scan.md`
- Handbook: `docs/handbook/03-differential-geometry.md`, `docs/handbook/04-exterior-calculus.md`
