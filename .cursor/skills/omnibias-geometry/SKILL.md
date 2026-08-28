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

`pullback_metric` reads the chart's own jet (`g = J^T h J`) in closed form.
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
