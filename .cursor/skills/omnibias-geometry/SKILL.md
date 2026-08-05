---
name: omnibias-geometry
description: Use omnibias for differential geometry -- metric, Christoffel symbols, covariant derivative, Laplace-Beltrami, Riemann / Ricci / scalar curvature, geodesics, exterior calculus, Einstein tensor, and pullback metrics of learned charts. Use when using omnibias to compute curvature or geometric operators, or when the user mentions manifolds, metrics, curvature, geodesics, or differential forms.
---

# Using omnibias: differential geometry

`omnibias-geometry` builds on `omnibias-fields` with PyTorch + JAX parity:
metric, connection, curvature, geodesics, exterior calculus, and the
general-relativity layer.

## Import map

| You want | Import from | Key entry points |
| --- | --- | --- |
| Metric / connection / curvature ops (torch) | `omnibias.geometry.torch.ops` | Christoffel, covariant derivative, `laplace_beltrami`, Riemann / Ricci / `scalar_curvature`, geodesics |
| JAX twin | `omnibias.geometry.jax.ops` | bit-identical (parity ~1e-9 in float64) |
| Learned-manifold pullback metric | `omnibias.geometry` | `ChartSpec`, `metric_spec_from_chart`, `pullback_metric` (`g = J^T h J`) |
| General relativity | `omnibias.geometry` | `einstein_tensor`, `einstein_equation_residual`, `kretschmann_scalar`, `weyl_tensor` |
| Exterior / de Rham | `omnibias.geometry` | `exterior_derivative`, `hodge_star`, `hodge_laplacian`, `betti_number`, `gauss_bonnet_euler` |

## Honesty labels (state these when you report results)

- **Field-function derivatives** (`grad f`, `hess f`, the field part of `Delta_g f`) are exact **closed form** (the sigma tower).
- **Metric derivatives** inside Christoffel / Riemann / Ricci / scalar curvature are exact **forward-mode autodiff of the analytic per-point metric** -- machine-precision, but not "closed form". Say which is which.
- `pullback_metric` is fully closed form (it reads the chart's own jet).
- Sibling packages: `omnibias.fractional` is **non-local / grid-based, NOT closed form**; `omnibias.score` is a pure composition of the field gradient / Hessian ops.

## Gotchas that bite

- **`pullback_metric` expects batched coordinates** shaped `(B, d)` and returns `(B, d, d)`. Reshape a single point to `(1, -1)` and index `[0]`.
- Combinatorial topology (persistent homology, simplicial homology) is out of scope; the de Rham Betti / degree integrals are numerical (quadrature) and certifiable via an `omnibias.core.verified.Interval` enclosure.

## Canonical runnable examples / more detail

- Cookbook: [geometry on the sphere](https://github.com/derivon-ai/omnibias/blob/main/docs/cookbook/geometry-sphere.md), [pullback metric on learned manifolds](https://github.com/derivon-ai/omnibias/blob/main/docs/cookbook/pullback-learned-manifolds.md)
- Handbook: [differential geometry](https://github.com/derivon-ai/omnibias/blob/main/docs/handbook/03-differential-geometry.md), [exterior calculus](https://github.com/derivon-ai/omnibias/blob/main/docs/handbook/04-exterior-calculus.md)
- API: [geometry](https://github.com/derivon-ai/omnibias/blob/main/docs/api/geometry.md)
