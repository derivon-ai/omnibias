---
name: omnibias-curvature-optim
description: Use omnibias second-order optimizers and closed-form curvature -- CubicNewton, GaussNewton, KFAC, TrustRegionNewtonCG, JetSubspaceTensor, NaturalGradient, and exact Fisher / Hessian sharpness. Use when using omnibias to run a curvature-aware optimizer, compute an exact Hessian / Fisher / natural-gradient step, or measure and regularize sharpness, or when the user mentions second-order optimization, natural gradient, KFAC, or Fisher information.
---

# Using omnibias: curvature-aware optimization

Because the activation derivative towers are closed form, omnibias exposes exact
second-order information (Hessian, Fisher, cubic tensor) and optimizers that
consume it.

## Import map

| You want | Import from | Key entry points |
| --- | --- | --- |
| Second-order optimizers (PyTorch) | `omnibias.torch.optim` | `CubicNewton`, `GaussNewton`, `KFAC`, `TrustRegionNewtonCG`, `StochasticNewtonCG`, `JetLBFGS`, `DiagonalCurvature`, `JetSubspaceTensor`, `NaturalGradient` |
| Natural-gradient / Fisher primitives | `omnibias.torch.optim` | `natural_gradient_direction`, `gauss_newton_fisher`, `gauss_newton_fisher_matvec` |
| Functional natural-gradient step | `omnibias.curvature.natural_gradient` | `natural_gradient_step`, `glm_natural_gradient_step` |
| Exact sharpness / Hessian (one-layer, JAX) | `omnibias.curvature.sharpness` | `mse_loss_hessian`, `hessian_trace`, `hessian_top_eigenvalue`, `sam_objective` |
| Matrix-free deep curvature (PyTorch) | `omnibias.curvature.torch` | `hvp`, `dense_hessian`, `top_eigenvalue`, `sharpness_aware_loss` |

## Gotchas that bite

- **`NaturalGradient` is metric-pluggable.** Pass a metric provider (a dense SPD tensor or a matrix-free matvec callable) built by the caller -- e.g. a Fisher from `gauss_newton_fisher` or a pullback metric from `omnibias.geometry`. The optimizer module stays dependency-free by design.
- **`omnibias.curvature.sharpness` is a one-layer primitive**; use `omnibias.curvature.torch` (matrix-free HVPs) for arbitrary depth.
- **`hvp` / `dense_hessian` are exact** (autograd double-backward on the closed-form tower); `hutchinson_*` are unbiased stochastic estimators -- for a reported number on a small net prefer `dense_hessian`.
- Curvature torch helpers need the extra: `pip install "omnibias-curvature[torch]"`.

## More detail

- API: [curvature](https://github.com/derivon-ai/omnibias/blob/main/docs/api/curvature.md), [torch](https://github.com/derivon-ai/omnibias/blob/main/docs/api/torch.md)
- Handbook: [information geometry & natural gradient](https://github.com/derivon-ai/omnibias/blob/main/docs/handbook/07-information-geometry.md)
