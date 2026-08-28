---
name: omnibias-curvature
description: Use closed-form Hessian / Fisher / KFAC factors and second-order optimizers — CubicNewton, GaussNewton, TrustRegionNewtonCG, JetSubspaceTensor, NaturalGradient — built on exact sigma^(n). Use when inventing a curvature-aware trainer, computing an exact HVP / Fisher / natural-gradient step, or measuring sharpness.
---

# Closed-form curvature and second-order training

Because activation derivative towers are closed form, omnibias exposes exact
second-order information (Hessian, Fisher, cubic tensor) and optimizers that
consume it. This is the same package that used to be reached through a
consumer alias; there is one skill name: `omnibias-curvature`.

## Why nested AD fails

A Hessian-vector product via nested reverse-mode AD is a second full sweep per
direction. Monte-Carlo Fisher estimates add noise on top. At PINN and jet
orders the optimizer's curvature estimate is the bottleneck: cubic Newton,
KFAC, and sharpness all want `sigma''` that generic autodiff either cannot
afford or approximates. There is no pack-Fisher metric on a vanilla Adam stack.

## What only this tower unlocks

One-hidden-layer Riccati fields have closed-form parameter Hessians and
Gauss-Newton Fisher / KFAC factors from `sigma'` / `sigma''`. Matrix-free
`hvp` / `dense_hessian` are exact double-backwards on that tower. Natural
gradient is metric-pluggable (Fisher or a pullback metric from
`omnibias-geometry`). `docs/benchmarks/derivative_order.json` and the
optimizer PINN bakeoff are the named workloads nested AD does not sustain.

## Use

| You want | Import from | Key entry points |
| --- | --- | --- |
| Second-order optimizers (PyTorch) | `omnibias.torch.optim` | `CubicNewton`, `GaussNewton`, `KFAC`, `TrustRegionNewtonCG`, `StochasticNewtonCG`, `JetLBFGS`, `DiagonalCurvature`, `JetSubspaceTensor`, `NaturalGradient` |
| Natural-gradient / Fisher primitives | `omnibias.torch.optim` | `natural_gradient_direction`, `gauss_newton_fisher`, `gauss_newton_fisher_matvec` |
| Functional natural-gradient step | `omnibias.curvature.natural_gradient` | `natural_gradient_step`, `glm_natural_gradient_step` |
| Exact sharpness / Hessian (one-layer, JAX) | `omnibias.curvature.sharpness` | `mse_loss_hessian`, `hessian_trace`, `hessian_top_eigenvalue`, `sam_objective` |
| Matrix-free deep curvature (PyTorch) | `omnibias.curvature.torch` | `hvp`, `dense_hessian`, `top_eigenvalue`, `sharpness_aware_loss` |
| Pack Fisher metric | `omnibias.curvature.information` | pack Fisher, not scalar `A''(theta)` |
| Composed two-layer Newton | `omnibias.{torch,jax}.optim_composed` | joint two-layer Newton |
| Sharpness schedule / regularizer | `omnibias.{torch,jax}.optim_sharpness`, `.optim_sharp_loss` | exact-HVP `lambda_max` |

`NaturalGradient` takes a metric provider (dense SPD tensor or matvec) built
by the caller. `omnibias.curvature.sharpness` is the one-layer primitive;
use `omnibias.curvature.torch` HVPs for arbitrary depth. Prefer `dense_hessian`
for a reported number on a small net; `hutchinson_*` are unbiased estimators.
Torch helpers: extra `omnibias-curvature[torch]`.

## Extend

- Source: `packages/omnibias-curvature`. Optimizers also live in `omnibias.torch.optim`.
- Tests: `python -m pytest packages/omnibias-curvature/tests -q`.
- Compose with `omnibias-torch`, `omnibias-jax`, `omnibias-geometry` (pullback
  metric), `omnibias-tab` (Newton boosting), `omnibias-pinn`, `omnibias-verify`.

## Next invention

A pack-Fisher metric whose exact `lambda_max` sets a cubic Newton step that
beats SAM on a named PINN residual, with a sealed sharpness enclosure.

## Bakeoffs

`docs/benchmarks/derivative_order.json`, `docs/benchmarks/optimizer_pinn.json`,
`docs/benchmarks/information_geometry.json`.

## Further references

- API: `docs/api/curvature.md`, `docs/api/torch.md`
- Handbook: `docs/handbook/07-information-geometry.md`
