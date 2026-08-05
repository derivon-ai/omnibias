# omnibias-convex

Differentiable + certified convex optimization (LP / QP) on the omnibias stack: a
closed-form-Hessian log-barrier interior-point solver, an `argmin` that is
differentiable through the KKT system (OptNet / cvxpylayers style), and an
optional rigorous optimality certificate from `omnibias.core.verified`.

The problem form is

\[
\min_x \; \tfrac12 x^\top Q x + c^\top x \quad\text{s.t.}\quad A x \le b,
\]

with `Q` positive semidefinite (`Q = 0` recovers an LP). The log-barrier
subproblem has the closed-form Hessian `H = t Q + A^T diag(1/s^2) A`
(`s = b - A x`), solved with `jnp.linalg.solve` along a short central path -- the
same closed-form-Hessian Newton pattern as `omnibias.curvature.mse_newton_step`.

The differentiators are **differentiability, batched/GPU execution, and a
certificate** -- an LP/QP you can drop into a network and train through, or solve
by the thousands on-device. For last-digit accuracy on a single program the Newton
interior point (or a simplex crossover from its almost-exact iterate) is the right
tool; the two solvers are complementary.

## Problem & solution containers

::: omnibias.convex.problem
    options:
      show_root_heading: false
      heading_level: 3

## Solver (JAX)

::: omnibias.convex.jax.solver
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable layer (JAX)

::: omnibias.convex.jax.layer
    options:
      show_root_heading: false
      heading_level: 3

## Solver & layer (torch)

::: omnibias.convex.torch.solver
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.convex.torch.layer
    options:
      show_root_heading: false
      heading_level: 3

## Gradient-descent temperature-collapse penalty solver

A first-order alternative to the Newton interior point: each constraint hyperplane
becomes a tempered temperature-collapse unit (a softplus), so the LP/QP becomes a smooth
gradient-descent problem with the closed-form gradient
`c + Q x + mu A^T sigma(beta (A x - b))`, annealed along a `beta` / `mu` homotopy.
Bit-identical JAX and torch twins. See the
[cookbook](../cookbook/differentiable-certified-lp.md#gradient-descent-temperature-collapse-solver).

::: omnibias.convex.jax.penalty
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.convex.torch.penalty
    options:
      show_root_heading: false
      heading_level: 3

## Verified optimality certificate

::: omnibias.convex.certify
    options:
      show_root_heading: false
      heading_level: 3

## Warm starts (temperature-collapse geometry)

::: omnibias.convex.warm_start
    options:
      show_root_heading: false
      heading_level: 3

A worked walkthrough is in the
[differentiable + certified LP cookbook](../cookbook/differentiable-certified-lp.md).

Status: Alpha (`0.1.0a1`).
