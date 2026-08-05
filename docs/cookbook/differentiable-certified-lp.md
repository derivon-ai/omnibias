# Differentiable + certified LP / QP

`omnibias-convex` makes `argmin` a layer. It solves the quadratic program

\[
\min_x \tfrac12 x^\top Q x + c^\top x \quad\text{s.t.}\quad A x \le b,
\qquad Q \succeq 0,
\]

(the linear program is the `Q = 0` case), it is **differentiable** in all of
`Q, c, A, b`, and it can hand back a **rigorous certificate** that bounds how far
the returned point is from the true optimum. The same solver and layer exist as
bit-identical jax and torch twins.

```python
import jax.numpy as jnp
from omnibias.convex.jax import solve_qp, qp_layer
from omnibias.convex import certify_qp_optimum

Q = jnp.eye(3)
c = -jnp.array([1.5, -2.0, 0.7])
A = -jnp.eye(3)              # x >= 0
b = jnp.zeros(3)

sol = solve_qp(Q, c, A, b)  # ConvexSolution(x, dual, slack, obj, gap, ...)
sol.x                       # relu of the target, to the barrier tolerance
```

## The solver: a closed-form-Hessian log-barrier method

The interior-point method follows the central path of the log-barrier
`phi_t(x) = t (1/2 x^T Q x + c^T x) - sum_i log(b_i - a_i^T x)`. Its gradient and
**Hessian are closed form** -- no autodiff, no finite differences:

\[
\nabla\phi_t = t(Qx+c) + A^\top s^{-1},\qquad
\nabla^2\phi_t = t\,Q + A^\top \operatorname{diag}(s^{-2})\,A,
\qquad s = b - Ax .
\]

Each centering step is a damped Newton solve of that Hessian (`jnp.linalg.solve`),
the barrier weight `t` grows geometrically, and the surrogate duality gap `m / t`
drives termination. When the origin is infeasible a **phase-1 LP** (the same
solver, minimising a slack variable) finds a strictly feasible start. The result
carries the primal `x`, the dual `lambda = 1/(t s) >= 0`, the slacks, the gap, and
the outer / Newton iteration counts.

Its edge is not raw single-solve speed but that it is *differentiable* and
*certifiable*: for last-digit accuracy on one program a simplex crossover from its
iterate is the natural finisher, while the value here is an LP/QP you can embed in
a model and prove bounds on.

## `argmin` as a differentiable op (KKT implicit-function theorem)

`qp_layer` / `lp_layer` differentiate through the optimum without unrolling the
solver. At a KKT point the residual `r(x, lambda) = 0` has Jacobian

\[
K = \begin{bmatrix} Q & A^\top \\
    \operatorname{diag}(\lambda)\,A & \operatorname{diag}(A x - b) \end{bmatrix},
\]

so for an upstream cotangent `g = dL/dx` one adjoint solve `K^T [y_x; y_l] = -[g; 0]`
yields every parameter gradient (Amos & Kolter's OptNet identities):

\[
\nabla_c L = y_x,\quad \nabla_b L = -\lambda \odot y_l,\quad
\nabla_A L = \lambda y_x^\top + (\lambda \odot y_l) x^\top,\quad
\nabla_Q L = \tfrac12 (y_x x^\top + x y_x^\top).
\]

```python
import jax

def loss(c):
    x = qp_layer(Q, c, A, b)        # differentiable argmin
    return jnp.sum(x ** 2)

grad_c = jax.grad(loss)(c)          # backprops through the optimiser
```

JAX runs the (Python-control-flow) interior-point loop eagerly inside
`jax.pure_callback` and attaches the adjoint with `custom_vjp`, so `qp_layer` is
usable under `grad` / `jit`. The torch twin is a `torch.autograd.Function` with
the identical backward. Their gradients are bit-identical:

```python
import torch
from omnibias.convex.torch import qp_layer as qp_layer_torch
```

A linear program's optimum is a vertex, so it is locally constant in `c`
(`grad_c = 0` almost everywhere) but moves with `b` -- `lp_layer` reflects exactly
that.

## A rigorous optimality certificate

A float solve gives a number; `certify_qp_optimum` gives a **proof**. It returns a
`Certificate` whose `enclosure` is an `omnibias.core.verified.Interval` that
provably contains the true optimal value `f*`, computed in outward-rounded
interval arithmetic independent of the solver:

```python
from omnibias.convex import certify_qp_optimum

cert = certify_qp_optimum(Q, c, A, b, sol.x, sol.dual)
cert.enclosure          # Interval [lower, upper] with  lower <= f* <= upper
cert.gap                # certified suboptimality (upper - lower)
cert.primal_feasible    # x verified to satisfy A x <= b (rigorously)
```

The two bounds:

- **Upper** -- if `x` is rigorously primal feasible, `f* <= f0(x)`, evaluated with
  interval rounding.
- **Lower** -- weak duality: for any `lambda >= 0`,
  `f* >= -1/2 v^T Q^{-1} v - b^T lambda` with `v = c + A^T lambda`. The term
  `v^T Q^{-1} v` is bounded rigorously using the Neumann certificate
  `neumann_inverse_norm_bound` for `||Q^{-1}||_inf` (so a strictly convex `Q` is
  required).

The interval width shrinks as the solve tightens (`v -> 0`, the primal/dual
objectives meet). Because feasibility is checked rigorously, the certificate is
evaluated at the solver's strictly interior point -- a point exactly on a
constraint cannot be certified feasible in floating point, which is the honest
answer.

## Warm starts from temperature-collapse geometry

This rides the `beta -> inf` *temperature collapse* axis: a unit's decision boundary
is an exact hyperplane, and the optimum of a non-degenerate LP is the
intersection of `n` active hyperplanes -- a vertex. A model that predicts the
**active set** can therefore guess the solution cheaply. `omnibias.convex.warm_start`
turns such a prediction into a strictly feasible start:

```python
from omnibias.convex import predicted_vertex, geometry_warm_start, active_set_warm_start

# Back-off needs a strictly feasible anchor, and the default anchor is the origin,
# so use a box rather than the `x >= 0` cone above (whose vertex is on the boundary).
A_box = jnp.concatenate([-jnp.eye(3), jnp.eye(3)])
b_box = jnp.ones(6)
scores = jnp.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])    # a predictor's active-set scores

x_hint = active_set_warm_start(A_box, b_box, scores)  # vertex of top-n constraints, backed inside
sol = solve_qp(Q, c, A_box, b_box, x0=x_hint)         # strictly feasible x0 => skips phase-1
```

`predicted_vertex` solves the `n` highest-scored constraint rows; `geometry_warm_start`
backtracks any hint to a strictly feasible interior point via a ratio test;
`active_set_warm_start` composes the two. A feasible `x0` lets the solver skip the
phase-1 LP entirely -- the saving shows up directly in
`ConvexSolution.newton_iterations`. Each of the three returns `None` rather than an
infeasible guess when the prediction is rank-deficient or no strictly feasible anchor
exists, and `solve_qp` then falls back to its own phase-1.

## Gradient-descent (temperature-collapse) solver

The warm start turns that hyperplane geometry into a *hint*. Pushed all the way, it turns the
whole LP into a **gradient-descent problem**. Each constraint `a_i^T x <= b_i` is a
hyperplane; the temperature-collapse unit `sigma(beta (a_i^T x - b_i))` has exactly that
hyperplane as its decision boundary, and integrating it gives the smooth hinge
penalty `rho_beta(u) = softplus(beta u) / beta -> max(u, 0)` as `beta -> inf` (the
same `beta -> inf` tempering as `omnibias.binary`'s tanh trick). Summing the
penalties gives a smooth objective

\[
F(x) = c^\top x + \tfrac12 x^\top Q x + \mu \sum_i \rho_\beta(a_i^\top x - b_i),
\]

whose gradient is **closed form** -- a weighted sum of temperature-collapse sigmoids, no
autodiff:

\[
\nabla F = c + Q x + \mu\, A^\top \sigma\!\big(\beta (A x - b)\big).
\]

```python
from omnibias.convex.jax import solve_lp_penalty, solve_qp_penalty, penalty_gradient

sol = solve_lp_penalty(c, A, b)         # accelerated GD along a beta/mu homotopy
sol.x                                   # -> the LP vertex, to a first-order tolerance
sol.dual                                # lambda = mu * sigma(beta (A x - b)) >= 0
```

`solve_lp_penalty` / `solve_qp_penalty` minimise `F` by accelerated (Nesterov)
gradient descent with a **closed-form Lipschitz step** `eta = step_safety / L`,
`L = ||Q||_2 + (mu beta / 4) ||A||_2^2`, while annealing the sharpness `beta` and
the penalty weight `mu` upward (`PenaltyOptions`). A small proximal-point term
anchors each stage to the previous iterate so the exterior penalty cannot run off
to infinity while `mu` is below the largest optimal multiplier; it vanishes at the
fixed point, so the recovered optimum is unbiased. The KKT multiplier estimate
falls straight out of the temperature-collapse activation, `lambda = mu sigma(beta u) >= 0`,
which you can feed to `certify_qp_optimum` for a rigorous cross-check.

Because the penalty is **exterior**, the solver needs no strictly feasible start
and no phase-1 -- pass any `x0` (even wildly infeasible). And because `F` is smooth
everywhere, it has **informative gradients everywhere** (unlike `lp_layer`, whose
`grad_c = 0` almost everywhere at the vertex): `penalty_descent` is a pure,
`jit` / `grad`-friendly primitive.

Honest scope: this is a first-order, GPU/batch-friendly, fully differentiable
*tolerance* solver (it matches the Newton interior point to ~`1e-3`–`1e-4` on
small problems), the **complement** of -- not a replacement for -- the Newton
`solve_lp` / `solve_qp` (which reach ~`1e-12`). See
[`convex_lp_gradient_descent.py`](../examples/convex_lp_gradient_descent.py) for a
runnable walkthrough (closed-form gradient check, `beta`-annealing convergence,
comparison to `solve_lp` / scipy, and a certified QP).

## Relationship to `CvxLogistic` / `cvxlayer`

`omnibias.torch.architectures.cvxlayer` (`CvxLasso` / `CvxLogistic`) *unrolls* a
fixed number of proximal / Newton steps as a differentiable block. `omnibias-convex`
is the complementary tool: it solves the program to optimality, differentiates
through the **exact** KKT conditions (constant memory, no unroll depth to tune),
and can certify the result. Reach for `cvxlayer` when you want a shallow,
fully-unrolled inner loop; reach for `omnibias-convex` when you want a true
`argmin` layer with a certificate.
