# omnibias-convex

**Status: Alpha (0.1.0a1).**

NN-native convex optimization for omnibias: solve linear and quadratic programs
*inside* the autodiff graph. Three things set it apart from a classical LP/QP
solver:

- **Differentiable `argmin`.** The optimum is a differentiable function of
  `(Q, c, A, b)` -- either through the exact KKT system (OptNet / cvxpylayers
  style) or as a plain gradient-descent problem via the temperature-collapse penalty --
  so an LP/QP becomes a first-class *layer* inside a network.
- **Batched and GPU-native.** The gradient-descent solver is pure matmul plus a
  closed-form temperature-collapse sigmoid gradient: no pivoting, no factorization, no
  data-dependent branching, so thousands of programs solve in one batched call on
  the same device as the rest of the model.
- **Certifiable.** An optional rigorous interval enclosure of the optimum /
  duality gap from `omnibias.core.verified`.

Two solvers behind one API: a closed-form-Hessian **Newton interior point**
(`solve_lp` / `solve_qp`) for last-digit accuracy on a single program, and a
first-order **temperature-collapse / hard-hinge gradient-descent** solver (`solve_lp_penalty` /
`solve_qp_penalty`) for differentiable, batched, moderate-accuracy solving and
warm starts. Need `1e-12` or an exact vertex for one program? Use the interior
point (or hand its almost-exact iterate to a simplex crossover). Want an
optimization *layer* that trains end-to-end and batches on a GPU? Use the
gradient-descent solver. They are complementary, not competitors.

## Problem form

\[
\min_x \; \tfrac12 x^\top Q x + c^\top x \quad\text{s.t.}\quad A x \le b
\]

with `Q` positive semidefinite (`Q = 0` recovers an LP). The log-barrier
subproblem has the **closed-form Hessian**

\[
H = t\,Q + A^\top \operatorname{diag}(1/s^2)\, A, \qquad s = b - A x > 0,
\]

solved with `jnp.linalg.solve` along a short central path -- the same
closed-form-Hessian Newton pattern as `omnibias.curvature.mse_newton_step`.

## Public API

```python
import jax.numpy as jnp
from omnibias.convex.jax import solve_lp, solve_qp

c = -jnp.array([1.0, 2.0])                               # maximise x + 2y
A = jnp.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])    # x + y <= 1, x >= 0, y >= 0
b = jnp.array([1.0, 0.0, 0.0])
Q = jnp.eye(2)

sol = solve_lp(c, A, b)          # min c^T x s.t. A x <= b  -> the vertex (0, 1)
sol = solve_qp(Q, c, A, b)       # + 1/2 x^T Q x
sol.x, sol.dual, sol.slack, sol.gap, sol.converged
```

Backends: `omnibias.convex.jax` and `omnibias.convex.torch`.

### Gradient-descent (temperature-collapse) solver

`solve_lp_penalty` / `solve_qp_penalty` turn the program into a plain
gradient-descent problem: each constraint hyperplane becomes a tempered
temperature-collapse (softplus) unit, so the objective is smooth with the **closed-form**
gradient `c + Q x + mu A^T sigma(beta (A x - b))`, minimised by accelerated GD along
a `beta` / `mu` homotopy (`PenaltyOptions`). It is exterior (no feasible start /
phase-1 needed), fully differentiable, and its dual estimate
`lambda = mu sigma(beta u) >= 0` feeds `certify_qp_optimum`. First-order *tolerance*
scope (~`1e-3`–`1e-4`), the complement of the Newton `solve_lp` / `solve_qp`.

```python
from omnibias.convex.jax import solve_lp_penalty
sol = solve_lp_penalty(c, A, b)   # -> the LP vertex, by gradient descent
```

## Tests

```bash
pip install -e "packages/omnibias-convex[jax,torch,test]"
python -m pytest packages/omnibias-convex/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
