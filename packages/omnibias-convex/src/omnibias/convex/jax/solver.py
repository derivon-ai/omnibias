# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-form-Hessian log-barrier interior-point LP/QP solver (JAX).

Solves

.. math::
    \min_x \tfrac12 x^\top Q x + c^\top x \quad\text{s.t.}\quad A x \le b

with ``Q`` positive semidefinite (``Q = 0`` => LP). The log-barrier subproblem

.. math::
    \varphi_t(x) = t\,\big(\tfrac12 x^\top Q x + c^\top x\big) - \sum_i \log s_i,
    \qquad s = b - A x > 0

is minimised by a damped Newton method whose gradient and **closed-form Hessian**

.. math::
    \nabla\varphi_t = t (Q x + c) + A^\top (1/s), \qquad
    \nabla^2\varphi_t = t\,Q + A^\top \operatorname{diag}(1/s^2) A

are assembled directly and solved with :func:`jax.numpy.linalg.solve` -- the same
closed-form-Hessian Newton pattern as ``omnibias.curvature.mse_newton_step``. The
barrier weight ``t`` is increased along a short central path; the surrogate
duality gap is ``m / t`` and the dual multipliers are ``lambda = 1/(t s)``.

A strictly feasible start is found from ``x = 0`` when ``b > 0``; otherwise a
phase-1 ``min tau s.t. A x - tau <= b`` LP (the same solver) supplies one.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.convex.problem import BarrierOptions, ConvexSolution, validate_shapes


class InfeasibleProblemError(ValueError):
    """Raised when no strictly feasible point ``A x < b`` can be found."""


def _objective(Q: Array, c: Array, x: Array) -> Array:
    return 0.5 * x @ (Q @ x) + c @ x


def _center(
    Q: Array, c: Array, A: Array, b: Array, x: Array, t: float, opts: BarrierOptions
) -> tuple[Array, int, bool]:
    """Damped-Newton centering of ``phi_t``; returns ``(x, steps, ok)``.

    ``ok`` is ``False`` if the barrier Hessian broke down at this ``t`` (a
    non-finite Newton step -- the float64 conditioning limit), in which case ``x``
    is returned unchanged so the caller can keep the last centered iterate.
    """
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    steps = 0
    for _ in range(opts.newton_iters):
        s = b - A @ x
        dinv = 1.0 / s
        grad = t * (Q @ x + c) + A.T @ dinv
        hess = t * Q + A.T @ (dinv[:, None] ** 2 * A) + opts.damping * eye
        dx = jnp.linalg.solve(hess, -grad)
        if not bool(jnp.isfinite(dx).all()):
            return x, steps, False
        decrement_sq = float(-(grad @ dx))
        if decrement_sq <= opts.newton_tol:
            break
        step = _line_search(Q, c, A, b, x, dx, t, float(grad @ dx), opts)
        if step <= 0.0:
            # No Armijo-acceptable feasible step exists at this t: the barrier Hessian
            # is past its float64 conditioning limit. Keep the last centered iterate
            # rather than a spurious tiny step into a divergent / infeasible x.
            return x, steps, False
        x = x + step * dx
        steps += 1
    return x, steps, True


def _phi(Q: Array, c: Array, A: Array, b: Array, x: Array, t: float) -> float:
    s = b - A @ x
    if bool(jnp.any(s <= 0.0)):
        return float("inf")
    return float(t * _objective(Q, c, x) - jnp.sum(jnp.log(s)))


def _line_search(
    Q: Array, c: Array, A: Array, b: Array, x: Array, dx: Array,
    t: float, grad_dot_dx: float, opts: BarrierOptions,
) -> float:
    """Backtracking line search: keep ``s > 0`` then satisfy Armijo on ``phi_t``."""
    step = 1.0
    phi0 = _phi(Q, c, A, b, x, t)
    for _ in range(60):
        if _phi(Q, c, A, b, x + step * dx, t) <= phi0 + opts.backtrack_alpha * step * grad_dot_dx:
            return step
        step *= opts.backtrack_beta
    return 0.0  # no Armijo-acceptable feasible step found (barrier past its conditioning limit)


def _central_path_solve(
    Q: Array, c: Array, A: Array, b: Array, x0: Array, opts: BarrierOptions
) -> tuple[Array, Array, float, int, float, bool, int]:
    """Path-following loop.

    Returns ``(x, slack, t, iterations, gap, converged, newton_iterations)``.
    """
    m = b.shape[0]
    x = x0
    t = opts.t0
    t_used = t
    gap = float(m) / t
    converged = gap <= opts.tol
    iterations = 0
    newton_total = 0
    for outer in range(opts.max_outer):
        x_next, steps, ok = _center(Q, c, A, b, x, t, opts)
        if not ok:
            # Centering broke down at this t (the barrier Hessian A^T diag(1/s^2) A
            # has condition ~ 1/s_min^2 ~ (t/m)^2, past the float64 limit). Keep the
            # last successfully centered iterate -- at t_used, so gap/dual stay
            # consistent -- instead of returning a divergent NaN as "converged".
            break
        x = x_next
        t_used = t
        newton_total += steps
        iterations = outer + 1
        gap = float(m) / t
        if gap <= opts.tol:
            converged = True
            break
        t *= opts.mu
    slack = b - A @ x
    return x, slack, t_used, iterations, gap, converged, newton_total


def _strictly_feasible_start(A: Array, b: Array, opts: BarrierOptions) -> tuple[Array, int]:
    """Return ``(x, phase1_newton)``: a strictly feasible ``x`` (``A x < b``)."""
    m, n = A.shape
    if bool(jnp.all(b > 0.0)):
        return jnp.zeros((n,), dtype=b.dtype), 0

    # Phase-1: min tau  s.t.  A x - tau <= b,  -tau <= 1   (keeps tau bounded).
    a_tau = -jnp.ones((m, 1), dtype=b.dtype)
    rows = jnp.concatenate([A, a_tau], axis=1)  # (m, n+1)
    cap = jnp.concatenate(
        [jnp.zeros((1, n), dtype=b.dtype), -jnp.ones((1, 1), dtype=b.dtype)], axis=1
    )
    a1 = jnp.concatenate([rows, cap], axis=0)  # (m+1, n+1)
    b1 = jnp.concatenate([b, jnp.ones((1,), dtype=b.dtype)], axis=0)
    c1 = jnp.concatenate([jnp.zeros((n,), dtype=b.dtype), jnp.ones((1,), dtype=b.dtype)])
    q1 = jnp.zeros((n + 1, n + 1), dtype=b.dtype)

    tau0 = float(-jnp.min(b)) + 1.0
    z0 = jnp.concatenate([jnp.zeros((n,), dtype=b.dtype), jnp.asarray([tau0], dtype=b.dtype)])
    z, _, _, _, _, _, phase1_newton = _central_path_solve(q1, c1, a1, b1, z0, opts)
    tau_star = float(z[n])
    if tau_star >= -1e-9:
        raise InfeasibleProblemError(
            "no strictly feasible point A x < b found "
            f"(phase-1 min infeasibility tau* = {tau_star:.3e} >= 0)"
        )
    return z[:n], phase1_newton


def solve_qp(
    Q: Array,
    c: Array,
    A: Array,
    b: Array,
    *,
    x0: Array | None = None,
    options: BarrierOptions | None = None,
) -> ConvexSolution[Array]:
    r"""Solve ``min 1/2 x^T Q x + c^T x  s.t.  A x <= b`` (``Q`` PSD).

    Parameters
    ----------
    Q, c, A, b:
        Problem data with shapes ``(n, n)``, ``(n,)``, ``(m, n)``, ``(m,)``.
    x0:
        Optional strictly feasible warm start (``A x0 < b``). If omitted, a start
        is found from ``x = 0`` (when ``b > 0``) or a phase-1 LP.
    options:
        :class:`~omnibias.convex.problem.BarrierOptions` tuning.

    Returns
    -------
    :class:`~omnibias.convex.problem.ConvexSolution` with primal ``x``, dual
    ``lambda = 1/(t s) >= 0``, slacks ``s``, objective, gap ``m/t`` and flags.
    """
    opts = options or BarrierOptions()
    Q = jnp.asarray(Q)
    c = jnp.asarray(c)
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    n = c.shape[0]
    m = b.shape[0]
    validate_shapes(n, m, A_shape=A.shape, b_shape=b.shape, c_shape=c.shape, Q_shape=Q.shape)

    if x0 is not None:
        start = jnp.asarray(x0, dtype=b.dtype)
        phase1_newton = 0
    else:
        start, phase1_newton = _strictly_feasible_start(A, b, opts)
    if bool(jnp.any(b - A @ start <= 0.0)):
        raise InfeasibleProblemError("provided x0 is not strictly feasible (A x0 < b violated)")

    x, slack, t, iterations, gap, converged, newton_total = _central_path_solve(
        Q, c, A, b, start, opts
    )
    dual = 1.0 / (t * slack)
    return ConvexSolution(
        x=x,
        dual=dual,
        slack=slack,
        obj=_objective(Q, c, x),
        gap=gap,
        iterations=iterations,
        converged=converged,
        newton_iterations=newton_total + phase1_newton,
    )


def solve_lp(
    c: Array,
    A: Array,
    b: Array,
    *,
    x0: Array | None = None,
    options: BarrierOptions | None = None,
) -> ConvexSolution[Array]:
    r"""Solve the LP ``min c^T x  s.t.  A x <= b`` (``Q = 0`` special case)."""
    c = jnp.asarray(c)
    n = c.shape[0]
    Q = jnp.zeros((n, n), dtype=c.dtype)
    return solve_qp(Q, c, A, b, x0=x0, options=options)


__all__ = ["InfeasibleProblemError", "solve_lp", "solve_qp"]
