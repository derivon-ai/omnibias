# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-form-Hessian log-barrier interior-point LP/QP solver (torch).

Bit-identical algorithm to :mod:`omnibias.convex.jax.solver`: a damped-Newton
central-path method with the closed-form barrier Hessian
``H = t Q + A^T diag(1/s^2) A`` solved by :func:`torch.linalg.solve`. Solves in
float64 for interior-point conditioning.
"""

from __future__ import annotations

import torch
from omnibias.convex.problem import BarrierOptions, ConvexSolution, validate_shapes
from torch import Tensor


class InfeasibleProblemError(ValueError):
    """Raised when no strictly feasible point ``A x < b`` can be found."""


def _objective(Q: Tensor, c: Tensor, x: Tensor) -> Tensor:
    return 0.5 * x @ (Q @ x) + c @ x


def _phi(Q: Tensor, c: Tensor, A: Tensor, b: Tensor, x: Tensor, t: float) -> float:
    s = b - A @ x
    if bool((s <= 0.0).any()):
        return float("inf")
    return float(t * _objective(Q, c, x) - torch.sum(torch.log(s)))


def _line_search(
    Q: Tensor, c: Tensor, A: Tensor, b: Tensor, x: Tensor, dx: Tensor,
    t: float, grad_dot_dx: float, opts: BarrierOptions,
) -> float:
    step = 1.0
    phi0 = _phi(Q, c, A, b, x, t)
    for _ in range(60):
        if _phi(Q, c, A, b, x + step * dx, t) <= phi0 + opts.backtrack_alpha * step * grad_dot_dx:
            return step
        step *= opts.backtrack_beta
    return 0.0  # no Armijo-acceptable feasible step found (barrier past its conditioning limit)


def _center(
    Q: Tensor, c: Tensor, A: Tensor, b: Tensor, x: Tensor, t: float, opts: BarrierOptions
) -> tuple[Tensor, int, bool]:
    """Damped-Newton centering of ``phi_t``; returns ``(x, steps, ok)``.

    ``ok`` is ``False`` if the barrier Hessian broke down at this ``t`` (singular
    or a non-finite Newton step -- the float64 conditioning limit), in which case
    ``x`` is returned unchanged so the caller can keep the last centered iterate.
    """
    n = x.shape[0]
    eye = torch.eye(n, dtype=x.dtype)
    steps = 0
    for _ in range(opts.newton_iters):
        s = b - A @ x
        dinv = 1.0 / s
        grad = t * (Q @ x + c) + A.T @ dinv
        hess = t * Q + A.T @ (dinv[:, None] ** 2 * A) + opts.damping * eye
        try:
            dx = torch.linalg.solve(hess, -grad)
        except RuntimeError:
            # torch raises LinAlgError (a RuntimeError subclass) when the barrier
            # Hessian is numerically singular; JAX instead returns a non-finite dx
            # (caught below). Either way the centering has broken down at this t.
            return x, steps, False
        if not bool(torch.isfinite(dx).all()):
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


def _central_path_solve(
    Q: Tensor, c: Tensor, A: Tensor, b: Tensor, x0: Tensor, opts: BarrierOptions
) -> tuple[Tensor, Tensor, float, int, float, bool, int]:
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


def _strictly_feasible_start(A: Tensor, b: Tensor, opts: BarrierOptions) -> tuple[Tensor, int]:
    m, n = A.shape
    if bool((b > 0.0).all()):
        return torch.zeros((n,), dtype=b.dtype), 0

    a_tau = -torch.ones((m, 1), dtype=b.dtype)
    rows = torch.cat([A, a_tau], dim=1)
    cap = torch.cat(
        [torch.zeros((1, n), dtype=b.dtype), -torch.ones((1, 1), dtype=b.dtype)], dim=1
    )
    a1 = torch.cat([rows, cap], dim=0)
    b1 = torch.cat([b, torch.ones((1,), dtype=b.dtype)], dim=0)
    c1 = torch.cat([torch.zeros((n,), dtype=b.dtype), torch.ones((1,), dtype=b.dtype)])
    q1 = torch.zeros((n + 1, n + 1), dtype=b.dtype)

    tau0 = float(-torch.min(b)) + 1.0
    z0 = torch.cat([torch.zeros((n,), dtype=b.dtype), torch.tensor([tau0], dtype=b.dtype)])
    z, _, _, _, _, _, phase1_newton = _central_path_solve(q1, c1, a1, b1, z0, opts)
    tau_star = float(z[n])
    if tau_star >= -1e-9:
        raise InfeasibleProblemError(
            "no strictly feasible point A x < b found "
            f"(phase-1 min infeasibility tau* = {tau_star:.3e} >= 0)"
        )
    return z[:n], phase1_newton


def _as64(x: object) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def solve_qp(
    Q: object,
    c: object,
    A: object,
    b: object,
    *,
    x0: object | None = None,
    options: BarrierOptions | None = None,
) -> ConvexSolution[Tensor]:
    r"""Solve ``min 1/2 x^T Q x + c^T x  s.t.  A x <= b`` (``Q`` PSD), in float64."""
    opts = options or BarrierOptions()
    Qt = _as64(Q)
    ct = _as64(c)
    At = _as64(A)
    bt = _as64(b)
    n = ct.shape[0]
    m = bt.shape[0]
    validate_shapes(
        n, m, A_shape=tuple(At.shape), b_shape=tuple(bt.shape),
        c_shape=tuple(ct.shape), Q_shape=tuple(Qt.shape),
    )

    if x0 is not None:
        start = _as64(x0)
        phase1_newton = 0
    else:
        start, phase1_newton = _strictly_feasible_start(At, bt, opts)
    if bool((bt - At @ start <= 0.0).any()):
        raise InfeasibleProblemError("provided x0 is not strictly feasible (A x0 < b violated)")

    x, slack, t, iterations, gap, converged, newton_total = _central_path_solve(
        Qt, ct, At, bt, start, opts
    )
    dual = 1.0 / (t * slack)
    return ConvexSolution(
        x=x,
        dual=dual,
        slack=slack,
        obj=_objective(Qt, ct, x),
        gap=gap,
        iterations=iterations,
        converged=converged,
        newton_iterations=newton_total + phase1_newton,
    )


def solve_lp(
    c: object,
    A: object,
    b: object,
    *,
    x0: object | None = None,
    options: BarrierOptions | None = None,
) -> ConvexSolution[Tensor]:
    r"""Solve the LP ``min c^T x  s.t.  A x <= b`` (``Q = 0``)."""
    ct = _as64(c)
    n = ct.shape[0]
    Q = torch.zeros((n, n), dtype=torch.float64)
    return solve_qp(Q, ct, A, b, x0=x0, options=options)


__all__ = ["InfeasibleProblemError", "solve_lp", "solve_qp"]
