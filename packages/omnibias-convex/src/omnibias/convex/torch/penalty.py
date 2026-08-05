# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Tempered temperature-collapse penalty LP/QP solver by gradient descent (torch).

Bit-identical twin of :mod:`omnibias.convex.jax.penalty` (float64). Turns a linear
(or convex-quadratic) program into a plain **gradient-descent** problem: each
constraint ``a_i^T x <= b_i`` is a hyperplane, the *temperature-collapse* unit
``sigma(beta (a_i^T x - b_i))`` (the omnibias tempered heaviside) has that
hyperplane as its decision boundary, and its integral is the smooth hinge penalty
``softplus(beta u) / beta -> max(u, 0)`` as ``beta -> inf``. Minimising

.. math::
    F(x) = c^\top x + \tfrac12 x^\top Q x
           + \mu \sum_i \tfrac1\beta \operatorname{softplus}(\beta (a_i^\top x - b_i))

has the **closed-form** gradient ``c + Q x + mu A^T sigma(beta (A x - b))``.
The gradient calls ``torch.sigmoid`` directly rather than the omnibias fastpath:
at order 1, ``softplus' = sigmoid`` *is* the tower's value, so the two are the
same expression and the framework primitive is the cheaper spelling. Any
higher-order term would have to go through :mod:`omnibias.torch`.
:func:`penalty_descent` runs accelerated (Nesterov) gradient descent with the
closed-form Lipschitz step ``eta = step_safety / L``; :func:`solve_qp_penalty` /
:func:`solve_lp_penalty` run it along a ``beta`` / ``mu`` homotopy and return a
:class:`~omnibias.convex.problem.ConvexSolution` with the dual estimate
``lambda = mu sigma(beta (A x - b)) >= 0``. See the JAX module for the full math
and honest-scope notes.

Terminology: "temperature-collapse" here is the ``beta -> inf`` **feasibility** sense
(a 0/1 step), distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to ``sigma^(K-1)``, a derivative; see
:mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

import math

import torch
from omnibias.convex.problem import ConvexSolution, PenaltyOptions, validate_shapes
from torch import Tensor


def _as64(x: object) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def penalty_gradient(
    Q: Tensor,
    c: Tensor,
    A: Tensor,
    b: Tensor,
    x: Tensor,
    *,
    beta: float,
    mu: float,
    A_eq: Tensor | None = None,
    b_eq: Tensor | None = None,
) -> Tensor:
    r"""Closed-form gradient of ``F`` (inequality + optional equality penalty).

    .. math::
        \nabla F = c + Q x + \mu A^\top \sigma(\beta (A x - b))
                   + \mu A_{eq}^\top (A_{eq} x - b_{eq}).

    Each term ``sigma(beta (a_i^T x - b_i))`` is the temperature-collapse unit of
    inequality ``i``; weighted by ``mu`` and summed through ``A^T`` it is the
    exterior-penalty force pushing ``x`` toward the feasible polytope. The optional
    equality block adds the **quadratic** exterior penalty ``(mu/2)||A_eq x - b_eq||^2``
    whose closed-form gradient is ``mu A_eq^T (A_eq x - b_eq)``.
    """
    u = A @ x - b
    activations = torch.sigmoid(beta * u)
    grad = c + Q @ x + mu * (A.t() @ activations)
    if A_eq is not None and b_eq is not None:
        grad = grad + mu * (A_eq.t() @ (A_eq @ x - b_eq))
    return grad


def _lipschitz_step(
    Q: Tensor,
    A: Tensor,
    beta: float,
    mu: float,
    prox: float,
    step_safety: float,
    A_eq: Tensor | None = None,
) -> Tensor:
    r"""Closed-form ``eta = step_safety / L`` from the penalty's Lipschitz constant.

    Uses the Frobenius bounds ``||A||_2^2 <= ||A||_F^2`` and ``||Q||_2 <= ||Q||_F``
    (elementwise, hence bit-identical to the JAX twin). The optional equality block
    adds the constant curvature ``mu ||A_eq||_F^2`` (no ``beta`` blow-up).
    """
    norm_a_sq = torch.sum(A * A)
    norm_q = torch.sqrt(torch.sum(Q * Q))
    lip = norm_q + prox + 0.25 * mu * beta * norm_a_sq + 1e-30
    if A_eq is not None:
        lip = lip + mu * torch.sum(A_eq * A_eq)
    return step_safety / lip


def penalty_descent(
    Q: Tensor,
    c: Tensor,
    A: Tensor,
    b: Tensor,
    x0: Tensor,
    *,
    beta: float,
    mu: float,
    steps: int,
    step_safety: float = 0.9,
    prox: float = 0.0,
    anchor: Tensor | None = None,
    A_eq: Tensor | None = None,
    b_eq: Tensor | None = None,
) -> Tensor:
    r"""Minimise ``F`` (plus an optional proximal term) for fixed ``(beta, mu)``.

    Accelerated (Nesterov) gradient descent with the closed-form Lipschitz step
    ``eta = step_safety / L``. The optional proximal term ``(prox / 2) ||x - anchor||^2``
    (``anchor`` defaults to ``x0``) makes the subproblem coercive/strongly convex
    and vanishes at a homotopy fixed point, so the recovered optimum is unbiased.
    Optional equality constraints ``A_eq x = b_eq`` are enforced by the quadratic
    exterior penalty ``(mu/2)||A_eq x - b_eq||^2`` (closed-form gradient, Lipschitz
    bump). Pure tensor arithmetic, so (with ``prox = 0`` on a bounded problem) it is
    autograd-friendly. Returns the iterate ``x``.
    """
    anchor_pt = x0 if anchor is None else anchor
    eta = _lipschitz_step(Q, A, beta, mu, prox, step_safety, A_eq)
    x = x0
    y = x0
    t_acc = 1.0
    for _ in range(steps):
        grad = (
            penalty_gradient(Q, c, A, b, y, beta=beta, mu=mu, A_eq=A_eq, b_eq=b_eq)
            + prox * (y - anchor_pt)
        )
        x_next = y - eta * grad
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_acc * t_acc))
        y = x_next + ((t_acc - 1.0) / t_next) * (x_next - x)
        x = x_next
        t_acc = t_next
    return x


def solve_qp_penalty(
    Q: object,
    c: object,
    A: object,
    b: object,
    *,
    A_eq: object | None = None,
    b_eq: object | None = None,
    x0: object | None = None,
    options: PenaltyOptions | None = None,
) -> ConvexSolution[Tensor]:
    r"""Solve ``min 1/2 x^T Q x + c^T x  s.t.  A x <= b [, A_eq x = b_eq]``.

    First-order homotopy (float64): accelerated gradient descent on the tempered
    penalty with ``beta`` / ``mu`` grown geometrically. ``x0`` may be **any** point
    (the exterior penalty needs no feasible start); it defaults to the origin.
    Optional equality constraints ``A_eq x = b_eq`` are enforced by a quadratic
    exterior penalty; the estimate ``nu = mu (A_eq x - b_eq)`` is returned in
    ``eq_dual`` (``None`` when no equalities are supplied).
    """
    opts = options or PenaltyOptions()
    Qt = _as64(Q)
    ct = _as64(c)
    At = _as64(A)
    bt = _as64(b)
    n = ct.shape[0]
    m = bt.shape[0]
    has_eq = A_eq is not None and b_eq is not None
    Aeq = _as64(A_eq) if has_eq else None
    beq = _as64(b_eq) if has_eq else None
    validate_shapes(
        n, m, A_shape=tuple(At.shape), b_shape=tuple(bt.shape),
        c_shape=tuple(ct.shape), Q_shape=tuple(Qt.shape),
        A_eq_shape=None if Aeq is None else tuple(Aeq.shape),
        b_eq_shape=None if beq is None else tuple(beq.shape),
    )

    x = torch.zeros((n,), dtype=torch.float64) if x0 is None else _as64(x0)
    beta = opts.beta0
    mu = opts.penalty0
    beta_used = beta
    mu_used = mu
    total_steps = 0
    stages_run = 0
    dx = float("inf")
    viol = float("inf")
    for _ in range(opts.stages):
        beta_used = beta
        mu_used = mu
        x_prev = x
        x = penalty_descent(
            Qt, ct, At, bt, x_prev,
            beta=beta, mu=mu, steps=opts.gd_steps, step_safety=opts.step_safety,
            prox=opts.prox, anchor=x_prev, A_eq=Aeq, b_eq=beq,
        )
        total_steps += opts.gd_steps
        stages_run += 1
        dx = float(torch.max(torch.abs(x - x_prev)))
        viol = float(torch.max(torch.clamp(At @ x - bt, min=0.0)))
        if has_eq:
            if Aeq is None or beq is None:
                raise ValueError(
                    "equality constraints require both A_eq and b_eq; "
                    f"got A_eq={'set' if Aeq is not None else None}, "
                    f"b_eq={'set' if beq is not None else None}"
                )
            viol = max(viol, float(torch.max(torch.abs(Aeq @ x - beq))))
        if dx <= opts.tol and viol <= opts.feas_tol:
            break
        beta *= opts.beta_growth
        mu *= opts.penalty_growth

    converged = viol <= opts.feas_tol
    u = At @ x - bt
    dual = mu_used * torch.sigmoid(beta_used * u)
    eq_dual = None
    if has_eq:
        if Aeq is None or beq is None:
            raise ValueError(
                "equality constraints require both A_eq and b_eq; "
                f"got A_eq={'set' if Aeq is not None else None}, "
                f"b_eq={'set' if beq is not None else None}"
            )
        eq_dual = mu_used * (Aeq @ x - beq)
    slack = bt - At @ x
    obj = 0.5 * x @ (Qt @ x) + ct @ x
    return ConvexSolution(
        x=x,
        dual=dual,
        slack=slack,
        obj=obj,
        gap=max(viol, dx),
        iterations=stages_run,
        converged=converged,
        newton_iterations=total_steps,
        eq_dual=eq_dual,
    )


def solve_lp_penalty(
    c: object,
    A: object,
    b: object,
    *,
    A_eq: object | None = None,
    b_eq: object | None = None,
    x0: object | None = None,
    options: PenaltyOptions | None = None,
) -> ConvexSolution[Tensor]:
    r"""Solve the LP ``min c^T x  s.t.  A x <= b [, A_eq x = b_eq]`` (``Q = 0``)."""
    ct = _as64(c)
    n = ct.shape[0]
    Q = torch.zeros((n, n), dtype=torch.float64)
    return solve_qp_penalty(Q, ct, A, b, A_eq=A_eq, b_eq=b_eq, x0=x0, options=options)


__all__ = ["penalty_descent", "penalty_gradient", "solve_lp_penalty", "solve_qp_penalty"]
