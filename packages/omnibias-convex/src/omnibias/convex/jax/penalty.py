# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Tempered temperature-collapse penalty LP/QP solver by gradient descent (JAX).

Turns a linear (or convex-quadratic) program into a plain **gradient-descent**
problem. Each constraint ``a_i^T x <= b_i`` is a hyperplane; the *temperature-collapse*
unit ``sigma(beta (a_i^T x - b_i))`` (the omnibias tempered heaviside) has that
hyperplane as its decision boundary, and its integral is the smooth hinge penalty
``softplus(beta u) / beta -> max(u, 0)`` as ``beta -> inf``. Summing the penalties
gives the smooth *exterior* objective

.. math::
    F(x) = c^\top x + \tfrac12 x^\top Q x
           + \mu \sum_i \tfrac1\beta \operatorname{softplus}(\beta (a_i^\top x - b_i)),

whose gradient is **closed form** -- a sum of temperature-collapse sigmoids, no autodiff:

.. math::
    \nabla F = c + Q x + \mu A^\top \sigma(\beta (A x - b)).

The sigmoid is spelled ``jax.nn.sigmoid`` rather than routed through the
omnibias fastpath: at order 1, ``softplus' = sigmoid`` *is* the tower's value,
so the two are the same expression and the framework primitive is the cheaper
spelling. Any higher-order term would have to go through :mod:`omnibias.jax`.

:func:`penalty_descent` minimises ``F`` for a fixed ``(beta, mu)`` by accelerated
(Nesterov) gradient descent with the closed-form Lipschitz step
``eta = step_safety / L``, ``L = ||Q||_2 + (mu beta / 4) ||A||_2^2``; it is pure
(``jit`` / ``grad`` friendly) -- the differentiable "LP as gradient descent"
primitive. :func:`solve_qp_penalty` / :func:`solve_lp_penalty` run it along a
``beta`` / ``mu`` homotopy (each stage warm-starts the next) and return a
:class:`~omnibias.convex.problem.ConvexSolution` whose dual estimate is the
temperature-collapse activation itself, ``lambda = mu sigma(beta (A x - b)) >= 0``.

Honest scope: a first-order, GPU/batch-friendly, differentiable homotopy solver
that converges to a *tolerance* -- the complement of, not a replacement for, the
Newton :func:`~omnibias.convex.jax.solver.solve_qp` (which reaches ~1e-12). The
exterior penalty needs **no** strictly feasible start and no phase-1.

Terminology: the *temperature-collapse* unit here is the ``beta -> inf`` **feasibility**
sense of "collapse" (the constraint sigmoid saturates to a 0/1 step); it is
distinct from omnibias's **founding bias collapse** -- the multi-bias
``delta -> 0`` limit ``sum_k s_k sigma(z + b_k) -> sigma^(K-1)`` that yields a
smooth *derivative* (see ``docs/theory.md`` and :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.convex.problem import ConvexSolution, PenaltyOptions, validate_shapes


def penalty_gradient(
    Q: Array,
    c: Array,
    A: Array,
    b: Array,
    x: Array,
    *,
    beta: float,
    mu: float,
    A_eq: Array | None = None,
    b_eq: Array | None = None,
) -> Array:
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
    activations = jax.nn.sigmoid(beta * u)
    grad = c + Q @ x + mu * (A.T @ activations)
    if A_eq is not None and b_eq is not None:
        grad = grad + mu * (A_eq.T @ (A_eq @ x - b_eq))
    return grad


def _lipschitz_step(
    Q: Array,
    A: Array,
    beta: float,
    mu: float,
    prox: float,
    step_safety: float,
    A_eq: Array | None = None,
) -> Array:
    r"""Closed-form ``eta = step_safety / L`` from the penalty's Lipschitz constant.

    Uses the Frobenius bounds ``||A||_2^2 <= ||A||_F^2`` and ``||Q||_2 <= ||Q||_F``
    (elementwise, hence bit-identical across backends) so ``L`` is a rigorous upper
    bound on the true gradient-Lipschitz constant ``||Q||_2 + prox +
    (mu beta / 4) ||A||_2^2 + mu ||A_eq||_2^2``. The equality block contributes the
    constant curvature ``mu A_eq^T A_eq`` (no ``beta`` blow-up).
    """
    norm_a_sq = jnp.sum(A * A)
    norm_q = jnp.sqrt(jnp.sum(Q * Q))
    lip = norm_q + prox + 0.25 * mu * beta * norm_a_sq + 1e-30
    if A_eq is not None:
        lip = lip + mu * jnp.sum(A_eq * A_eq)
    return step_safety / lip


def penalty_descent(
    Q: Array,
    c: Array,
    A: Array,
    b: Array,
    x0: Array,
    *,
    beta: float,
    mu: float,
    steps: int,
    step_safety: float = 0.9,
    prox: float = 0.0,
    anchor: Array | None = None,
    A_eq: Array | None = None,
    b_eq: Array | None = None,
) -> Array:
    r"""Minimise ``F`` (plus an optional proximal term) for fixed ``(beta, mu)``.

    Accelerated (Nesterov) gradient descent with the closed-form Lipschitz step
    ``eta = step_safety / L``. The optional proximal term ``(prox / 2) ||x - anchor||^2``
    (``anchor`` defaults to ``x0``) makes the subproblem coercive/strongly convex --
    it is what stops the exterior penalty running off to infinity while ``mu`` is
    below the largest optimal multiplier, and it vanishes at a homotopy fixed point
    (``x == anchor``) so the recovered optimum is unbiased. Optional equality
    constraints ``A_eq x = b_eq`` are enforced by the quadratic exterior penalty
    ``(mu/2)||A_eq x - b_eq||^2`` (closed-form gradient, Lipschitz bump).

    Pure (no data-dependent control flow), so with ``prox = 0`` on a bounded
    (strongly convex) problem it is ``jax.jit`` / ``jax.grad`` friendly -- the
    differentiable "LP as gradient descent" primitive. Returns the iterate ``x``.
    """
    anchor_pt = x0 if anchor is None else anchor
    eta = _lipschitz_step(Q, A, beta, mu, prox, step_safety, A_eq)

    def body(_: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, y, t_acc = carry
        grad = (
            penalty_gradient(Q, c, A, b, y, beta=beta, mu=mu, A_eq=A_eq, b_eq=b_eq)
            + prox * (y - anchor_pt)
        )
        x_next = y - eta * grad
        t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_acc * t_acc))
        y_next = x_next + ((t_acc - 1.0) / t_next) * (x_next - x)
        return x_next, y_next, t_next

    one = jnp.asarray(1.0, dtype=x0.dtype)
    x_final, _, _ = jax.lax.fori_loop(0, steps, body, (x0, x0, one))
    return cast(Array, x_final)


def solve_qp_penalty(
    Q: Array,
    c: Array,
    A: Array,
    b: Array,
    *,
    A_eq: Array | None = None,
    b_eq: Array | None = None,
    x0: Array | None = None,
    options: PenaltyOptions | None = None,
) -> ConvexSolution[Array]:
    r"""Solve ``min 1/2 x^T Q x + c^T x  s.t.  A x <= b [, A_eq x = b_eq]``.

    First-order homotopy: accelerated gradient descent on the tempered penalty
    with ``beta`` / ``mu`` grown geometrically. ``x0`` may be **any** point (the
    exterior penalty needs no feasible start); it defaults to the origin. Optional
    equality constraints ``A_eq x = b_eq`` are enforced by a quadratic exterior
    penalty. Returns a :class:`~omnibias.convex.problem.ConvexSolution` with the
    inequality dual ``lambda = mu sigma(beta (A x - b)) >= 0``, the equality dual
    estimate ``nu = mu (A_eq x - b_eq)`` in ``eq_dual`` (``None`` if no equalities),
    ``iterations`` = homotopy stages run and ``newton_iterations`` = total steps.
    """
    opts = options or PenaltyOptions()
    Qa = jnp.asarray(Q)
    ca = jnp.asarray(c)
    Aa = jnp.asarray(A)
    ba = jnp.asarray(b)
    n = ca.shape[0]
    m = ba.shape[0]
    has_eq = A_eq is not None and b_eq is not None
    Aeq = jnp.asarray(A_eq, dtype=ba.dtype) if has_eq else None
    beq = jnp.asarray(b_eq, dtype=ba.dtype) if has_eq else None
    validate_shapes(
        n, m, A_shape=Aa.shape, b_shape=ba.shape, c_shape=ca.shape, Q_shape=Qa.shape,
        A_eq_shape=None if Aeq is None else Aeq.shape,
        b_eq_shape=None if beq is None else beq.shape,
    )

    x = jnp.zeros((n,), dtype=ba.dtype) if x0 is None else jnp.asarray(x0, dtype=ba.dtype)
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
            Qa, ca, Aa, ba, x_prev,
            beta=beta, mu=mu, steps=opts.gd_steps, step_safety=opts.step_safety,
            prox=opts.prox, anchor=x_prev, A_eq=Aeq, b_eq=beq,
        )
        total_steps += opts.gd_steps
        stages_run += 1
        dx = float(jnp.max(jnp.abs(x - x_prev)))
        viol = float(jnp.max(jnp.maximum(Aa @ x - ba, 0.0)))
        if has_eq:
            if Aeq is None or beq is None:
                raise ValueError(
                    "equality constraints require both A_eq and b_eq; "
                    f"got A_eq={'set' if Aeq is not None else None}, "
                    f"b_eq={'set' if beq is not None else None}"
                )
            viol = max(viol, float(jnp.max(jnp.abs(Aeq @ x - beq))))
        if dx <= opts.tol and viol <= opts.feas_tol:
            break
        beta *= opts.beta_growth
        mu *= opts.penalty_growth

    converged = viol <= opts.feas_tol
    u = Aa @ x - ba
    dual = mu_used * jax.nn.sigmoid(beta_used * u)
    eq_dual = None
    if has_eq:
        if Aeq is None or beq is None:
            raise ValueError(
                "equality constraints require both A_eq and b_eq; "
                f"got A_eq={'set' if Aeq is not None else None}, "
                f"b_eq={'set' if beq is not None else None}"
            )
        eq_dual = mu_used * (Aeq @ x - beq)
    slack = ba - Aa @ x
    obj = 0.5 * x @ (Qa @ x) + ca @ x
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
    c: Array,
    A: Array,
    b: Array,
    *,
    A_eq: Array | None = None,
    b_eq: Array | None = None,
    x0: Array | None = None,
    options: PenaltyOptions | None = None,
) -> ConvexSolution[Array]:
    r"""Solve the LP ``min c^T x  s.t.  A x <= b [, A_eq x = b_eq]`` (``Q = 0``)."""
    ca = jnp.asarray(c)
    n = ca.shape[0]
    Q = jnp.zeros((n, n), dtype=ca.dtype)
    return solve_qp_penalty(Q, ca, A, b, A_eq=A_eq, b_eq=b_eq, x0=x0, options=options)


__all__ = ["penalty_descent", "penalty_gradient", "solve_lp_penalty", "solve_qp_penalty"]
