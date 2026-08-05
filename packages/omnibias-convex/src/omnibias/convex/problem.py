# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-agnostic problem / solution containers for omnibias-convex.

The numeric solvers live in :mod:`omnibias.convex.jax` and
:mod:`omnibias.convex.torch`; these containers only hold the data so the two
backends present an identical surface. Arrays are stored untyped (``Any``) so the
same :class:`ConvexSolution` works for ``jax.Array`` and ``torch.Tensor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class ConvexSolution(Generic[ArrayT]):
    """Result of an LP / QP solve.

    Attributes
    ----------
    x:
        Primal optimum, shape ``(n,)``.
    dual:
        Dual multipliers ``lambda >= 0`` for ``A x <= b``, shape ``(m,)``
        (``lambda = 1 / (t * s)`` on the central path).
    slack:
        Constraint slacks ``s = b - A x >= 0``, shape ``(m,)``.
    obj:
        Objective value ``1/2 x^T Q x + c^T x`` at ``x``.
    gap:
        Surrogate duality gap ``m / t`` of the final central-path point.
    iterations:
        Number of outer (barrier) iterations taken.
    converged:
        Whether ``gap <= tol`` was reached within the iteration budget.
    newton_iterations:
        Total inner (centering) Newton steps across all outer iterations -- the
        main cost a warm start reduces.
    eq_dual:
        Dual multipliers ``nu`` for the optional equality constraints
        ``A_eq x = b_eq``, shape ``(m_eq,)`` (the quadratic-penalty estimate
        ``nu = mu (A_eq x - b_eq)``); ``None`` when no equalities were supplied.
        Unlike ``dual`` it is sign-free.
    """

    x: ArrayT
    dual: ArrayT
    slack: ArrayT
    obj: ArrayT
    gap: float
    iterations: int
    converged: bool
    newton_iterations: int = 0
    eq_dual: ArrayT | None = None


@dataclass(frozen=True)
class BarrierOptions:
    """Tuning knobs for the log-barrier interior-point solver.

    Defaults follow the textbook path-following method (Boyd & Vandenberghe,
    *Convex Optimization*, ch. 11): increase the barrier weight ``t`` by ``mu``
    each outer step, re-centre with a damped Newton phase, and stop when the
    surrogate gap ``m / t`` drops below ``tol``.
    """

    tol: float = 1e-9
    mu: float = 10.0
    t0: float = 1.0
    max_outer: int = 60
    newton_iters: int = 50
    newton_tol: float = 1e-12
    backtrack_alpha: float = 0.25
    backtrack_beta: float = 0.5
    damping: float = 1e-12


@dataclass(frozen=True)
class PenaltyOptions:
    r"""Tuning knobs for the tempered temperature-collapse (gradient-descent) solver.

    The first-order solver minimises the smooth *exterior* penalty

    .. math::
        F(x) = c^\top x + \tfrac12 x^\top Q x
               + \mu \sum_i \tfrac1\beta \operatorname{softplus}(\beta (a_i^\top x - b_i))

    by accelerated (Nesterov) gradient descent with a closed-form Lipschitz step
    ``eta = step_safety / L``, ``L = ||Q||_2 + (mu * beta / 4) ||A||_2^2`` (the
    penalty Hessian is ``mu beta A^T diag(sigma') A`` with ``sigma' <= 1/4``), over
    a homotopy that grows the sharpness ``beta`` and the penalty weight ``mu``
    geometrically.     ``beta -> inf`` is the *temperature-collapse limit*: each softplus edge
    collapses onto its exact constraint hyperplane ``a_i^T x = b_i`` (the same
    ``beta -> inf`` homotopy as :mod:`omnibias.binary`'s tanh surrogate).
    (Terminology: this is the ``beta -> inf`` **feasibility** sense of
    "collapse" -- a 0/1 step; *not* the **founding bias collapse**, the
    multi-bias ``delta -> 0`` limit ``sum_k s_k sigma(z + b_k) -> sigma^(K-1)``
    that yields a derivative. See ``docs/theory.md`` and
    :mod:`omnibias.torch.unit`.)

    Attributes
    ----------
    beta0, beta_growth:
        Initial sharpness and geometric growth factor per homotopy stage.
    penalty0, penalty_growth:
        Initial penalty weight ``mu`` and its geometric growth factor.
    stages:
        Number of homotopy stages (``beta``/``mu`` increases).
    gd_steps:
        Accelerated-gradient steps per stage (each stage warm-starts the next).
    step_safety:
        Fraction of the Lipschitz step ``1 / L`` to take (``0 < step_safety <= 1``).
    prox:
        Strength of the proximal-point term ``(prox / 2) ||x - x_prev||^2`` that
        anchors each stage to the previous iterate. It makes every subproblem
        coercive (so the exterior penalty cannot run off to infinity while ``mu``
        is below the largest optimal multiplier) and vanishes at a fixed point, so
        the recovered optimum is unbiased. Must be ``> 0`` for an LP (``Q = 0``).
    tol:
        Path-settled tolerance ``||x_k - x_{k-1}||_inf``; the homotopy stops early
        once the path stops moving (and the point is feasible).
    feas_tol:
        Maximum constraint violation ``max_i (a_i^T x - b_i)_+`` accepted for the
        ``converged`` flag.
    """

    beta0: float = 1.0
    beta_growth: float = 3.0
    penalty0: float = 1.0
    penalty_growth: float = 1.6
    stages: int = 11
    gd_steps: int = 3000
    step_safety: float = 0.9
    prox: float = 1.0
    tol: float = 1e-4
    feas_tol: float = 1e-4

    def __post_init__(self) -> None:
        if self.beta0 <= 0.0 or self.penalty0 <= 0.0:
            raise ValueError("beta0 and penalty0 must be > 0")
        if self.beta_growth < 1.0 or self.penalty_growth < 1.0:
            raise ValueError("beta_growth and penalty_growth must be >= 1")
        if self.stages < 1 or self.gd_steps < 1:
            raise ValueError("stages and gd_steps must be >= 1")
        if not 0.0 < self.step_safety <= 1.0:
            raise ValueError("step_safety must be in (0, 1]")
        if self.prox < 0.0:
            raise ValueError("prox must be >= 0")


def validate_shapes(
    n: int,
    m: int,
    *,
    A_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    c_shape: tuple[int, ...],
    Q_shape: tuple[int, ...] | None = None,
    A_eq_shape: tuple[int, ...] | None = None,
    b_eq_shape: tuple[int, ...] | None = None,
) -> None:
    """Raise ``ValueError`` if the LP/QP data shapes are inconsistent.

    ``A_eq_shape`` / ``b_eq_shape`` are optional equality-constraint shapes; when
    either is given both must be, with ``A_eq`` of shape ``(m_eq, n)`` and ``b_eq``
    of shape ``(m_eq,)`` (``m_eq`` inferred from ``b_eq``).
    """
    if A_shape != (m, n):
        raise ValueError(f"A must have shape (m, n) = ({m}, {n}), got {A_shape}")
    if b_shape != (m,):
        raise ValueError(f"b must have shape (m,) = ({m},), got {b_shape}")
    if c_shape != (n,):
        raise ValueError(f"c must have shape (n,) = ({n},), got {c_shape}")
    if Q_shape is not None and Q_shape != (n, n):
        raise ValueError(f"Q must have shape (n, n) = ({n}, {n}), got {Q_shape}")
    if A_eq_shape is not None or b_eq_shape is not None:
        if A_eq_shape is None or b_eq_shape is None:
            raise ValueError("A_eq and b_eq must be supplied together")
        if len(b_eq_shape) != 1:
            raise ValueError(f"b_eq must have shape (m_eq,), got {b_eq_shape}")
        m_eq = b_eq_shape[0]
        if A_eq_shape != (m_eq, n):
            raise ValueError(
                f"A_eq must have shape (m_eq, n) = ({m_eq}, {n}), got {A_eq_shape}"
            )


__all__ = ["BarrierOptions", "ConvexSolution", "PenaltyOptions", "validate_shapes"]
