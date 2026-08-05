# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Tight LP-dual optimality-gap certificate for a decoded combinatorial solution.

:func:`certify_gap` sandwiches the true minimum objective between

* a **rigorous lower bound** -- the polytope's LP dual, solved exactly (scipy HiGHS) for
  sign-consistent multipliers and then **sealed** by the Neumaier-Shcherbina verified
  bound (:func:`omnibias.convex.lp_dual_lower_bound`, outward-rounded interval arithmetic
  over :mod:`omnibias.core.verified`); and

* the **decoded solution's objective** as the upper bound.

The result is a certified gap ``lower <= optimum <= objective``. Because the assignment /
transportation / flow / matroid polytopes are **integral**, the LP relaxation equals the
integer optimum, so this gap is *tight* (``~0``) -- but the certificate reports the gap,
it never asserts exactness. Without ``omnibias-convex`` the bound degrades to the (still
valid) float LP optimum with ``certified=False``.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from omnibias.combinatorics._core.decode import solve_lp
from omnibias.combinatorics.problem import (
    AssignmentProblem,
    CombinatorialCertificate,
    MatroidProblem,
    MinCostFlowProblem,
    TransportProblem,
)

Problem: TypeAlias = AssignmentProblem | TransportProblem | MinCostFlowProblem | MatroidProblem


def _seal(system: object, lp_value: float, lam: object, nu: object) -> tuple[float, bool]:
    """Seal the LP value with the verified NS bound (``convex`` extra), else the float value."""
    try:
        from omnibias.convex import lp_dual_lower_bound
    except ImportError:  # convex not installed -> valid float bound, not interval-sealed
        return lp_value, False
    sys = system  # PolytopeSystem
    interval = lp_dual_lower_bound(
        sys.c,  # type: ignore[attr-defined]
        sys.A_ineq,  # type: ignore[attr-defined]
        sys.b_ineq,  # type: ignore[attr-defined]
        lam,
        A_eq=sys.A_eq,  # type: ignore[attr-defined]
        b_eq=sys.b_eq,  # type: ignore[attr-defined]
        eq_dual=nu,
        x_lower=sys.x_lower,  # type: ignore[attr-defined]
        x_upper=sys.x_upper,  # type: ignore[attr-defined]
    )
    return float(interval.lo), True


def certify_gap(problem: Problem, solution: object) -> CombinatorialCertificate:
    r"""Certify how close the decoded ``solution`` is to optimal for ``problem``.

    Parameters
    ----------
    problem:
        Any combinatorics problem (assignment / transport / min-cost flow / matroid).
    solution:
        A decoded feasible vertex as a flat variable vector (e.g. from
        :func:`omnibias.combinatorics.decode`) -- the upper bound. Its min-space objective
        ``c^T x`` is recomputed from the problem's cost.

    Returns
    -------
    :class:`~omnibias.combinatorics.problem.CombinatorialCertificate` with the rigorous
    (tight, integral-polytope) lower bound and the decoded objective.
    """
    system = problem.system()
    x = np.asarray(solution, dtype=float).reshape(-1)
    if x.shape[0] != system.n_vars:
        raise ValueError(f"solution must have {system.n_vars} entries, got {x.shape[0]}")
    objective = float(system.c @ x)

    _, lp_value, lam, nu = solve_lp(system)
    lower, certified = _seal(system, lp_value, lam, nu)
    method = "lp_dual" if certified else "lp_float"
    return CombinatorialCertificate(
        lower_bound=lower,
        objective=objective,
        polytope=system.name,
        method=method,
        certified=certified,
    )


__all__ = ["certify_gap"]
