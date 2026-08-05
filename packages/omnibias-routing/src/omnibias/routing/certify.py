# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous optimality-gap certificate for a decoded tour.

:func:`certify_tour_gap` sandwiches the true optimal tour cost between

* a **rigorous lower bound** -- the chosen poly-size relaxation's LP optimum
  (assignment / flow / Held-Karp), solved exactly (scipy HiGHS) and then **sealed**
  by the Neumaier-Shcherbina verified bound (:func:`omnibias.convex.lp_dual_lower_bound`,
  outward-rounded interval arithmetic over :mod:`omnibias.core.verified`) so it is a
  genuine certificate that survives floating-point error. Any relaxation feasible
  region contains every integer tour, so its LP optimum is ``<=`` the true optimum;

* the **decoded tour cost** as the upper bound.

The result is a certified optimality gap ``lower <= optimum <= tour_cost`` -- never an
exact-optimality (P = NP) claim, and honest about relaxation strength (a weaker
relaxation only widens the certified gap). The exact LP solve needs ``scipy`` (a core
dependency); the rigorous interval seal reuses the ``convex`` extra -- without it the
bound degrades gracefully to the (still valid) float LP value with ``certified=False``.
"""

from __future__ import annotations

import numpy as np
from omnibias.routing._core.decode import is_valid_tour, tour_cost
from omnibias.routing._core.relax_systems import RelaxSystem, build_system
from omnibias.routing.problem import HeldKarpCertificate, RoutingProblem


class CertificationError(RuntimeError):
    """Raised when the relaxation LP cannot be solved for the certificate."""


def _solve_relaxation_lp(c: np.ndarray, system: RelaxSystem) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact LP optimum + dual multipliers (scipy HiGHS) for the relaxation."""
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise CertificationError(
            "certify_tour_gap needs scipy for the exact LP solve: pip install scipy"
        ) from exc
    res = linprog(
        c, A_ub=system.A_ineq, b_ub=system.b_ineq,
        A_eq=system.A_eq, b_eq=system.b_eq, bounds=(None, None), method="highs",
    )
    if not res.success:
        raise CertificationError(f"relaxation LP solve failed: {res.message}")
    # HiGHS marginals are d(obj)/d(rhs); the Lagrangian sign convention used by the
    # Neumaier-Shcherbina bound (lambda >= 0 for A x <= b) is their negation.
    lam = -np.asarray(res.ineqlin.marginals, dtype=float)
    nu = -np.asarray(res.eqlin.marginals, dtype=float)
    return float(res.fun), lam, nu


def _seal_lower_bound(
    c: np.ndarray, system: RelaxSystem, lp_value: float, lam: np.ndarray, nu: np.ndarray
) -> tuple[float, bool]:
    """Rigorously seal the LP value with the NS bound (``convex`` extra), else float."""
    try:
        from omnibias.convex import lp_dual_lower_bound
    except ImportError:  # convex not installed -> valid float bound, not interval-sealed
        return lp_value, False
    interval = lp_dual_lower_bound(
        c, system.A_ineq, system.b_ineq, lam,
        A_eq=system.A_eq, b_eq=system.b_eq, eq_dual=nu,
        x_lower=system.x_lower, x_upper=system.x_upper,
    )
    return float(interval.lo), True


def certify_tour_gap(
    problem: RoutingProblem,
    tour: tuple[int, ...] | list[int],
    *,
    kind: str = "flow",
) -> HeldKarpCertificate:
    r"""Certify how close ``tour`` is to optimal for ``problem`` (rigorous gap).

    Parameters
    ----------
    problem:
        The routing instance (its ``cost`` matrix defines optimality).
    tour:
        A valid tour (permutation of ``range(n)``) -- the upper bound.
    kind:
        Relaxation strength for the lower bound: ``"assignment"`` (loosest),
        ``"flow"`` (default, subtour-free), or ``"held_karp"`` (tightest, small ``n``).

    Returns
    -------
    :class:`~omnibias.routing.problem.HeldKarpCertificate` with the rigorous lower
    bound, the tour cost, and ``certified=True`` when the interval seal is available.
    """
    cost = problem.cost
    n = problem.n
    if not is_valid_tour(tuple(tour), n):
        raise ValueError(f"tour must be a permutation of range({n}), got {tuple(tour)}")

    system = build_system(n, kind)
    c = system.cost_vector(cost)
    lp_value, lam, nu = _solve_relaxation_lp(c, system)
    lower, certified = _seal_lower_bound(c, system, lp_value, lam, nu)
    upper = tour_cost(tuple(tour), cost)
    return HeldKarpCertificate(
        lower_bound=lower,
        tour_cost=upper,
        relaxation=kind,
        certified=certified,
    )


__all__ = ["CertificationError", "certify_tour_gap"]
