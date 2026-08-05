# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Verified LP optimality certificate (Neumaier-Shcherbina weak-duality bound).

The lower bound is rigorous for *any* ``lambda >= 0`` / ``nu`` given a finite box
containing the feasible set -- so the first-order penalty solver's approximate
duals are admissible, looser duals only loosen (never invalidate) the bound, and
the exact optimum is always sandwiched ``lower <= f* <= upper = c^T x``.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.convex import CertificationError, certify_lp_optimum, lp_dual_lower_bound

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from omnibias.convex.jax import solve_lp, solve_lp_penalty  # noqa: E402

scipy_linprog = pytest.importorskip("scipy.optimize").linprog


def _arr(x: object) -> jnp.ndarray:
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def _lp_2d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # min -3x - 2y  s.t. x+y<=4, x+3y<=6, x>=0, y>=0 ; optimum (4, 0), f* = -12.
    c = np.array([-3.0, -2.0])
    A = np.array([[1.0, 1.0], [1.0, 3.0], [-1.0, 0.0], [0.0, -1.0]])
    b = np.array([4.0, 6.0, 0.0, 0.0])
    return c, A, b


def test_lp_certificate_sandwiches_optimum() -> None:
    c, A, b = _lp_2d()
    sol = solve_lp(_arr(c), _arr(A), _arr(b))
    # The upper bound needs a *rigorously* feasible point; a vertex sits on active
    # constraints whose outward-rounded slack straddles 0, so use an interior point
    # (its c^T x is a valid, if looser, upper bound). The lower bound stays tight.
    x_feas = np.array([1.0, 1.0])
    cert = certify_lp_optimum(
        c, A, b, x_feas, np.asarray(sol.dual), x_lower=0.0, x_upper=10.0
    )
    f_star = -12.0
    assert cert.primal_feasible
    assert cert.enclosure.lo <= f_star <= cert.enclosure.hi
    assert cert.enclosure.lo == pytest.approx(f_star, abs=1e-1)  # near-optimal dual
    assert cert.gap >= 0.0


def test_lower_bound_is_sound_for_arbitrary_dual() -> None:
    # lambda = 0 is admissible: bound = min_{x in box} c^T x <= f* (very loose).
    c, A, b = _lp_2d()
    ref = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None)).fun
    lo0 = lp_dual_lower_bound(c, A, b, np.zeros(4), x_lower=0.0, x_upper=10.0)
    assert lo0.lo <= ref + 1e-9


def test_better_dual_gives_tighter_lower_bound() -> None:
    # The solver's near-optimal dual should beat the trivial lambda = 0 bound.
    c, A, b = _lp_2d()
    ref = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None)).fun
    sol = solve_lp(_arr(c), _arr(A), _arr(b))
    lo0 = lp_dual_lower_bound(c, A, b, np.zeros(4), x_lower=0.0, x_upper=10.0)
    lo1 = lp_dual_lower_bound(c, A, b, np.asarray(sol.dual), x_lower=0.0, x_upper=10.0)
    assert lo0.lo <= lo1.lo <= ref + 1e-6
    assert abs(lo1.lo - ref) < 1e-1  # near-optimal dual -> near-tight bound


def test_lp_certificate_with_equalities_simplex() -> None:
    # min c^T x  s.t. 1^T x = 1, x >= 0 ; optimum vertex (1, 0, 0), f* = -3.
    c = np.array([-3.0, -1.0, -2.0])
    A = -np.eye(3)
    b = np.zeros(3)
    A_eq = np.ones((1, 3))
    b_eq = np.array([1.0])
    sol = solve_lp_penalty(
        _arr(c), _arr(A), _arr(b), A_eq=_arr(A_eq), b_eq=_arr(b_eq),
    )
    x_feas = np.array([0.5, 0.25, 0.25])  # interior of the simplex (rigorously feasible)
    cert = certify_lp_optimum(
        c, A, b, x_feas, np.asarray(sol.dual),
        A_eq=A_eq, b_eq=b_eq, eq_dual=np.asarray(sol.eq_dual),
        x_lower=0.0, x_upper=1.0,
    )
    assert cert.enclosure.lo <= -3.0 <= cert.enclosure.hi
    assert cert.gap >= 0.0


def test_certify_lp_raises_on_infeasible_x() -> None:
    c, A, b = _lp_2d()
    with pytest.raises(CertificationError):
        certify_lp_optimum(
            c, A, b, np.array([5.0, 5.0]), np.zeros(4), x_lower=0.0, x_upper=10.0
        )


def test_certify_lp_requires_finite_box() -> None:
    c, A, b = _lp_2d()
    with pytest.raises(CertificationError):
        lp_dual_lower_bound(c, A, b, np.zeros(4), x_lower=0.0, x_upper=np.inf)
