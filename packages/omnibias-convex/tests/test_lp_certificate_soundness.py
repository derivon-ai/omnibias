# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Independent exact optima exposing cancellation in certificate checks."""

from fractions import Fraction

import numpy as np
import pytest
from omnibias.convex import CertificationError, certify_lp_optimum, lp_dual_lower_bound


def cancellation_case(first):
    # Equalities force y=z=1 and x + 1e16*y - 1e16*z = 1, hence x=1 exactly.
    return dict(
        c=np.array([1.0, 0.0, 0.0]),
        A=np.zeros((1, 3)),
        b=np.ones(1),
        x=np.array([first, 1.0, 1.0]),
        dual=np.zeros(1),
        A_eq=np.array([[1.0, 1e16, -1e16], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        b_eq=np.ones(3),
        eq_dual=np.zeros(3),
        x_lower=np.zeros(3),
        x_upper=np.full(3, 2.0),
    )


@pytest.mark.parametrize("candidate", [0.0, 0.5, np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)])
def test_cancellation_does_not_prove_equality_feasibility(candidate):
    with pytest.raises(CertificationError, match="primal feasible"):
        certify_lp_optimum(**cancellation_case(candidate))


def test_exact_cancelling_equality_point_still_certifies():
    result = certify_lp_optimum(**cancellation_case(1.0))
    assert result.primal_feasible
    assert result.enclosure.lo <= 1 <= result.enclosure.hi


def test_rounding_close_simplex_point_is_not_exactly_feasible():
    x = np.array([0.5, np.nextafter(0.5, 1.0)])
    assert sum(Fraction.from_float(float(v)) for v in x) != 1
    with pytest.raises(CertificationError):
        certify_lp_optimum(
            np.zeros(2),
            -np.eye(2),
            np.zeros(2),
            x,
            np.zeros(2),
            A_eq=np.ones((1, 2)),
            b_eq=np.ones(1),
            x_lower=0.0,
            x_upper=1.0,
        )


def test_exact_dyadic_simplex_point_is_accepted():
    result = certify_lp_optimum(
        [-1.0, -2.0],
        -np.eye(2),
        np.zeros(2),
        [0.5, 0.5],
        [0.0, 0.0],
        A_eq=[[1.0, 1.0]],
        b_eq=[1.0],
        x_lower=0.0,
        x_upper=1.0,
    )
    assert result.enclosure.lo <= -2 <= result.enclosure.hi


@pytest.mark.parametrize("missing", ["A_eq", "b_eq"])
def test_partial_equality_pair_refuses_in_both_lp_entry_points(missing):
    args = cancellation_case(1.0)
    args[missing] = None
    with pytest.raises(CertificationError, match="together"):
        certify_lp_optimum(**args)
    args.pop("x")
    with pytest.raises(CertificationError, match="together"):
        lp_dual_lower_bound(**args)


def test_dual_residual_diagnostic_encloses_cancellation():
    # The exact stationarity residual is 1, while a floating dot can report 0.
    A = np.array([[1e16], [1.0], [-1e16]])
    result = certify_lp_optimum(
        [0.0], A, [2e16, 2.0, 0.0], [1.0], [1.0, 1.0, 1.0], x_lower=0.0, x_upper=2.0
    )
    assert result.dual_residual_inf >= 1
    assert result.enclosure.lo <= 0 <= result.enclosure.hi
