# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""W5 -- certified truncation error for finite-difference PDE stencils.

These verify that the ``omnibias-difference`` remainder engine, decoupled from the
activation dictionary, certifies the truncation error of standard PDE stencils for
arbitrary smooth functions (including ``exp``, which is *not* one of the nine
built-in activations), that the certified consistency order matches the measured
empirical order (the classical baseline), and that a separable discrete Laplacian
sums the per-axis 1-D certificates soundly.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.verify import (
    certified_laplacian_truncation,
    certified_stencil_truncation,
    measured_consistency_order,
)

# Derivative-tower oracles that do NOT go through the finite-difference activation
# dictionary (exp is not one of the nine activations at all).
_EXP_BOUND = lambda k, box: exp_iv(box)  # noqa: E731 -- all derivatives of exp are exp
_SIN_BOUND = lambda k, box: sigma_tower_interval("sin", box, k)[k]  # noqa: E731
_COS_BOUND = lambda k, box: sigma_tower_interval("cos", box, k)[k]  # noqa: E731


def _poly_bound(k: int, box: Interval) -> Interval:
    """Exact derivative tower of f(x) = x^3 (proves the engine needs no dictionary)."""
    if k == 0:
        return box.pow_int(3)
    if k == 1:
        return Interval.from_rational(3) * box.pow_int(2)
    if k == 2:
        return Interval.from_rational(6) * box
    if k == 3:
        return Interval.from_rational(6)
    return Interval.point(0.0)


# --------------------------------------------------------------------------- #
# Certified truncation on functions outside the activation dictionary          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("x", [0.3, 0.7, 1.2, -0.4])
def test_exp_second_derivative_certified(x: float) -> None:
    cert = certified_stencil_truncation(math.exp, _EXP_BOUND, x, 2, 0.05, "central")
    true = math.exp(x)  # exp'' = exp
    assert cert.consistency_order == 2
    assert cert.consistent
    assert cert.derivative_enclosure.contains(true)
    assert cert.true_value_interval.contains(true)
    assert abs(cert.estimate - true) <= cert.truncation_bound + 1e-12


def test_cubic_central_second_derivative_is_essentially_exact() -> None:
    # f=x^3 has f''''=0, so the central 2nd-derivative stencil has ~zero *truncation*.
    cert = certified_stencil_truncation(lambda t: t**3, _poly_bound, 1.5, 2, 0.1, "central")
    assert cert.truncation_bound < 1e-15  # underflows to ~0 (only the smallest subnormal)
    assert cert.derivative_enclosure.contains(9.0)
    # The certified bound is about TRUNCATION, not float roundoff; the plain-float
    # estimate still carries ~1e-14 machine error, so we check it to float tol here.
    assert cert.estimate == pytest.approx(9.0, abs=1e-12)


def test_truncation_bound_shrinks_like_h_squared() -> None:
    # central stencil: halving h quarters the certified bound (O(h^2)); the extra
    # factor exp(0.05) is the sup|f''''| box [x-h, x+h] widening with h.
    b1 = certified_stencil_truncation(math.exp, _EXP_BOUND, 0.5, 2, 0.1, "central").truncation_bound
    b2 = certified_stencil_truncation(math.exp, _EXP_BOUND, 0.5, 2, 0.05, "central").truncation_bound
    assert b2 < b1
    assert b1 / b2 == pytest.approx(4.0 * math.exp(0.05), rel=1e-3)


# --------------------------------------------------------------------------- #
# Certified consistency order == measured empirical order (the baseline)       #
# --------------------------------------------------------------------------- #
_STEPS = [0.2, 0.1, 0.05, 0.025]


def test_central_first_derivative_order_two() -> None:
    cert = certified_stencil_truncation(math.sin, _SIN_BOUND, 0.4, 1, 0.05, "central")
    emp = measured_consistency_order(math.sin, _SIN_BOUND, 0.4, 1, _STEPS, "central")
    assert cert.consistency_order == 2
    assert abs(emp - 2.0) < 0.15


def test_central_second_derivative_order_two() -> None:
    cert = certified_stencil_truncation(math.sin, _SIN_BOUND, 0.4, 2, 0.05, "central")
    emp = measured_consistency_order(math.sin, _SIN_BOUND, 0.4, 2, _STEPS, "central")
    assert cert.consistency_order == 2
    assert abs(emp - 2.0) < 0.15


def test_forward_first_derivative_order_one() -> None:
    cert = certified_stencil_truncation(math.sin, _SIN_BOUND, 0.4, 1, 0.05, "forward")
    emp = measured_consistency_order(math.sin, _SIN_BOUND, 0.4, 1, _STEPS, "forward")
    assert cert.consistency_order == 1
    assert abs(emp - 1.0) < 0.2


def test_measured_order_needs_two_steps() -> None:
    with pytest.raises(ValueError, match="two step sizes"):
        measured_consistency_order(math.sin, _SIN_BOUND, 0.4, 1, [0.1], "central")


# --------------------------------------------------------------------------- #
# Separable discrete Laplacian: sum of per-axis 1-D certificates               #
# --------------------------------------------------------------------------- #
def test_separable_laplacian_sound() -> None:
    # f(x,y,z) = sin(x) + cos(y) + exp(z); Lap f = -sin(x) - cos(y) + exp(z).
    point = [0.4, 0.9, 0.2]
    cert = certified_laplacian_truncation(
        [math.sin, math.cos, math.exp],
        [_SIN_BOUND, _COS_BOUND, _EXP_BOUND],
        point,
        0.05,
    )
    true = -math.sin(0.4) - math.cos(0.9) + math.exp(0.2)
    assert cert.dimension == 3
    assert cert.consistency_order == 2
    assert cert.laplacian_enclosure.contains(true)
    assert cert.true_value_interval.contains(true)
    assert cert.consistent


def test_laplacian_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share length"):
        certified_laplacian_truncation([math.sin], [_SIN_BOUND, _COS_BOUND], [0.1, 0.2], 0.05)
