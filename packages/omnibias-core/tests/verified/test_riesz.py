# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified 2-D Riesz / Leray tests against an independent mpmath reference."""

from __future__ import annotations

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.riesz import (
    blob,
    blob_gradient,
    leray_blob_field,
    leray_divergence_residual,
    newtonian_potential,
    riesz_double_blob,
    riesz_tail_bound,
)

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

_PTS = [(0.3, -0.2), (1.1, 0.4), (-0.5, 0.9), (0.0, 0.6)]
_A = 0.7


def _N(x: float, y: float, a: float) -> object:
    return mp.log(x * x + y * y + a * a) / (4 * mp.pi)


def test_blob_is_laplacian_of_potential() -> None:
    for x, y in _PTS:
        hxx = mp.diff(lambda t: _N(t, y, _A), x, 2)  # noqa: B023
        hyy = mp.diff(lambda t: _N(x, t, _A), y, 2)  # noqa: B023
        lap = float(hxx + hyy)
        fb = blob(x, y, _A)
        assert fb.lo <= lap <= fb.hi


def test_potential_matches_mpmath() -> None:
    for x, y in _PTS:
        enc = newtonian_potential(x, y, _A)
        assert enc.lo <= float(_N(x, y, _A)) <= enc.hi


def _f(x: float, y: float, a: float) -> object:
    return a * a / (mp.pi * (x * x + y * y + a * a) ** 2)


def test_blob_gradient_matches_mpmath() -> None:
    for x, y in _PTS:
        fx = float(mp.diff(lambda t: _f(t, y, _A), x))  # noqa: B023
        fy = float(mp.diff(lambda t: _f(x, t, _A), y))  # noqa: B023
        gx, gy = blob_gradient(x, y, _A)
        assert gx.lo <= fx <= gx.hi
        assert gy.lo <= fy <= gy.hi


def test_blob_gradient_is_radial_and_perp_to_biot_savart() -> None:
    # nabla f_a is parallel to (x, y); the Biot-Savart velocity nabla^perp N_a is
    # parallel to (-y, x); their dot product is the zero polynomial (-y*x + x*y).
    for x, y in _PTS:
        gx, gy = blob_gradient(x, y, _A)
        # u ~ (-y, x): perpendicularity dot = (-y) gx + (x) gy must enclose 0.
        dot = Interval.point(-y) * gx + Interval.point(x) * gy
        assert dot.lo <= 0.0 <= dot.hi


def test_blob_gradient_zero_scale_rejected() -> None:
    with pytest.raises(ValueError):
        blob_gradient(0.1, 0.2, 0.0)


def test_double_riesz_matches_mpmath_hessian() -> None:
    for x, y in _PTS:
        hxx = mp.diff(lambda t: _N(t, y, _A), x, 2)  # noqa: B023
        hyy = mp.diff(lambda t: _N(x, t, _A), y, 2)  # noqa: B023
        hxy = mp.diff(lambda u: mp.diff(lambda t: _N(t, u, _A), x), y)  # noqa: B023
        r11 = riesz_double_blob(0, 0, x, y, _A)
        r22 = riesz_double_blob(1, 1, x, y, _A)
        r12 = riesz_double_blob(0, 1, x, y, _A)
        r21 = riesz_double_blob(1, 0, x, y, _A)
        assert r11.lo <= float(-hxx) <= r11.hi
        assert r22.lo <= float(-hyy) <= r22.hi
        assert r12.lo <= float(-hxy) <= r12.hi
        assert r21.lo <= float(-hxy) <= r21.hi  # symmetry


def test_riesz_trace_identity_equals_minus_blob() -> None:
    # R_11 f + R_22 f = -f  (the (xi_1^2 + xi_2^2)/|xi|^2 = 1 multiplier identity).
    for x, y in _PTS:
        r11 = riesz_double_blob(0, 0, x, y, _A)
        r22 = riesz_double_blob(1, 1, x, y, _A)
        fb = blob(x, y, _A)
        s = r11 + r22
        assert s.lo <= -fb.mid <= s.hi
        # tight: the trace sum and -blob agree to rounding
        assert (s + fb).abs().hi < 1e-12


def test_leray_field_matches_mpmath_closed_form() -> None:
    c1, c2 = 1.3, -0.8
    for x, y in _PTS:
        d = x * x + y * y + _A * _A
        denom = 2 * mp.pi * d * d
        pv1_true = (c1 * (_A**2 + x * x - y * y) + 2 * c2 * x * y) / denom
        pv2_true = (c2 * (_A**2 + y * y - x * x) + 2 * c1 * x * y) / denom
        pv1, pv2 = leray_blob_field(c1, c2, x, y, _A)
        assert pv1.lo <= float(pv1_true) <= pv1.hi
        assert pv2.lo <= float(pv2_true) <= pv2.hi


def test_leray_is_divergence_free_residual_encloses_zero() -> None:
    for c1, c2 in [(1.3, -0.8), (0.0, 1.0), (-2.1, 0.5)]:
        for x, y in _PTS:
            res = leray_divergence_residual(c1, c2, x, y, _A)
            assert res.lo <= 0.0 <= res.hi
            assert res.width < 1e-10


def test_leray_divergence_free_via_independent_mpmath() -> None:
    c1, c2 = 1.3, -0.8

    def pv1(x: float, y: float) -> object:
        d = x * x + y * y + _A * _A
        return (c1 * (_A**2 + x * x - y * y) + 2 * c2 * x * y) / (2 * mp.pi * d * d)

    def pv2(x: float, y: float) -> object:
        d = x * x + y * y + _A * _A
        return (c2 * (_A**2 + y * y - x * x) + 2 * c1 * x * y) / (2 * mp.pi * d * d)

    for x, y in _PTS:
        div = mp.diff(lambda t: pv1(t, y), x) + mp.diff(lambda t: pv2(x, t), y)  # noqa: B023
        # analytic div is exactly 0; mp.diff is finite-difference, so ~1e-17 floor
        assert abs(float(div)) < 1e-12


def test_riesz_tail_bound_upper_bounds_real_tail_at_origin() -> None:
    # At rho = 0 (x0 = origin) with K_const = C = 1 the real tail integral
    # int_{|t|>=X} |t|^{-p}/|t|^2 dt = 2*pi*X^{-p}/p; B must enclose/upper-bound it.
    p, x_trunc = 3.0, 2.0
    real_tail = float(2 * mp.pi * mp.mpf(x_trunc) ** (-p) / p)
    b = riesz_tail_bound(1.0, 1.0, p, x_trunc, 0.0)
    assert b.hi >= real_tail
    assert b.lo <= real_tail <= b.hi  # symmetric interval contains the positive tail
    # tight at rho = 0: bound equals the integral to rounding
    assert abs(b.hi - real_tail) < 1e-12


def test_riesz_tail_bound_grows_with_core_radius() -> None:
    base = riesz_tail_bound(1.0, 1.0, 3.0, 2.0, 0.0).hi
    closer = riesz_tail_bound(1.0, 1.0, 3.0, 2.0, 1.5).hi
    assert closer > base  # shrinking the gap X - rho inflates the bound


def test_riesz_tail_bound_validation() -> None:
    with pytest.raises(ValueError):
        riesz_tail_bound(1.0, 1.0, 0.0, 2.0, 0.0)  # p <= 0
    with pytest.raises(ValueError):
        riesz_tail_bound(1.0, 1.0, 3.0, 2.0, 2.0)  # rho >= X
    with pytest.raises(ValueError):
        riesz_tail_bound(-1.0, 1.0, 3.0, 2.0, 0.0)  # negative kernel const


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        blob(0.1, 0.2, 0.0)  # zero scale
    with pytest.raises(ValueError):
        riesz_double_blob(2, 0, 0.1, 0.2, 0.7)  # bad axis
