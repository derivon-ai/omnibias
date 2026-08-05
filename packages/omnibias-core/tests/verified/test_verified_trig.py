# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rigor tests for the verified ``sin``/``cos`` interval tower and Fourier jets.

The soundness contract is *containment*: every enclosure produced by
``sin_iv`` / ``cos_iv``, the ``sin`` / ``cos`` rows of ``sigma_tower_interval``,
and a certified plane-wave jet must contain the true value at every sampled point
of its argument interval / box.  ``sin``/``cos`` are non-monotone, so these tests
deliberately straddle the extrema where a naive endpoint bound would be unsound.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_mv import jet_partials, mlp_jet_mv
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.core.verified.transcend import cos_iv, sin_iv

# A spread of intervals: point, sub-period, exactly straddling extrema, negative,
# and wider than a full 2*pi period (must saturate to [-1, 1]).
_INTERVALS = [
    (0.0, 0.0),
    (0.7, 0.7),
    (0.2, 0.5),
    (-0.3, 0.1),
    (1.4, 1.8),  # straddles pi/2 (sin max)
    (2.9, 3.4),  # straddles pi (cos min)
    (-1.7, -1.4),  # straddles -pi/2 (sin min)
    (6.0, 6.6),  # straddles 2*pi (cos max)
    (-5.0, -2.0),
    (0.0, 7.0),  # > 2*pi -> full range
    (-10.0, 10.0),  # many periods
    (100.0, 100.9),  # large argument, sub-period
]


def _samples(lo: float, hi: float, n: int = 400) -> list[float]:
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


@pytest.mark.parametrize(("lo", "hi"), _INTERVALS)
def test_cos_iv_contains_all_samples(lo: float, hi: float) -> None:
    enc = cos_iv(Interval(lo, hi))
    assert -1.0 <= enc.lo <= enc.hi <= 1.0
    for x in _samples(lo, hi):
        assert enc.lo <= math.cos(x) <= enc.hi


@pytest.mark.parametrize(("lo", "hi"), _INTERVALS)
def test_sin_iv_contains_all_samples(lo: float, hi: float) -> None:
    enc = sin_iv(Interval(lo, hi))
    assert -1.0 <= enc.lo <= enc.hi <= 1.0
    for x in _samples(lo, hi):
        assert enc.lo <= math.sin(x) <= enc.hi


def test_trig_iv_saturates_over_full_period() -> None:
    # Any interval at least 2*pi wide contains a max and a min of each function.
    wide = Interval(0.3, 0.3 + 2.0 * math.pi + 0.01)
    assert cos_iv(wide) == Interval(-1.0, 1.0)
    assert sin_iv(wide) == Interval(-1.0, 1.0)


def test_trig_iv_detects_exact_extrema() -> None:
    # An interval straddling pi pins cos's minimum to exactly -1.
    assert cos_iv(Interval(3.0, 3.3)).lo == -1.0
    # straddling 2*pi pins cos's maximum to +1.
    assert cos_iv(Interval(6.2, 6.4)).hi == 1.0
    # straddling pi/2 pins sin's maximum to +1.
    assert sin_iv(Interval(1.5, 1.7)).hi == 1.0


def test_trig_iv_tight_on_monotone_subinterval() -> None:
    # No interior extremum: cos is monotone decreasing on [0.2, 0.5], so the
    # enclosure must hug the endpoint values (no spurious saturation to +/-1) and
    # overshoot the true range by at most a few ulp on each side.
    enc = cos_iv(Interval(0.2, 0.5))
    assert 0.0 <= enc.hi - math.cos(0.2) < 1e-9  # upper end at cos(0.2)
    assert 0.0 <= math.cos(0.5) - enc.lo < 1e-9  # lower end at cos(0.5)


def test_inclusion_isotonic() -> None:
    inner = Interval(1.4, 1.8)
    outer = Interval(1.0, 2.5)
    ci, co = cos_iv(inner), cos_iv(outer)
    si, so = sin_iv(inner), sin_iv(outer)
    assert co.lo <= ci.lo and ci.hi <= co.hi
    assert so.lo <= si.lo and si.hi <= so.hi


# ----- the closed-form 4-cycle tower encloses the true derivative tower ----- #
mpmath = pytest.importorskip("mpmath")


@pytest.mark.parametrize("name", ["sin", "cos"])
@pytest.mark.parametrize("z0", [-1.3, -0.25, 0.0, 0.7, 2.1, 5.5])
def test_trig_tower_encloses_mpmath(name: str, z0: float) -> None:
    order = 7

    def f(z: object) -> object:
        return mpmath.cos(z) if name == "cos" else mpmath.sin(z)

    with mpmath.workdps(50):
        taylor = mpmath.taylor(f, z0, order)  # a_k = f^(k)(z0)/k!
        tower_true = [taylor[k] * math.factorial(k) for k in range(order + 1)]

    tower = sigma_tower_interval(name, Interval.point(z0), order)
    for k in range(order + 1):
        assert tower[k].lo <= float(tower_true[k]) <= tower[k].hi
        assert tower[k].width <= 1e-9 + 1e-9 * abs(float(tower_true[k]))


def test_trig_tower_is_four_periodic() -> None:
    # f^(k) == f^(k+4): the enclosures must coincide structurally.
    z = Interval(0.2, 0.6)
    for name in ("sin", "cos"):
        tower = sigma_tower_interval(name, z, 8)
        for k in range(5):
            assert tower[k] == tower[k + 4]


# ----- a classical plane-wave / Fourier mode is now certifiable ----- #
def _cos_deriv(level: int, arg: float) -> float:
    """``cos^(level)(arg) = cos(arg + level*pi/2)`` evaluated with libm."""
    return math.cos(arg + level * math.pi / 2.0)


def test_plane_wave_jet_encloses_analytic_partials() -> None:
    # f(x, y) = cos(w1 x + w2 y + b): a single Fourier mode on a box.
    w1, w2, b = 1.3, -0.7, 0.4
    order = 4
    box = [(0.2, 0.5), (-0.3, 0.1)]
    jet = mlp_jet_mv(box, [([[w1, w2]], [b], "cos")], order)
    partials = jet_partials(jet, dim=2, order=order)

    # Soundness: every mixed partial encloses the analytic value at interior points.
    xs = [0.2, 0.31, 0.5]
    ys = [-0.3, -0.05, 0.1]
    for alpha, enc in partials.items():
        p, q = alpha
        for x in xs:
            for y in ys:
                arg = w1 * x + w2 * y + b
                true = (w1**p) * (w2**q) * _cos_deriv(p + q, arg)
                assert enc[0].lo <= true <= enc[0].hi


def test_separable_fourier_mode_via_jet_product() -> None:
    # u(x, y) = sin(x) * cos(y): product of two one-axis modes (Cauchy product).
    from omnibias.core.verified.jet_mv import jet_multiply

    order = 4
    box = [(0.25, 0.55), (0.8, 1.15)]
    sin_x = mlp_jet_mv(box, [([[1.0, 0.0]], [0.0], "sin")], order)
    cos_y = mlp_jet_mv(box, [([[0.0, 1.0]], [0.0], "cos")], order)
    prod = jet_multiply(sin_x, cos_y, dim=2, order=order)
    partials = jet_partials(prod, dim=2, order=order)

    xs = [0.25, 0.4, 0.55]
    ys = [0.8, 1.0, 1.15]
    for alpha, enc in partials.items():
        p, q = alpha
        for x in xs:
            for y in ys:
                # d^p/dx^p sin(x) = sin^(p)(x); d^q/dy^q cos(y) = cos^(q)(y)
                true = math.sin(x + p * math.pi / 2.0) * math.cos(y + q * math.pi / 2.0)
                assert enc[0].lo <= true <= enc[0].hi
