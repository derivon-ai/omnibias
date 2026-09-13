# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite xi / chi / functional-equation evaluator tests."""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.dirichlet import zeta_enclosure, zeta_euler_maclaurin
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.xi import (
    chi_factor,
    continuation_honesty,
    xi_functional_equation_residual,
    zeta_continued,
    zeta_strip_winding_enclosure,
    zeta_via_functional_equation,
    zeta_winding_count,
)


def _encloses(enc: ComplexInterval, value: complex) -> bool:
    return enc.re.contains(value.real) and enc.im.contains(value.imag)


def test_honesty_never_claims_rh() -> None:
    h = continuation_honesty()
    assert h["rh_claim"] is False
    assert h["continuation_theorem"] is False
    assert h["zeros"] is False


def test_zeta_continued_halfplane_matches_series() -> None:
    left = zeta_continued(2.0, num_terms=200)
    right = zeta_enclosure(2.0, num_terms=200)
    pi2_over_6 = math.pi**2 / 6.0
    assert left.re.contains(pi2_over_6)
    assert right.re.contains(pi2_over_6)


@pytest.mark.parametrize("s", [-2.0, -4.0, -6.0, -20.0])
def test_zeta_via_fe_trivial_zero(s: float) -> None:
    assert zeta_via_functional_equation(s) == ComplexInterval.zero()


@pytest.mark.parametrize("s", [-2.0 - 1e-13, -2.0 + 1e-13])
def test_zeta_via_fe_near_trivial_zero_is_nonzero(s: float) -> None:
    mp = pytest.importorskip("mpmath")
    enc = zeta_via_functional_equation(s)
    with mp.workdps(60):
        true = complex(mp.zeta(mp.mpf(s)))
    assert true.real != 0.0
    assert not enc.re.contains(0.0)
    assert _encloses(enc, true), (s, enc, true)


def test_zeta_via_fe_tiny_nonzero_rectangle_encloses_grid_and_random() -> None:
    mp = pytest.importorskip("mpmath")
    # The old tolerance shortcut mapped this entire nonzero rectangle to {0}.
    box = ComplexInterval(Interval(-2.0 + 2e-13, -2.0 + 8e-13), Interval(-1e-14, 1e-14))
    enc = zeta_via_functional_equation(box)
    assert not enc.re.contains(0.0)
    rng = random.Random(712)
    points = [
        complex(box.re.lo + i * (box.re.hi - box.re.lo) / 8, -1e-14 + j * 2e-14 / 8)
        for i in range(9)
        for j in range(9)
    ]
    points.extend(
        complex(rng.uniform(box.re.lo, box.re.hi), rng.uniform(box.im.lo, box.im.hi))
        for _ in range(30)
    )
    with mp.workdps(60):
        for s in points:
            true = complex(mp.zeta(mp.mpc(s.real, s.imag)))
            assert _encloses(enc, true), (s, enc, true)


@pytest.mark.parametrize(
    "box",
    [
        ComplexInterval(Interval(-2.0 - 1e-13, -2.0 + 1e-13), Interval(-1e-13, 1e-13)),
        ComplexInterval(Interval.point(-2.0), Interval(-1e-13, 1e-13)),
        ComplexInterval(Interval(-2.0 - 1e-13, -2.0 + 1e-13), Interval.point(0.0)),
    ],
)
def test_zeta_via_fe_tiny_pole_crossing_rectangles_are_refused(box: ComplexInterval) -> None:
    # These boxes contain a trivial zero and nonzero values, so {0} is unsound.
    with pytest.raises(ValueError, match="pole"):
        zeta_via_functional_equation(box)


def test_zeta_via_fe_nearest_nonzero_float_is_refused() -> None:
    # Outward rounding cannot separate this Gamma argument from its pole.
    with pytest.raises(ValueError, match="pole"):
        zeta_via_functional_equation(math.nextafter(-2.0, 0.0))


def test_zeta_continued_zero_encloses_negative_half() -> None:
    enc = zeta_continued(0.0)
    assert _encloses(enc, -0.5 + 0.0j)
    assert not enc.re.contains(0.0)


def test_zeta_via_fe_encloses_mpmath_left_half() -> None:
    mp = pytest.importorskip("mpmath")
    for s in (-1.0, -0.5, -1.5 + 0.3j, -0.7 - 0.4j):
        enc = zeta_via_functional_equation(s, num_terms=200)
        with mp.workdps(40):
            true = complex(mp.zeta(mp.mpc(complex(s).real, complex(s).imag)))
        assert _encloses(enc, true), (s, enc, true)


@pytest.mark.parametrize("s", [0.0, 1e-13, 0.5])
def test_zeta_via_fe_refuses_nonnegative_real_part(s: float) -> None:
    with pytest.raises(ValueError, match="Re\\(s\\) < 0"):
        zeta_via_functional_equation(s)
    with pytest.raises(ValueError, match="pole"):
        zeta_continued(1.0)


def test_chi_times_zeta_matches_halfplane() -> None:
    mp = pytest.importorskip("mpmath")
    s = 2.5
    # zeta(s) = chi(s) zeta(1-s); 1-s = -1.5 is Re<0 so FE path on the right.
    left = zeta_enclosure(s, num_terms=200)
    right = chi_factor(s) * zeta_via_functional_equation(1.0 - s, num_terms=200)
    with mp.workdps(40):
        true = complex(mp.zeta(mp.mpf(s)))
    assert _encloses(left, true)
    assert _encloses(right, true)


def test_xi_functional_equation_residual_contains_zero() -> None:
    residual = xi_functional_equation_residual(2.0, num_terms=200)
    assert residual.re.contains(0.0)
    assert residual.im.contains(0.0)
    assert residual.re.width < 0.25
    assert residual.im.width < 0.25
    residual_c = xi_functional_equation_residual(2.3 + 0.2j, num_terms=200)
    assert residual_c.re.contains(0.0)
    assert residual_c.im.contains(0.0)


def test_zeta_continued_strip_agrees_with_em() -> None:
    mp = pytest.importorskip("mpmath")
    s = 0.5 + 3.0j
    enc = zeta_continued(s, num_terms=80)
    em = zeta_euler_maclaurin(s, num_sum_terms=25, order=6)
    with mp.workdps(40):
        true = complex(mp.zeta(mp.mpc(s.real, s.imag)))
    assert _encloses(enc, true)
    assert _encloses(em, true)


def test_zeta_winding_zero_off_the_strip() -> None:
    count, winding = zeta_winding_count(3.0 + 0.0j, 0.25, 0.25, segments=8, num_terms=60)
    assert count == 0
    assert winding is not None
    assert winding.contains(0.0)


def test_zeta_winding_refuses_strip_rectangle() -> None:
    with pytest.raises(ValueError, match="Re\\(s\\) > 1"):
        zeta_winding_count(0.5 + 14.13j, 0.2, 0.2)


def test_zeta_strip_winding_may_block() -> None:
    winding = zeta_strip_winding_enclosure(
        0.5 + 14.13j, 0.15, 0.15, segments=8, num_sum_terms=12, em_order=4
    )
    # Fat Euler-Maclaurin images on a zero-adjacent box are allowed to BLOCK.
    assert winding is None or not winding.contains_zero()
    assert continuation_honesty()["zeros"] is False
