# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the sound Taylor-model antiderivative / definite integral.

Integration is the natural rigorous operation on a Taylor model (the remainder
integrates soundly, unlike differentiation of a flat remainder).  These tests
check the fundamental theorem of calculus on the polynomial part, the closed-form
definite integral on a symmetric cell, and that a non-zero remainder is absorbed
rigorously.

Containment is asserted with :mod:`_enclosure`, which takes no tolerance: where
the reference is an exact closed form, the enclosure must contain it outright.
A tolerance is used only for *tightness* claims (the enclosure is not wider than
some bound), which is the direction where being off by a rounding unit is
harmless rather than false.
"""

from __future__ import annotations

import random

import pytest
from _enclosure import assert_encloses, assert_encloses_interval
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.taylor_model import TaylorModel
from omnibias.core.verified.taylor_model_mv import TaylorModelMV


def _horner(coeffs: list[Interval], x: Interval) -> Interval:
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = acc * x + c
    return acc


# --------------------------------------------------------------------------- #
# 1-D TaylorModel.
# --------------------------------------------------------------------------- #
def test_antiderivative_of_x_squared() -> None:
    # f(x) = x^2 on [-0.5, 0.5]; F = x^3 / 3.
    f = TaylorModel(
        0.0,
        0.5,
        [Interval.point(0.0), Interval.point(0.0), Interval.point(1.0)],
        Interval.point(0.0),
    )
    F = f.antiderivative()
    assert F.order == 3
    assert F.coeffs[0].lo == 0.0 and F.coeffs[0].hi == 0.0
    assert abs(F.coeffs[3].mid - 1.0 / 3.0) < 1e-15
    assert F.coeffs[1].mid == 0.0 and F.coeffs[2].mid == 0.0


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_definite_integral_monomials(k: int) -> None:
    h = 0.6
    coeffs = [Interval.point(1.0 if j == k else 0.0) for j in range(k + 1)]
    f = TaylorModel(0.0, h, coeffs, Interval.point(0.0))
    di = f.definite_integral()
    analytic = 0.0 if k % 2 == 1 else 2.0 * h ** (k + 1) / (k + 1)
    assert_encloses(di, analytic, what=f"int x^{k}")
    assert di.width < 1e-10  # exact polynomial -> tight


def test_definite_integral_matches_antiderivative_endpoints() -> None:
    # exact polynomial 2 + 3x - x^2 ; remainder zero -> two routes agree exactly.
    h = 0.4
    f = TaylorModel(
        0.0,
        h,
        [Interval.point(2.0), Interval.point(3.0), Interval.point(-1.0)],
        Interval.point(0.0),
    )
    di = f.definite_integral()
    F = f.antiderivative()
    endpoints = _horner(F.coeffs, Interval.point(h)) - _horner(
        F.coeffs, Interval.point(-h)
    )
    assert abs(di.mid - endpoints.mid) < 1e-12
    # analytic: ∫_{-h}^{h} (2 + 3x - x^2) dx = 4h - 2h^3/3
    analytic = 4.0 * h - 2.0 * h**3 / 3.0
    assert_encloses(di, analytic, what="int (2 + 3x - x^2)")


def test_definite_integral_absorbs_remainder() -> None:
    # f(x) ∈ 1 + [-0.1, 0.1] on [-0.5, 0.5]; ∫ ∈ [0.9, 1.1].
    f = TaylorModel(0.0, 0.5, [Interval.point(1.0)], Interval(-0.1, 0.1))
    di = f.definite_integral()
    # Every g enclosed by f has an integral in [0.9, 1.1], so the enclosure must
    # cover that whole range -- not merely overlap it.
    assert_encloses_interval(di, (0.9, 1.1), what="attainable integrals")
    for c in (-0.1, 0.0, 0.07, 0.1):
        true_int = (1.0 + c) * 1.0  # width of cell = 1
        assert_encloses(di, true_int, what=f"int (1 + {c})")


def test_antiderivative_remainder_grows_with_radius() -> None:
    f = TaylorModel(0.0, 0.5, [Interval.point(1.0)], Interval(-0.2, 0.2))
    F = f.antiderivative()
    # A *tightness* claim: |F remainder| <= |R| * radius = 0.2 * 0.5 = 0.1. Overshooting
    # by a rounding unit would only make the model looser, never unsound, so unlike the
    # containment assertions above this one legitimately carries a tolerance.
    assert F.remainder.lo >= -0.1 - 1e-12 and F.remainder.hi <= 0.1 + 1e-12


# --------------------------------------------------------------------------- #
# Multivariate TaylorModelMV.
# --------------------------------------------------------------------------- #
def test_mv_antiderivative_cross_axis() -> None:
    # f(x, y) = x ; ∫ f dy = x*y, ∫ f dx = x^2 / 2.
    fx = TaylorModelMV.coordinate(0, [0.0, 0.0], [0.5, 0.5], 1)
    fy = fx.antiderivative(1)
    assert fy.order == 2
    val = fy.eval([Interval.point(0.3), Interval.point(0.4)])
    assert abs(val.mid - 0.12) < 1e-12  # 0.3 * 0.4
    fx2 = fx.antiderivative(0)
    val2 = fx2.eval([Interval.point(0.3), Interval.point(0.0)])
    assert abs(val2.mid - 0.045) < 1e-12  # 0.3^2 / 2


def test_mv_antiderivative_polynomial_ftc() -> None:
    # f(x, y) = x^2 + y on a box; ∫_x f = x^3/3 + x*y (constant 0 at x = center).
    center = [0.0, 0.0]
    radius = [0.4, 0.4]
    order = 2
    f = (
        TaylorModelMV.coordinate(0, center, radius, order).pow_int(2)
        + TaylorModelMV.coordinate(1, center, radius, order)
    )
    F = f.antiderivative(0)
    for dx, dy in [(0.3, 0.2), (-0.2, 0.4), (0.4, -0.4)]:
        analytic = dx**3 / 3.0 + dx * dy
        enc = F.eval([Interval.point(dx), Interval.point(dy)])
        assert_encloses(enc, analytic, what=f"F({dx}, {dy})")


def test_mv_antiderivative_absorbs_remainder() -> None:
    f = TaylorModelMV.constant(1.0, [0.0, 0.0], [0.5, 0.3], 1)
    f = TaylorModelMV(
        f.center, f.radius, f.order, f.coeffs, Interval(-0.1, 0.1)
    )
    F = f.antiderivative(1)
    # A tightness claim (see `test_antiderivative_remainder_grows_with_radius`):
    # |remainder| <= 0.1 * radius[1] = 0.1 * 0.3 = 0.03.
    assert F.remainder.lo >= -0.03 - 1e-12 and F.remainder.hi <= 0.03 + 1e-12


def test_mv_antiderivative_axis_validation() -> None:
    f = TaylorModelMV.constant(1.0, [0.0, 0.0], [0.5, 0.5], 1)
    with pytest.raises(ValueError):
        f.antiderivative(2)
    with pytest.raises(ValueError):
        f.antiderivative(-1)


def test_mv_definite_integral_sound_vs_grid_and_random() -> None:
    """Mandatory soundness check (verified-enclosures rule): the certified
    definite integral contains the true value, cross-checked against a dense
    deterministic grid AND a random sample, with a *genuine* remainder.

    ``g(x, y) = (x + y)^4`` is enclosed by an order-2 Taylor model, so every
    degree-``> 2`` term is rigorously absorbed into the interval remainder -- the
    ``remainder * vol`` path is exercised, not just the exact polynomial part.

    The headline assertion is against the closed form
    ``int int (x+y)^4 = 64 a^6 / 15`` (only the even-even terms of the binomial
    expansion survive the symmetric cell), so it needs no tolerance. The grid and
    Monte-Carlo estimates are approximations carrying their own error, so they are
    widened into reference *intervals* -- the error budget goes on the reference,
    never on the enclosure under test.
    """
    a = 0.4
    center = [0.0, 0.0]
    radius = [a, a]
    order = 2
    x = TaylorModelMV.coordinate(0, center, radius, order)
    y = TaylorModelMV.coordinate(1, center, radius, order)
    tm = (x + y).pow_int(4)  # order-2 TM of (x+y)^4 -> non-trivial remainder
    di = tm.definite_integral()

    def g(px: float, py: float) -> float:
        return (px + py) ** 4

    # dense deterministic midpoint grid
    n = 120
    h = 2.0 * a / n
    axis = [-a + h * (k + 0.5) for k in range(n)]
    grid = h * h * sum(g(px, py) for px in axis for py in axis)

    # random-sample Monte-Carlo
    rng = random.Random(0)
    samples = 200_000
    vol = (2.0 * a) ** 2
    mc = vol * sum(g(rng.uniform(-a, a), rng.uniform(-a, a)) for _ in range(samples)) / samples

    exact = 64.0 * a**6 / 15.0
    assert_encloses(di, exact, what="int int (x+y)^4")

    # Each estimate's error budget is derived, not guessed, so it is a statement about
    # the estimate rather than a concession on the enclosure. Composite midpoint:
    # E ~ (h^2/24) int (f_xx + f_yy); here f_xx = f_yy = 12 (x+y)^2 and
    # int (x+y)^2 = 8 a^4 / 3, giving (h^2/24) * 64 a^4. Doubled for the tail.
    quad_err = 2.0 * (h**2 / 24.0) * 64.0 * a**4
    # Monte-Carlo: 5 sigma of the sample mean, sigma^2 = vol^2 Var(g)/samples with
    # Var(g) <= E[g^2] = int (x+y)^8 / vol = (2 a^10 * 256/45) / vol.
    mc_err = 5.0 * vol * ((256.0 / 45.0) * 2.0 * a**10 / vol / samples) ** 0.5
    assert_encloses_interval(di, (grid - quad_err, grid + quad_err), what="midpoint estimate")
    assert_encloses_interval(di, (mc - mc_err, mc + mc_err), what="Monte-Carlo estimate")
    # Both estimates must actually land within their own budget of the closed form,
    # otherwise the checks above would be vacuously wide.
    assert abs(grid - exact) < quad_err, (grid, exact, quad_err)
    assert abs(mc - exact) < mc_err, (mc, exact, mc_err)
    # the enclosure must be non-vacuous
    assert di.width > 0.0
