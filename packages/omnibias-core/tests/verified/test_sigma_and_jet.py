# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Containment + tightness tests for the verified sigma tower and jets.

The independent oracle is ``mpmath.taylor`` (exact high-precision Taylor
coefficients); the verified enclosures must contain it and be tight.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest
from omnibias.core.polynomials import sech_polynomial_coeffs, tanh_polynomial_coeffs
from omnibias.core.verified.coeffs import (
    hermite_coeffs_exact,
    sech_poly_coeffs_exact,
    sigmoid_poly_coeffs_exact,
    tanh_poly_coeffs_exact,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet import (
    antiderivative_jet,
    derivative_jet,
    jet_to_tower,
    mlp_jet,
)
from omnibias.core.verified.sigma import sigma_tower_interval

mpmath = pytest.importorskip("mpmath")


# ----- exact coefficients match the float generators where float is exact ----- #
@pytest.mark.parametrize("n", range(0, 12))
def test_tanh_exact_coeffs_match_float(n: int) -> None:
    exact = tanh_poly_coeffs_exact(n)
    flt = tanh_polynomial_coeffs(n)
    assert len(exact) == len(flt)
    for ce, cf in zip(exact, flt, strict=True):
        assert float(ce) == cf  # still exact as double at these orders


def test_sigmoid_exact_coeffs_small() -> None:
    # P_1(s) = s - s^2 ; P_2(s) = s - 3 s^2 + 2 s^3.
    assert sigmoid_poly_coeffs_exact(1) == (0, 1, -1)
    assert sigmoid_poly_coeffs_exact(2) == (0, 1, -3, 2)


@pytest.mark.parametrize("n", range(0, 14))
def test_sech_exact_coeffs_match_float(n: int) -> None:
    exact = sech_poly_coeffs_exact(n)
    flt = sech_polynomial_coeffs(n)
    assert len(exact) == len(flt)
    for ce, cf in zip(exact, flt, strict=True):
        assert float(ce) == cf  # still exact as double at these orders


def test_sech_exact_coeffs_small() -> None:
    # Q_1(t) = -t ; Q_2(t) = 2 t^2 - 1 ; constant terms are Euler numbers.
    assert sech_poly_coeffs_exact(0) == (1,)
    assert sech_poly_coeffs_exact(1) == (0, -1)
    assert sech_poly_coeffs_exact(2) == (-1, 0, 2)
    assert [sech_poly_coeffs_exact(n)[0] for n in range(9)] == [1, 0, -1, 0, 5, 0, -61, 0, 1385]


def test_hermite_exact_coeffs_small() -> None:
    assert hermite_coeffs_exact(2) == (-1, 0, 1)  # He_2 = z^2 - 1
    assert hermite_coeffs_exact(3) == (0, -3, 0, 1)  # He_3 = z^3 - 3z


# ----- sigma tower encloses the true derivative tower ----- #
@pytest.mark.parametrize("name", ["tanh", "sigmoid", "gaussian"])
@pytest.mark.parametrize("z0", [-1.3, -0.25, 0.0, 0.7, 2.1])
def test_sigma_tower_encloses_mpmath(name: str, z0: float) -> None:
    order = 6

    def f(z: object) -> object:
        if name == "tanh":
            return mpmath.tanh(z)
        if name == "sigmoid":
            return mpmath.mpf(1) / (1 + mpmath.e ** (-z))
        return mpmath.e ** (-(z**2) / 2)

    with mpmath.workdps(50):
        taylor = mpmath.taylor(f, z0, order)  # a_k = f^(k)(z0)/k!
        tower_true = [taylor[k] * math.factorial(k) for k in range(order + 1)]

    tower = sigma_tower_interval(name, Interval.point(z0), order)
    for k in range(order + 1):
        assert tower[k].lo <= float(tower_true[k]) <= tower[k].hi
        # tightness: enclosure width is small at a point argument.
        assert tower[k].width <= 1e-9 + 1e-9 * abs(float(tower_true[k]))


@pytest.mark.parametrize("z0", [-1.3, -0.25, 0.0, 0.7, 2.1])
def test_sech_tower_encloses_mpmath(z0: float) -> None:
    """The ``sech`` tower encloses the true derivatives tightly at a point."""
    order = 6
    with mpmath.workdps(50):
        taylor = mpmath.taylor(lambda z: mpmath.sech(z), z0, order)  # a_k = f^(k)/k!
        tower_true = [taylor[k] * math.factorial(k) for k in range(order + 1)]

    tower = sigma_tower_interval("sech", Interval.point(z0), order)
    for k in range(order + 1):
        assert tower[k].lo <= float(tower_true[k]) <= tower[k].hi
        assert tower[k].width <= 1e-8 + 1e-8 * abs(float(tower_true[k]))


def test_sech_tower_encloses_interval_samples() -> None:
    """The ``sech`` interval tower over a box encloses the true derivative at a dense
    grid AND random samples inside that box (the founding delta->0 soundness rule)."""
    order = 5
    lo, hi = 0.3, 0.9
    rng = random.Random(11)
    tower = sigma_tower_interval("sech", Interval(lo, hi), order)
    samples = [lo + (hi - lo) * i / 20 for i in range(21)]
    samples += [rng.uniform(lo, hi) for _ in range(40)]
    with mpmath.workdps(50):
        for z0 in samples:
            taylor = mpmath.taylor(lambda z: mpmath.sech(z), z0, order)
            for k in range(order + 1):
                v = float(taylor[k] * math.factorial(k))
                assert tower[k].lo - 1e-12 <= v <= tower[k].hi + 1e-12


def _mp_smooth_neural(name: str) -> Callable[[object], object]:
    """The mpmath reference for the smooth closed-form neural activations."""
    if name == "silu":
        return lambda z: z / (1 + mpmath.e ** (-z))
    if name == "softplus":
        return lambda z: mpmath.log(1 + mpmath.e**z)
    # exact gelu: z * Phi(z),  Phi(z) = (1 + erf(z / sqrt 2)) / 2.
    return lambda z: z * (1 + mpmath.erf(z / mpmath.sqrt(2))) / 2


@pytest.mark.parametrize("name", ["silu", "gelu", "softplus"])
@pytest.mark.parametrize("z0", [-2.1, -0.7, -0.25, 0.0, 0.4, 1.3, 3.0])
def test_smooth_neural_tower_encloses_mpmath(name: str, z0: float) -> None:
    """silu / gelu / softplus towers enclose the true derivatives tightly at a point."""
    order = 6
    f = _mp_smooth_neural(name)
    with mpmath.workdps(50):
        taylor = mpmath.taylor(f, z0, order)  # a_k = f^(k)(z0)/k!
        tower_true = [taylor[k] * math.factorial(k) for k in range(order + 1)]

    tower = sigma_tower_interval(name, Interval.point(z0), order)
    for k in range(order + 1):
        assert tower[k].lo <= float(tower_true[k]) <= tower[k].hi
        # tightness: enclosure width is small at a point argument.
        assert tower[k].width <= 1e-8 + 1e-8 * abs(float(tower_true[k]))


@pytest.mark.parametrize("name", ["silu", "gelu", "softplus"])
def test_smooth_neural_tower_encloses_interval_samples(name: str) -> None:
    """The interval tower over a box encloses the true derivative at a dense grid
    AND random samples inside that box (the way ``mlp_jet_mv`` consumes it)."""
    order = 5
    lo, hi = 0.3, 0.9
    rng = random.Random(7)
    f = _mp_smooth_neural(name)
    tower = sigma_tower_interval(name, Interval(lo, hi), order)
    samples = [lo + (hi - lo) * i / 20 for i in range(21)]
    samples += [rng.uniform(lo, hi) for _ in range(40)]
    with mpmath.workdps(50):
        for z0 in samples:
            taylor = mpmath.taylor(f, z0, order)
            for k in range(order + 1):
                v = float(taylor[k] * math.factorial(k))
                assert tower[k].lo - 1e-12 <= v <= tower[k].hi + 1e-12


# ----- verified jet encloses the exact directional Taylor tower ----- #
@pytest.mark.parametrize("name", ["tanh", "sigmoid"])
def test_single_layer_jet_encloses_taylor(name: str) -> None:
    a, c = 1.37, -0.42
    order = 7

    def f(t: object) -> object:
        u = a * t + c
        if name == "tanh":
            return mpmath.tanh(u)
        return mpmath.mpf(1) / (1 + mpmath.e ** (-u))

    with mpmath.workdps(50):
        taylor = mpmath.taylor(f, 0.0, order)  # jet coefficients a_k

    jet = mlp_jet([0.0], [1.0], [([[a]], [c], name)], order)
    for k in range(order + 1):
        comp = jet[k][0]
        assert comp.lo <= float(taylor[k]) <= comp.hi
        assert comp.width <= 1e-8 + 1e-8 * abs(float(taylor[k]))


def test_two_layer_jet_encloses_taylor() -> None:
    # f(t) = tanh( w2 * tanh(w1 (x0 + t) + b1) + b2 )
    x0, w1, b1, w2, b2 = 0.3, 1.1, -0.2, 0.8, 0.05
    order = 5

    def f(t: object) -> object:
        h = mpmath.tanh(w1 * (x0 + t) + b1)
        return mpmath.tanh(w2 * h + b2)

    with mpmath.workdps(50):
        taylor = mpmath.taylor(f, 0.0, order)

    jet = mlp_jet(
        [x0],
        [1.0],
        [([[w1]], [b1], "tanh"), ([[w2]], [b2], "tanh")],
        order,
    )
    for k in range(order + 1):
        comp = jet[k][0]
        assert comp.lo <= float(taylor[k]) <= comp.hi


def test_jet_to_tower_roundtrip() -> None:
    jet = mlp_jet([0.2], [1.0], [([[1.0]], [0.0], "tanh")], 4)
    series = [jet[k][0] for k in range(len(jet))]
    tower = jet_to_tower(series)
    # tower[1] = tanh'(0.2) = 1 - tanh(0.2)^2
    t = math.tanh(0.2)
    assert tower[1].lo <= 1 - t * t <= tower[1].hi


# ----- two-sided (integral) tower: antiderivative_jet / derivative_jet ----- #
def test_antiderivative_jet_encloses_mpmath_and_inverts() -> None:
    """The verified integral tower of ``net(t) = tanh(a t + c)`` encloses the mpmath
    Taylor jet of its antiderivative, and ``derivative_jet`` inverts it (FTC part 1)."""
    a, c = 1.37, -0.42
    order = 6
    net = mlp_jet([0.0], [1.0], [([[a]], [c], "tanh")], order)
    series = [net[k][0] for k in range(len(net))]

    # Independent oracle: F(t) = (log cosh(a t + c) - log cosh c)/a, so F' = net, F(0)=0.
    def big_f(t: object) -> object:
        return (mpmath.log(mpmath.cosh(a * t + c)) - mpmath.log(mpmath.cosh(c))) / a

    with mpmath.workdps(50):
        taylor_f = mpmath.taylor(big_f, 0.0, order + 1)  # A_k, k = 0 .. order+1

    anti = antiderivative_jet(series)
    assert len(anti) == len(series) + 1
    for k in range(order + 2):
        assert anti[k].lo <= float(taylor_f[k]) <= anti[k].hi
        assert anti[k].width <= 1e-8 + 1e-8 * abs(float(taylor_f[k]))

    # FTC part 1: differentiating the antiderivative recovers the original jet.
    back = derivative_jet(anti)
    assert len(back) == len(series)
    for orig, got in zip(series, back, strict=True):
        assert got.lo <= orig.lo + 1e-12
        assert got.hi >= orig.hi - 1e-12
        assert got.width <= orig.width + 1e-12


def test_antiderivative_jet_constant_and_empty_edges() -> None:
    jet = [Interval.point(2.0), Interval.point(6.0), Interval.point(9.0)]
    anti = antiderivative_jet(jet, constant=Interval.point(-4.0))
    assert anti[0].lo == anti[0].hi == -4.0  # integration constant
    assert anti[1].lo <= 2.0 <= anti[1].hi  # A_1 = a_0 / 1
    assert anti[2].lo <= 3.0 <= anti[2].hi  # A_2 = a_1 / 2
    # a constant jet has no derivative information: derivative_jet yields length 0.
    assert derivative_jet([Interval.point(5.0)]) == []
