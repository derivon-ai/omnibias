# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for verified quadrature, linear algebra, and root isolation."""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import (
    inf_norm_matrix,
    neumann_inverse_norm_bound,
    to_interval_matrix,
)
from omnibias.core.verified.quadrature import midpoint_integral, trapezoid_integral
from omnibias.core.verified.rootfind import (
    bisection_bracket,
    certified_sign_change,
    interval_newton,
)


# --------------------------------------------------------------------------- #
# Quadrature
# --------------------------------------------------------------------------- #
def test_trapezoid_encloses_known_integral() -> None:
    # f(x) = x^2 on [0, 1], exact integral = 1/3, f'' = 2.
    n = 8
    nodes = [Interval.point((i / n) ** 2) for i in range(n + 1)]
    enc = trapezoid_integral(nodes, 0.0, 1.0, Interval.point(2.0))
    assert enc.lo <= 1.0 / 3.0 <= enc.hi
    assert enc.width < 1e-2  # derived remainder is tight for a smooth function


def test_trapezoid_encloses_cubic() -> None:
    # f(x)=x^3 on [0,2], exact=4, f''=6x in [0,12].
    n = 16
    nodes = [Interval.point((2.0 * i / n) ** 3) for i in range(n + 1)]
    enc = trapezoid_integral(nodes, 0.0, 2.0, Interval(0.0, 12.0))
    assert enc.lo <= 4.0 <= enc.hi


def test_trapezoid_rejects_reversed_limits() -> None:
    nodes = [Interval.point(0.0), Interval.point(1.0)]
    with pytest.raises(ValueError, match="a <= b"):
        trapezoid_integral(nodes, 1.0, 0.0, Interval.point(0.0))


def test_midpoint_rejects_reversed_limits() -> None:
    mids = [Interval.point(0.25), Interval.point(0.75)]
    with pytest.raises(ValueError, match="a <= b"):
        midpoint_integral(mids, 1.0, 0.0, Interval.point(0.0))


def test_quadrature_allows_degenerate_equal_limits() -> None:
    # a == b is a zero-width interval -> integral 0; must not raise.
    nodes = [Interval.point(1.0), Interval.point(1.0)]
    enc = trapezoid_integral(nodes, 1.0, 1.0, Interval.point(2.0))
    assert enc.lo <= 0.0 <= enc.hi


# --------------------------------------------------------------------------- #
# Linear algebra / Neumann inverse-norm certificate
# --------------------------------------------------------------------------- #
def test_inf_norm_matrix_upper_bound() -> None:
    a = to_interval_matrix([[1.0, -2.0], [3.0, 0.5]])
    assert inf_norm_matrix(a) >= 3.5  # row sums 3, 3.5


def test_neumann_bound_certifies_well_conditioned() -> None:
    a = [[2.0, 0.0], [0.0, 4.0]]
    b = [[0.5, 0.0], [0.0, 0.25]]  # exact inverse
    rep = neumann_inverse_norm_bound(a, b)
    assert rep["certified"] is True
    assert rep["kappa"] < 1e-12
    # ||A^{-1}||_inf = 0.5 ; bound must be a valid (>=) upper bound.
    assert rep["inverse_norm_bound"] >= 0.5 - 1e-12


def test_neumann_bound_with_approximate_inverse() -> None:
    a = [[1.0, 0.1], [0.2, 1.0]]
    # slightly perturbed approximate inverse
    b = [[1.02, -0.1], [-0.205, 1.01]]
    rep = neumann_inverse_norm_bound(a, b)
    assert rep["certified"] is True
    assert 0.0 < rep["kappa"] < 1.0
    assert math.isfinite(rep["inverse_norm_bound"])


def test_neumann_fails_for_singular() -> None:
    a = [[1.0, 1.0], [1.0, 1.0]]  # singular
    b = [[1.0, 0.0], [0.0, 1.0]]
    rep = neumann_inverse_norm_bound(a, b)
    assert rep["certified"] is False
    assert rep["inverse_norm_bound"] == float("inf")


# --------------------------------------------------------------------------- #
# Root isolation
# --------------------------------------------------------------------------- #
def test_certified_sign_change() -> None:
    g = lambda x: x.pow_int(2) - Interval.point(2.0)  # noqa: E731
    assert certified_sign_change(g, 1.0, 2.0) is True
    assert certified_sign_change(g, 1.5, 1.6) is False  # both negative


def test_interval_newton_isolates_sqrt2() -> None:
    g = lambda x: x.pow_int(2) - Interval.point(2.0)  # noqa: E731
    gp = lambda x: Interval.point(2.0) * x  # noqa: E731
    res = interval_newton(g, gp, (1.0, 2.0))
    assert res["status"] == "unique_root"
    assert res["unique"] is True
    lo, hi = res["enclosure"]
    assert lo <= math.sqrt(2.0) <= hi
    assert res["width"] < 1e-12


def test_bisection_then_newton() -> None:
    def g(x: Interval) -> Interval:
        return x.pow_int(3) - Interval.point(2.0) * x - Interval.point(5.0)

    def gp(x: Interval) -> Interval:
        return Interval.point(3.0) * x.pow_int(2) - Interval.point(2.0)

    br = bisection_bracket(g, 2.0, 3.0, iters=20)
    res = interval_newton(g, gp, br)
    assert res["status"] == "unique_root"
    lo, hi = res["enclosure"]
    # Wallis' root of x^3 - 2x - 5 ~ 2.0945514815423265
    assert lo <= 2.0945514815423265 <= hi
