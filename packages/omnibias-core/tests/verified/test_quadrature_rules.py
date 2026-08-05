# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""W3-ext certified quadrature: Simpson / Gauss-Legendre / Euler-Maclaurin /
Romberg-class / Clenshaw-Curtis rules with derived remainders, plus the
tanh-sinh numerical estimator.

Soundness discipline: every certified rule's enclosure must contain the exact
integral (deterministic integrands + a random-polynomial sweep), the ``mpmath``
oracle must agree, and Gauss must beat the fixed-node trapezoid baseline at an
equal node budget.
"""

from __future__ import annotations

import importlib
import math
import random
from fractions import Fraction

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.quadrature import (
    QuadEstimate,
    clenshaw_curtis_integral,
    euler_maclaurin_quadrature,
    gauss_legendre_integral,
    gauss_legendre_nodes,
    romberg_integral,
    simpson_integral,
    tanh_sinh_estimate,
    trapezoid_integral,
)
from omnibias.core.verified.transcend import cos_iv, exp_iv, sin_iv


def _mpmath():
    try:
        return importlib.import_module("mpmath")
    except ImportError:  # pragma: no cover - environment dependent
        return None


# --- integrands with easy derivative bounds --------------------------------- #
def _exp_iv(x: Interval) -> Interval:
    return exp_iv(x)


def _exp_deriv(k: int, x: Interval) -> Interval:  # every derivative of exp is exp
    return exp_iv(x)


def _sin_iv(x: Interval) -> Interval:
    return sin_iv(x)


def _sin_deriv(k: int, x: Interval) -> Interval:
    # d^k sin = sin, cos, -sin, -cos cycle; |.| <= 1, enclose tightly by cases.
    r = k % 4
    if r == 0:
        return sin_iv(x)
    if r == 1:
        return cos_iv(x)
    if r == 2:
        return -sin_iv(x)
    return -cos_iv(x)


class TestCertifiedRulesEncloseExact:
    def test_simpson_encloses_exp(self) -> None:
        enc = simpson_integral(_exp_iv, 0.0, 1.0, panels=8, fourth_deriv_bound=exp_iv(Interval(0.0, 1.0)))
        assert enc.contains(math.e - 1.0)

    def test_gauss_encloses_exp_and_is_tight(self) -> None:
        enc = gauss_legendre_integral(_exp_iv, 0.0, 1.0, n=5, deriv_2n_bound=exp_iv(Interval(0.0, 1.0)))
        assert enc.contains(math.e - 1.0)
        assert enc.width < 1e-10

    def test_gauss_is_exact_for_low_degree_polynomials(self) -> None:
        # n=3 Gauss integrates degree <= 5 exactly; use f(x)=x^5 on [0,1] -> 1/6.
        f = lambda x: x.pow_int(5)  # noqa: E731
        # f^(6) == 0, so the remainder vanishes and the rule is exact.
        enc = gauss_legendre_integral(f, 0.0, 1.0, n=3, deriv_2n_bound=Interval.point(0.0))
        assert enc.contains(1.0 / 6.0)
        assert enc.width < 1e-12

    def test_euler_maclaurin_and_romberg_enclose_sin(self) -> None:
        # int_0^pi sin = 2.
        exact = 2.0
        for fn in (euler_maclaurin_quadrature, romberg_integral):
            enc = fn(_sin_iv, _sin_deriv, 0.0, math.pi, panels=16, terms=3)
            assert enc.contains(exact), fn.__name__

    def test_clenshaw_curtis_encloses_sin(self) -> None:
        enc = clenshaw_curtis_integral(_sin_iv, 0.0, math.pi, n=8, deriv_np1_bound=Interval(-1.0, 1.0))
        assert enc.contains(2.0)

    @pytest.mark.skipif(_mpmath() is None, reason="mpmath not installed")
    def test_matches_mpmath_quad(self) -> None:
        mp = _mpmath()
        ref = float(mp.quad(lambda t: mp.e**t, [0, 1]))
        enc = gauss_legendre_integral(_exp_iv, 0.0, 1.0, n=6, deriv_2n_bound=exp_iv(Interval(0.0, 1.0)))
        assert enc.contains(ref)


class TestRandomPolynomialSoundness:
    """A random-polynomial sweep: the enclosure must contain the exact integral."""

    def test_gauss_random_polynomials(self) -> None:
        rng = random.Random(20260724)
        for _ in range(64):
            deg = rng.randint(0, 8)  # n=5 Gauss exact for deg <= 9
            coeffs = [Fraction(rng.randint(-5, 5)) for _ in range(deg + 1)]
            a, b = 0.0, 1.0

            def f_iv(x: Interval, cs: list[Fraction] = coeffs) -> Interval:
                acc = Interval.point(0.0)
                for k, c in enumerate(cs):
                    acc = acc + Interval.from_rational(c) * x.pow_int(k)
                return acc

            # exact integral of a polynomial over [0,1]
            exact = sum(Fraction(c, k + 1) for k, c in enumerate(coeffs))
            # f^(10) == 0 for deg <= 9 -> remainder vanishes
            enc = gauss_legendre_integral(f_iv, a, b, n=5, deriv_2n_bound=Interval.point(0.0))
            assert enc.lo <= float(exact) <= enc.hi


class TestBeatsBaseline:
    def test_gauss_beats_fixed_trapezoid_at_equal_budget(self) -> None:
        # equal node budget ~ 9 points: trapezoid with 8 panels vs Gauss n=5 (twice, 10) -- use n=5.
        trap_nodes = [exp_iv(Interval.point(k / 8.0)) for k in range(9)]
        trap = trapezoid_integral(trap_nodes, 0.0, 1.0, exp_iv(Interval(0.0, 1.0)))
        gauss = gauss_legendre_integral(_exp_iv, 0.0, 1.0, n=5, deriv_2n_bound=exp_iv(Interval(0.0, 1.0)))
        assert gauss.width < trap.width
        # many orders tighter
        assert gauss.width < trap.width / 1e6


class TestGaussNodes:
    def test_nodes_are_mapped_and_symmetric(self) -> None:
        nodes = gauss_legendre_nodes(4, -1.0, 1.0)
        assert len(nodes) == 4
        # symmetric about 0
        assert nodes[0].mid == pytest.approx(-nodes[-1].mid, abs=1e-12)

    def test_unknown_n_raises(self) -> None:
        with pytest.raises(ValueError):
            gauss_legendre_nodes(99, 0.0, 1.0)


class TestTanhSinhEstimator:
    def test_handles_endpoint_singularity(self) -> None:
        # int_0^1 1/sqrt(x) dx = 2 (an endpoint singularity that defeats derivative bounds).
        est = tanh_sinh_estimate(lambda x: 1.0 / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0, level=8)
        assert isinstance(est, QuadEstimate)
        assert est.label == "numerical"
        assert abs(est.value - 2.0) < 1e-5

    def test_smooth_estimate_is_accurate(self) -> None:
        est = tanh_sinh_estimate(math.exp, 0.0, 1.0, level=6)
        assert abs(est.value - (math.e - 1.0)) < 1e-12


class TestInputGuards:
    def test_simpson_needs_even_panels(self) -> None:
        with pytest.raises(ValueError):
            simpson_integral(_exp_iv, 0.0, 1.0, panels=7, fourth_deriv_bound=Interval.point(1.0))

    def test_reversed_limits_raise(self) -> None:
        with pytest.raises(ValueError):
            gauss_legendre_integral(_exp_iv, 1.0, 0.0, n=3, deriv_2n_bound=Interval.point(1.0))
