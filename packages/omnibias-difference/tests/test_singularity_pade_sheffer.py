# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""W7-ext -- singularity analysis / Pade / Sheffer-Riordan.

Acceptance-gate + regression tests for the transfer theorem
(:mod:`omnibias.difference._core.generating`), Pade / Thiele rational approximation
(:mod:`omnibias.difference._core.pade`), and the Sheffer / Riordan additions to
:mod:`omnibias.difference._core.umbral`:

* the transfer theorem's **exact** singular-template coefficient is verified against the
  binomial series and encloses the value; the leading asymptotic error decreases with
  ``n`` and the certified error bar is rigorous;
* Pade approximants match the series through order ``m + n`` (exact), beat raw truncation,
  reproduce rational functions exactly, and the certified remainder *contains* the true
  error over a disc;
* Thiele continued fractions interpolate exactly and detect unattainable points;
* Sheffer classification, the Riordan group law (product / inverse -> identity), and
  connection constants recover textbook results (Pascal, Stirling) exactly.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, cosh, exp, factorial, sqrt

import mpmath
import pytest
from omnibias.core.verified.interval import Interval
from omnibias.difference._core.generating import (
    catalan_asymptotic,
    dominant_pole_coefficient_asymptotic,
    rational_ogf_coefficients,
    singular_template_coefficient,
    transfer_theorem,
)
from omnibias.difference._core.pade import (
    pade_approximant,
    pade_certified_remainder,
    pade_evaluate,
    pade_evaluate_interval,
    rational_series,
    thiele_evaluate,
    thiele_interpolation,
)
from omnibias.difference._core.stirling import stirling_second
from omnibias.difference._core.umbral import (
    compose_series,
    compositional_inverse,
    connection_constants,
    riordan_array,
    riordan_inverse,
    riordan_product,
    series_reciprocal,
    sheffer_classify,
)


class TestSingularTemplateCoefficient:
    def test_geometric_pole(self) -> None:
        # [z^n] (1 - z)^{-1} = 1 for all n.
        assert all(singular_template_coefficient(1, n) == 1 for n in range(6))

    def test_square_root_branch(self) -> None:
        # (1 - z)^{1/2} = 1 - z/2 - z^2/8 - z^3/16 - ...
        expected = [Fraction(1), Fraction(-1, 2), Fraction(-1, 8), Fraction(-1, 16)]
        assert [singular_template_coefficient(Fraction(-1, 2), n) for n in range(4)] == expected

    @pytest.mark.parametrize("alpha_num,alpha_den", [(2, 1), (3, 1), (3, 2), (-1, 2), (5, 3)])
    def test_matches_binomial_series(self, alpha_num: int, alpha_den: int) -> None:
        alpha = Fraction(alpha_num, alpha_den)
        for n in range(7):
            # binom(n + alpha - 1, n) = prod_{j<n} (alpha + j)/(j+1).
            expected = Fraction(1)
            for j in range(n):
                expected *= (alpha + j) / Fraction(j + 1)
            assert singular_template_coefficient(alpha, n) == expected

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            singular_template_coefficient(1, -1)


class TestTransferTheorem:
    def test_simple_pole_exact(self) -> None:
        # f = 1/(1 - 2z) => [z^n] = 2^n; alpha = 1 so the asymptotic is exact.
        for n in (1, 5, 10):
            est = transfer_theorem(1, Fraction(1, 2), 1, n)
            assert est.exact_coefficient.contains(float(2**n))
            assert est.rel_error < 1e-12

    def test_catalan_leading_matches_classic_asymptotic(self) -> None:
        # Catalan singular part -2 (1 - 4z)^{1/2}; leading term == catalan_asymptotic.
        for n in (10, 20, 40):
            est = transfer_theorem(-2, Fraction(1, 4), Fraction(-1, 2), n)
            assert est.leading == pytest.approx(catalan_asymptotic(n), rel=1e-9)

    def test_exact_coefficient_encloses_template(self) -> None:
        # exact_coefficient must contain scale * radius^{-n} * template.
        scale, radius, alpha = Fraction(3), Fraction(1, 3), Fraction(3, 2)
        for n in range(1, 8):
            tmpl = singular_template_coefficient(alpha, n)
            true = float(scale) * float(radius) ** (-n) * float(tmpl)
            assert transfer_theorem(scale, radius, alpha, n).exact_coefficient.contains(true)

    def test_certified_error_bar_is_rigorous(self) -> None:
        # abs_error must bound |exact - leading| (exact enclosed, leading a point).
        est = transfer_theorem(-2, Fraction(1, 4), Fraction(-1, 2), 30)
        gap = max(
            abs(est.exact_coefficient.lo - est.leading),
            abs(est.exact_coefficient.hi - est.leading),
        )
        assert est.abs_error >= gap - 1e-6

    def test_non_positive_integer_exponent_raises(self) -> None:
        with pytest.raises(ValueError):
            transfer_theorem(1, Fraction(1, 2), 0, 5)
        with pytest.raises(ValueError):
            transfer_theorem(1, Fraction(1, 2), -1, 5)

    def test_n_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            transfer_theorem(1, Fraction(1, 2), 1, 0)


class TestDominantPole:
    def test_fibonacci(self) -> None:
        # F(z) = z/(1 - z - z^2); F_n ~ phi^n / sqrt(5). The subdominant pole leaves an
        # O(phi^{-2n}) relative error, so the asymptotic *converges* rather than matching
        # exactly at finite n -- assert that convergence and a tight large-n value.
        errs = []
        for n in (10, 20, 30):
            exact = int(rational_ogf_coefficients([0, 1], [1, -1, -1], n + 1)[n])
            approx = dominant_pole_coefficient_asymptotic([0, 1], [1, -1, -1], n)
            errs.append(abs(approx - exact) / exact)
        assert errs[0] > errs[1] > errs[2]  # monotone convergence
        assert errs[2] < 1e-9  # tight by n = 30

    def test_simple_geometric(self) -> None:
        # 1/(1 - 2z) has [z^n] = 2^n exactly (single real pole).
        assert dominant_pole_coefficient_asymptotic([1], [1, -2], 12) == pytest.approx(4096.0)

    def test_complex_dominant_pole_raises(self) -> None:
        # 1/(1 + z^2) has poles at +/- i (purely imaginary, oscillatory).
        with pytest.raises(ValueError):
            dominant_pole_coefficient_asymptotic([1], [1, 0, 1], 5)


class TestPadeApproximant:
    def test_matches_series_through_order(self) -> None:
        c = [Fraction(1, factorial(k)) for k in range(7)]  # exp
        p, q = pade_approximant(c, 3, 3)
        assert rational_series(p, q, 6) == tuple(c)

    def test_exp_2_2_textbook(self) -> None:
        c = [Fraction(1, factorial(k)) for k in range(5)]
        p, q = pade_approximant(c, 2, 2)
        assert p == (Fraction(1), Fraction(1, 2), Fraction(1, 12))
        assert q == (Fraction(1), Fraction(-1, 2), Fraction(1, 12))

    def test_recovers_rational_exactly(self) -> None:
        # 1/(1 - z/2) = sum (z/2)^k; its [1/1] Pade is exact.
        c = [Fraction(1, 2**k) for k in range(3)]
        p, q = pade_approximant(c, 1, 1)
        assert pade_evaluate(p, q, Fraction(1, 3)) == Fraction(1) / (1 - Fraction(1, 6))

    def test_beats_raw_truncation(self) -> None:
        c = [Fraction(1, factorial(k)) for k in range(7)]
        p, q = pade_approximant(c, 3, 3)
        x = 0.5
        pade_err = abs(exp(x) - float(pade_evaluate(p, q, Fraction(1, 2))))
        trunc_err = abs(exp(x) - sum(float(c[k]) * x**k for k in range(7)))
        assert pade_err < trunc_err

    def test_singular_system_raises(self) -> None:
        # A [2/2] Pade of the genuinely-[0/1] rational 1/(1-z/2) is degenerate.
        c = [Fraction(1, 2**k) for k in range(5)]
        with pytest.raises(ValueError):
            pade_approximant(c, 2, 2)

    def test_too_few_coeffs_raises(self) -> None:
        with pytest.raises(ValueError):
            pade_approximant([Fraction(1), Fraction(1)], 2, 2)


class TestPadeCertifiedRemainder:
    def test_exp_remainder_contains_true_error(self) -> None:
        c = [Fraction(1, factorial(k)) for k in range(13)]
        p, q = pade_approximant(c, 3, 3)
        ivs = [Interval.from_rational(Fraction(1, factorial(k))) for k in range(13)]
        r = 0.3
        rem = pade_certified_remainder(p, q, ivs, r, tail_bound=1.0, tail_ratio=0.5)
        true_err = abs(exp(r) - float(pade_evaluate(p, q, Fraction(3, 10))))
        assert true_err <= rem.hi
        assert rem.contains(0.0)

    def test_rational_remainder_near_zero(self) -> None:
        c = [Fraction(1, 2**k) for k in range(13)]
        p, q = pade_approximant(c, 1, 1)
        ivs = [Interval.from_rational(Fraction(1, 2**k)) for k in range(13)]
        rem = pade_certified_remainder(p, q, ivs, 0.4, tail_bound=1.0, tail_ratio=0.5)
        assert rem.contains(0.0)
        assert rem.mag < 1e-8

    def test_interval_eval_soundness(self) -> None:
        c = [Fraction(1, factorial(k)) for k in range(7)]
        p, q = pade_approximant(c, 3, 3)
        box = pade_evaluate_interval(p, q, Interval(0.2, 0.3))
        assert box.contains(float(pade_evaluate(p, q, Fraction(1, 4))))

    def test_divergent_tail_raises(self) -> None:
        c = [Fraction(1, factorial(k)) for k in range(13)]
        p, q = pade_approximant(c, 3, 3)
        ivs = [Interval.from_rational(Fraction(1, factorial(k))) for k in range(13)]
        with pytest.raises(ValueError):
            pade_certified_remainder(p, q, ivs, 3.0, tail_bound=1.0, tail_ratio=0.5)


class TestThiele:
    def test_interpolates_exactly(self) -> None:
        xs = [0, 1, 2, 3]
        ys = [1, 3, 2, 5]
        a = thiele_interpolation(xs, ys)
        assert all(thiele_evaluate(xs, a, x) == Fraction(y) for x, y in zip(xs, ys, strict=True))

    def test_reciprocal_function(self) -> None:
        # Attainable data from a generic smooth function.
        xs = [Fraction(1), Fraction(2), Fraction(4), Fraction(8)]
        ys = [x * x + 1 for x in xs]
        a = thiele_interpolation(xs, ys)
        assert thiele_evaluate(xs, a, Fraction(3)) == Fraction(10)

    def test_unattainable_point_raises(self) -> None:
        # f(x) = x/(x+1) is [1/1] rational; too many nodes -> unattainable.
        xs = [1, 2, 3, 4, 5]
        ys = [Fraction(x, x + 1) for x in xs]
        with pytest.raises(ZeroDivisionError):
            thiele_interpolation(xs, ys)

    def test_distinct_nodes_required(self) -> None:
        with pytest.raises(ValueError):
            thiele_interpolation([0, 1, 1], [1, 2, 3])


class TestSeriesOps:
    def test_composition(self) -> None:
        # (1 + t)^2 composed with t + t^2 -> matches direct expansion.
        out = compose_series([1, 2, 1], [0, 1, 1], 4)
        # (1 + (t+t^2))^2 = 1 + 2t + 3t^2 + 2t^3 + t^4.
        assert out == (Fraction(1), Fraction(2), Fraction(3), Fraction(2), Fraction(1))

    def test_compositional_inverse(self) -> None:
        # inverse of t/(1-t) is t/(1+t).
        inv = compositional_inverse([0, 1, 1, 1, 1, 1], 5)
        assert inv == (Fraction(0), Fraction(1), Fraction(-1), Fraction(1), Fraction(-1), Fraction(1))

    def test_inverse_round_trip(self) -> None:
        h = [0, 1, 2, 3, 1]
        hbar = compositional_inverse(h, 5)
        composed = compose_series(h, hbar, 5)
        assert composed[:2] == (Fraction(0), Fraction(1))
        assert all(c == 0 for c in composed[2:])

    def test_series_reciprocal(self) -> None:
        # 1/(1 - t) = 1 + t + t^2 + ...
        assert series_reciprocal([1, -1], 5) == tuple(Fraction(1) for _ in range(6))

    def test_reciprocal_zero_constant_raises(self) -> None:
        with pytest.raises(ValueError):
            series_reciprocal([0, 1], 3)


class TestSheffer:
    def test_appell(self) -> None:
        cls = sheffer_classify([1, 2, 3], [0, 1])
        assert cls.kind == "appell" and cls.is_appell

    def test_associated(self) -> None:
        cls = sheffer_classify([1], [0, 1, 1])
        assert cls.kind == "associated" and cls.is_associated

    def test_general_sheffer(self) -> None:
        cls = sheffer_classify([1, 1], [0, 1, 2])
        assert cls.kind == "sheffer" and not cls.is_appell and not cls.is_associated

    def test_invalid_g_raises(self) -> None:
        with pytest.raises(ValueError):
            sheffer_classify([0, 1], [0, 1])

    def test_invalid_f_raises(self) -> None:
        with pytest.raises(ValueError):
            sheffer_classify([1], [1, 1])  # f(0) != 0


class TestRiordan:
    def test_pascal_triangle(self) -> None:
        d = series_reciprocal([1, -1], 6)  # 1/(1-t)
        h = (Fraction(0), *series_reciprocal([1, -1], 5))  # t/(1-t)
        matrix = riordan_array(d, h, 6)
        for n in range(6):
            for k in range(6):
                assert matrix[n][k] == (Fraction(comb(n, k)) if k <= n else Fraction(0))

    def test_group_product_inverse_identity(self) -> None:
        d = [Fraction(1)] * 6  # 1/(1-t)
        h = [Fraction(0), *([Fraction(1)] * 5)]  # t/(1-t)
        d_inv, h_inv = riordan_inverse(d, h, 5)
        prod_d, prod_h = riordan_product((d, h), (d_inv, h_inv), 5)
        assert prod_d[0] == 1 and all(c == 0 for c in prod_d[1:])
        assert prod_h[:2] == (Fraction(0), Fraction(1)) and all(c == 0 for c in prod_h[2:])

    def test_improper_array_raises(self) -> None:
        with pytest.raises(ValueError):
            riordan_array([0, 1], [0, 1], 4)  # d(0) == 0


class TestConnectionConstants:
    def test_monomials_to_falling_is_stirling_second(self) -> None:
        # x^n = sum_k S(n, k) (x)_k.
        def falling(k: int) -> list[Fraction]:
            coeffs = [Fraction(1)]
            for j in range(k):
                nxt = [Fraction(0)] * (len(coeffs) + 1)
                for i, ci in enumerate(coeffs):
                    nxt[i] += -Fraction(j) * ci
                    nxt[i + 1] += ci
                coeffs = nxt
            return coeffs

        target = [falling(k) for k in range(6)]
        source = [[Fraction(0)] * n + [Fraction(1)] for n in range(6)]
        cc = connection_constants(source, target)
        for n in range(6):
            for k in range(n + 1):
                assert cc[n][k] == stirling_second(n, k)

    def test_non_graded_target_raises(self) -> None:
        # target[1] = [1, 0] has a zero degree-1 coefficient -> not a graded basis.
        with pytest.raises(ValueError):
            connection_constants([[Fraction(1), Fraction(1)]], [[Fraction(1)], [Fraction(1), Fraction(0)]])
