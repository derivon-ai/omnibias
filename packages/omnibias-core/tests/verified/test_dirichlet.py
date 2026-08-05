# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified Dirichlet-series enclosures on ``Re(s) > 1`` (verified substrate).

Rigor checks (every enclosure must *contain* the ``mpmath`` ground truth):

* ``zeta_enclosure`` brackets ``mpmath.zeta`` on a real / complex grid, and the
  analytic values ``zeta(2) = pi^2/6``, ``zeta(4) = pi^4/90``;
* the enclosure width shrinks as more terms are retained (tail control);
* ``l_function_enclosure`` brackets the Dirichlet beta ``beta(2) = Catalan`` and
  ``beta(3)``;
* ``p_series_tail_bound`` is a true upper bound on the omitted ``p``-series tail;
* ``theta_enclosure`` brackets ``mpmath.jtheta``;
* domain guards reject ``Re(s) <= 1`` and ``t <= 0``.
"""

from __future__ import annotations

import pytest

mp = pytest.importorskip("mpmath")

from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.dirichlet import (
    certified_dirichlet_series,
    complex_exp,
    dirichlet_beta_odd,
    l_function_enclosure,
    n_power_neg_s,
    p_series_tail_bound,
    theta_enclosure,
    zeta_enclosure,
    zeta_euler_maclaurin,
    zeta_even,
    zeta_negative_odd,
)
from omnibias.core.verified.interval import Interval

mp.mp.dps = 40


def _encloses(enc: ComplexInterval, value: complex) -> bool:
    return enc.re.contains(value.real) and enc.im.contains(value.imag)


class TestZeta:
    @pytest.mark.parametrize("s", [1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
    def test_real_axis_encloses_mpmath(self, s: float) -> None:
        enc = zeta_enclosure(s, num_terms=3000)
        assert _encloses(enc, complex(mp.zeta(s)))

    def test_analytic_zeta_2_and_4(self) -> None:
        assert zeta_enclosure(2.0, num_terms=5000).re.contains(float(mp.pi**2 / 6))
        assert zeta_enclosure(4.0, num_terms=3000).re.contains(float(mp.pi**4 / 90))

    @pytest.mark.parametrize("s", [complex(2.0, 1.0), complex(1.5, -3.0), complex(3.0, 0.5)])
    def test_complex_encloses_mpmath(self, s: complex) -> None:
        enc = zeta_enclosure(s, num_terms=4000)
        assert _encloses(enc, complex(mp.zeta(mp.mpc(s.real, s.imag))))

    def test_box_input_encloses_endpoints(self) -> None:
        # A genuine interval Re(s) in [2, 2.5] must enclose zeta at both ends.
        s_box = ComplexInterval.from_parts(Interval(2.0, 2.5), Interval.point(0.0))
        enc = zeta_enclosure(s_box, num_terms=3000)
        assert enc.re.contains(float(mp.zeta(2.0)))
        assert enc.re.contains(float(mp.zeta(2.5)))

    def test_width_shrinks_with_more_terms(self) -> None:
        w_small = zeta_enclosure(2.0, num_terms=100).re
        w_large = zeta_enclosure(2.0, num_terms=10000).re
        assert (w_large.hi - w_large.lo) < (w_small.hi - w_small.lo)

    def test_requires_re_gt_1(self) -> None:
        with pytest.raises(ValueError, match="Re"):
            zeta_enclosure(1.0)
        with pytest.raises(ValueError, match="Re"):
            zeta_enclosure(complex(0.5, 14.0))


class TestZetaSpecialValues:
    """W7: exact ``zeta(2m)`` / ``zeta(1-2m)`` / ``beta(2m+1)`` from Bernoulli / Euler."""

    @pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_zeta_even_encloses_mpmath(self, m: int) -> None:
        assert zeta_even(m).contains(float(mp.zeta(2 * m)))

    def test_zeta_even_known_closed_forms(self) -> None:
        assert zeta_even(1).contains(float(mp.pi**2 / 6))
        assert zeta_even(2).contains(float(mp.pi**4 / 90))
        assert zeta_even(3).contains(float(mp.pi**6 / 945))

    @pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_zeta_negative_odd_encloses_mpmath(self, m: int) -> None:
        assert zeta_negative_odd(m).contains(float(mp.zeta(1 - 2 * m)))

    def test_zeta_negative_odd_exact_rationals(self) -> None:
        # -B_2m/(2m): zeta(-1)=-1/12, zeta(-3)=1/120, zeta(-5)=-1/252.
        assert zeta_negative_odd(1).contains(-1.0 / 12.0)
        assert zeta_negative_odd(2).contains(1.0 / 120.0)
        assert zeta_negative_odd(3).contains(-1.0 / 252.0)

    def test_beta_1_is_pi_over_4(self) -> None:
        assert dirichlet_beta_odd(0).contains(float(mp.pi / 4))

    @pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_beta_odd_encloses_mpmath(self, m: int) -> None:
        # Dirichlet beta(2m+1) = sum_{k>=0} (-1)^k (2k+1)^{-(2m+1)}.
        ref = complex(mp.nsum(lambda k: (-1) ** k / (2 * k + 1) ** (2 * m + 1), [0, mp.inf]))
        assert dirichlet_beta_odd(m).contains(ref.real)

    def test_beta_3_is_pi_cubed_over_32(self) -> None:
        assert dirichlet_beta_odd(1).contains(float(mp.pi**3 / 32))

    def test_domain_guards(self) -> None:
        with pytest.raises(ValueError, match="m must be >= 1"):
            zeta_even(0)
        with pytest.raises(ValueError, match="m must be >= 1"):
            zeta_negative_odd(0)
        with pytest.raises(ValueError, match="m must be >= 0"):
            dirichlet_beta_odd(-1)


class TestZetaEulerMaclaurin:
    """W7: an *attempted* critical-strip enclosure via Euler-Maclaurin (numerical)."""

    # K = 8 points: on the critical line (incl. near the first zero), off it, and
    # a point with Re(s) > 1 (where it must agree with the majorant enclosure).
    STRIP_POINTS = [
        complex(0.5, 3.0),
        complex(0.5, 14.134725),  # near the first nontrivial zero
        complex(0.5, 21.022040),  # near the second nontrivial zero
        complex(0.25, 10.0),
        complex(0.75, 1.0),
        complex(0.1, 5.0),
        complex(0.9, -7.0),
        complex(2.0, 0.5),
    ]

    @pytest.mark.parametrize("s", STRIP_POINTS)
    def test_encloses_mpmath_in_the_strip(self, s: complex) -> None:
        enc = zeta_euler_maclaurin(s, num_sum_terms=25, order=6)
        ref = complex(mp.zeta(mp.mpc(s.real, s.imag)))
        assert _encloses(enc, ref)
        # a genuinely tight numerical enclosure, not a vacuous bound.
        assert enc.re.width < 1e-9 and enc.im.width < 1e-9

    def test_agrees_with_majorant_enclosure_above_the_wall(self) -> None:
        s = complex(3.0, 2.0)
        em = zeta_euler_maclaurin(s, num_sum_terms=25, order=6)
        wall = zeta_enclosure(s, num_terms=4000)
        ref = complex(mp.zeta(mp.mpc(3.0, 2.0)))
        assert _encloses(em, ref) and _encloses(wall, ref)

    def test_near_first_zero_is_small_but_no_rh_claim(self) -> None:
        # Near the first nontrivial zero the enclosed magnitude is tiny, but a small
        # |zeta| is NOT a proof about Re(s) -- RH stays an external obligation. The
        # sample 14.134725 is slightly off the true zero, so |zeta| ~ 1.1e-7 (not
        # identically 0); the enclosure resolves that value without any RH inference.
        s = complex(0.5, 14.134725)
        enc = zeta_euler_maclaurin(s, num_sum_terms=30, order=8)
        ref = complex(mp.zeta(mp.mpc(s.real, s.imag)))
        assert _encloses(enc, ref)
        assert enc.mag < 1e-6

    def test_pole_and_domain_guards(self) -> None:
        with pytest.raises(ValueError, match="pole at s = 1"):
            zeta_euler_maclaurin(complex(1.0, 0.0))
        with pytest.raises(ValueError, match="Re\\(s\\) >"):
            # Re(s) = -20 is past the -(2*order+1) = -13 continuation floor.
            zeta_euler_maclaurin(complex(-20.0, 1.0), order=6)
        with pytest.raises(ValueError, match="num_sum_terms"):
            zeta_euler_maclaurin(complex(0.5, 3.0), num_sum_terms=0)
        with pytest.raises(ValueError, match="order"):
            zeta_euler_maclaurin(complex(0.5, 3.0), order=0)

    def test_trivial_zero_at_minus_two(self) -> None:
        # The negative even integers are genuine (trivial) zeros; the EM enclosure
        # continued to Re(s) < 0 must still bracket the exact value zeta(-2) = 0.
        enc = zeta_euler_maclaurin(complex(-2.0, 0.0), num_sum_terms=30, order=8)
        assert enc.re.contains(0.0) and enc.im.contains(0.0)


class TestNPowerNegS:
    def test_integer_base_two(self) -> None:
        z = n_power_neg_s(2, 2.0)
        assert z.re.contains(0.25) and z.im.contains(0.0)

    def test_one_is_identity(self) -> None:
        z = n_power_neg_s(1, complex(3.0, 7.0))
        assert z.re.contains(1.0) and z.im.contains(0.0)

    def test_modulus_is_n_pow_neg_sigma(self) -> None:
        # |n^{-s}| = n^{-Re(s)} regardless of Im(s)
        z = n_power_neg_s(5, complex(2.0, 4.0))
        assert z.modulus().contains(5.0**-2.0)

    def test_matches_mpmath(self) -> None:
        z = n_power_neg_s(7, complex(2.5, 1.5))
        ref = complex(mp.mpf(7) ** (-mp.mpc(2.5, 1.5)))
        assert _encloses(z, ref)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            n_power_neg_s(0, 2.0)


class TestPSeriesTail:
    @pytest.mark.parametrize("sigma", [1.5, 2.0, 3.0])
    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_is_upper_bound_on_true_tail(self, sigma: float, n: int) -> None:
        bound = p_series_tail_bound(n, sigma).hi
        true_tail = float(mp.nsum(lambda k: k ** (-sigma), [n + 1, mp.inf]))
        assert bound >= true_tail

    def test_shrinks_with_more_terms(self) -> None:
        b100 = p_series_tail_bound(100, 2.0).hi
        b1000 = p_series_tail_bound(1000, 2.0).hi
        assert b100 > b1000 > 0.0

    def test_requires_sigma_gt_1(self) -> None:
        with pytest.raises(ValueError, match="Re"):
            p_series_tail_bound(100, 1.0)


class TestLFunction:
    # Non-principal character mod 4: chi(1)=1, chi(3)=-1 -> Dirichlet beta.
    CHI4 = [0, 1, 0, -1]

    def test_beta_2_is_catalan(self) -> None:
        enc = l_function_enclosure(self.CHI4, 2.0, num_terms=5000)
        assert enc.re.contains(float(mp.catalan)) and enc.im.contains(0.0)

    def test_beta_3_matches_mpmath(self) -> None:
        ref = complex(mp.nsum(lambda k: (-1) ** k / (2 * k + 1) ** 3, [0, mp.inf]))
        enc = l_function_enclosure(self.CHI4, 3.0, num_terms=5000)
        assert _encloses(enc, ref)

    def test_principal_character_reduces_to_zeta_like(self) -> None:
        # All-ones "character" (period 1) is exactly the zeta series.
        enc = l_function_enclosure([1], 2.0, num_terms=3000)
        assert enc.re.contains(float(mp.zeta(2.0)))

    def test_complex_s(self) -> None:
        s = complex(2.0, 1.0)
        ref = complex(mp.nsum(lambda k: (-1) ** k / (2 * k + 1) ** mp.mpc(2, 1), [0, mp.inf]))
        enc = l_function_enclosure(self.CHI4, s, num_terms=6000)
        assert _encloses(enc, ref)

    def test_guards(self) -> None:
        with pytest.raises(ValueError, match="period"):
            l_function_enclosure([], 2.0)
        with pytest.raises(ValueError, match="Re"):
            l_function_enclosure(self.CHI4, 1.0)


class TestCertifiedDirichletSeries:
    def test_general_contract_encloses_zeta(self) -> None:
        # Build zeta(2) by hand: retained terms n^{-2} + p-series tail bound.
        n = 500
        terms = [n_power_neg_s(k, 2.0) for k in range(1, n + 1)]
        tail = p_series_tail_bound(n, 2.0).hi
        enc = certified_dirichlet_series(terms, tail)
        assert enc.re.contains(float(mp.zeta(2.0)))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="retained term"):
            certified_dirichlet_series([], 0.1)


class TestTheta:
    @pytest.mark.parametrize("u,t", [(0.0, 0.5), (1.2, 0.3), (2.5, 0.8), (-1.0, 1.0)])
    def test_matches_jtheta(self, u: float, t: float) -> None:
        enc = theta_enclosure(u, t, num_terms=40)
        ref = complex(mp.jtheta(3, u / 2.0, mp.e ** (-t)))
        assert enc.contains(ref.real)

    def test_positive_and_thin(self) -> None:
        enc = theta_enclosure(0.7, 0.5, num_terms=40)
        assert enc.lo > 0.0
        assert (enc.hi - enc.lo) < 1e-9

    def test_requires_positive_t(self) -> None:
        with pytest.raises(ValueError, match="t > 0"):
            theta_enclosure(0.0, 0.0)


class TestNumberTheoryBoundary:
    """Enforcement: the analytic slice never claims discrete-hardness or RH results.

    Fails if a factoring / discrete-log / primality / RH surface is silently added,
    or if the ``Re(s) > 1`` majorant boundary / RH-obligation note is deleted.
    """

    FORBIDDEN = {
        "factor",
        "factorize",
        "integer_factorization",
        "discrete_log",
        "discrete_logarithm",
        "pollard_rho",
        "rsa_break",
        "rsa_recover",
        "ecc_break",
        "is_prime",
        "primality_proof",
        "prove_prime",
        "riemann_hypothesis",
        "zeta_zero",
        "nontrivial_zero",
        "zero_counting",
        "analytic_continuation",
    }

    def test_no_hardness_surface_exported(self) -> None:
        import omnibias.core.verified.dirichlet as D

        assert set(D.__all__).isdisjoint(self.FORBIDDEN)

    def test_docstring_records_convergence_and_rh_boundary(self) -> None:
        import omnibias.core.verified.dirichlet as D

        doc = (D.__doc__ or "").lower()
        assert "re(s) > 1" in doc
        assert "riemann hypothesis" in doc
        assert "external proof obligation" in doc

    def test_continuation_past_critical_line_is_refused(self) -> None:
        # The engine must refuse to evaluate where its majorant is not proved,
        # i.e. it never silently "continues" past Re(s) = 1.
        with pytest.raises(ValueError, match="Re"):
            zeta_enclosure(complex(0.5, 14.134725))  # near the first zeta zero


class TestComplexExp:
    def test_real_exp(self) -> None:
        z = complex_exp(ComplexInterval.from_parts(1.0, 0.0))
        assert z.re.contains(float(mp.e)) and z.im.contains(0.0)

    def test_imag_unit_quarter_turn(self) -> None:
        # exp(i pi/2) = i, cross-checked at the *rounded* argument (cos of the
        # double nearest pi/2 is ~6.1e-17, not exactly 0 -- the enclosure must
        # bracket the true value at the argument it was given).
        arg = float(mp.pi / 2)
        z = complex_exp(ComplexInterval.from_parts(0.0, arg))
        ref = complex(mp.e ** (mp.mpc(0, arg)))
        assert _encloses(z, ref)
        assert z.im.contains(1.0)
