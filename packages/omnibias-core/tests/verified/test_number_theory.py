# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""W5-ext rigorous number theory: Hurwitz zeta, polylogarithm, Lerch transcendent,
and exact Dirichlet ``L`` negative-integer values.

Every enclosure must contain the ``mpmath`` value (grid of ``K >= 8`` seeds); the
exact special values are rational (closed-form pi-multiples where relevant); GRH
stays an external obligation, never inferred.
"""

from __future__ import annotations

import importlib
from fractions import Fraction

import pytest
from omnibias.core.verified.dirichlet import dirichlet_l_negative_integer
from omnibias.core.verified.hurwitz import hurwitz_zeta, hurwitz_zeta_negative_integer
from omnibias.core.verified.polylog import lerch_transcendent, polylog_enclosure


def _mpmath():
    try:
        return importlib.import_module("mpmath")
    except ImportError:  # pragma: no cover
        return None


mp = _mpmath()
requires_mpmath = pytest.mark.skipif(mp is None, reason="mpmath not installed")


class TestHurwitzNegativeInteger:
    def test_zeta0_a_is_half_minus_a(self) -> None:
        # zeta(0, a) = 1/2 - a.
        for a in [Fraction(1), Fraction(1, 2), Fraction(3, 4)]:
            assert hurwitz_zeta_negative_integer(0, a).contains(float(Fraction(1, 2) - a))

    def test_reduces_to_riemann_zeta_at_a_one(self) -> None:
        from omnibias.core.verified.dirichlet import zeta_negative_odd

        # zeta(-1) = -1/12, zeta(-3) = 1/120 via a=1.
        assert hurwitz_zeta_negative_integer(1, 1).contains(-1.0 / 12.0)
        assert hurwitz_zeta_negative_integer(3, 1).contains(zeta_negative_odd(2).mid)

    @requires_mpmath
    def test_matches_mpmath(self) -> None:
        with mp.workdps(40):
            for n in range(0, 6):
                for a in [Fraction(1), Fraction(1, 2), Fraction(1, 3), Fraction(2, 5)]:
                    got = hurwitz_zeta_negative_integer(n, a)
                    ref = float(mp.zeta(-n, mp.mpf(a.numerator) / a.denominator))
                    assert got.contains(ref)

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            hurwitz_zeta_negative_integer(-1, 1)


class TestHurwitzZeta:
    @requires_mpmath
    def test_matches_mpmath_real_and_complex(self) -> None:
        seeds = [2.0, 1.5, 3.5, 0.5, -0.5, complex(2, 1), complex(0.5, 3), complex(-1.5, 2)]
        max_w = 0.0
        with mp.workdps(40):
            for s in seeds:
                for a in [1.0, 0.5, 2.3]:
                    enc = hurwitz_zeta(s, a, num_sum_terms=25, order=6)
                    sr = mp.mpc(getattr(s, "real", s), getattr(s, "imag", 0.0))
                    ref = complex(mp.zeta(sr, a))
                    assert enc.re.contains(ref.real) and enc.im.contains(ref.imag)
                    max_w = max(max_w, enc.re.width, enc.im.width)
        assert max_w < 1e-6

    def test_pole_and_domain_guards(self) -> None:
        with pytest.raises(ValueError):
            hurwitz_zeta(1.0, 1.0)  # pole at s=1
        with pytest.raises(ValueError):
            hurwitz_zeta(2.0, -1.0)  # a <= 0
        with pytest.raises(ValueError):
            hurwitz_zeta(-20.0, 1.0, order=6)  # Re(s) <= -(2*order+1)


class TestPolylog:
    @requires_mpmath
    def test_matches_mpmath(self) -> None:
        with mp.workdps(40):
            for s in [2.0, 3.0, 1.0, 0.5, -1.0, complex(2, 1)]:
                for z in [0.3, -0.5, 0.5j, complex(0.4, 0.3), -0.7]:
                    enc = polylog_enclosure(s, z, num_terms=140)
                    sr = mp.mpc(s.real, s.imag) if isinstance(s, complex) else mp.mpf(s)
                    zr = mp.mpc(getattr(z, "real", z), getattr(z, "imag", 0.0))
                    ref = complex(mp.polylog(sr, zr))
                    assert enc.re.contains(ref.real) and enc.im.contains(ref.imag)

    def test_li1_is_minus_log1_minus_z(self) -> None:
        import math

        enc = polylog_enclosure(1.0, 0.5, num_terms=200)
        assert enc.re.contains(-math.log(0.5))

    def test_domain_guard(self) -> None:
        with pytest.raises(ValueError):
            polylog_enclosure(2.0, 1.5)  # |z| >= 1


class TestLerch:
    @requires_mpmath
    def test_matches_mpmath(self) -> None:
        with mp.workdps(40):
            for z in [0.3, -0.5, complex(0.2, 0.2)]:
                for s in [2.0, 1.5, complex(2, 1)]:
                    for a in [1.0, 0.5, 2.0]:
                        enc = lerch_transcendent(z, s, a, num_terms=140)
                        zr = mp.mpc(getattr(z, "real", z), getattr(z, "imag", 0.0))
                        sr = mp.mpc(getattr(s, "real", s), getattr(s, "imag", 0.0))
                        ref = complex(mp.lerchphi(zr, sr, a))
                        assert enc.re.contains(ref.real) and enc.im.contains(ref.imag)

    def test_domain_guards(self) -> None:
        with pytest.raises(ValueError):
            lerch_transcendent(1.5, 2.0, 1.0)  # |z| >= 1
        with pytest.raises(ValueError):
            lerch_transcendent(0.3, 2.0, -1.0)  # a <= 0


class TestDirichletLNegativeInteger:
    def test_chi4_matches_euler_numbers(self) -> None:
        # chi_4 non-principal: L(0)=1/2, L(-2)=E_2/2=-1/2, L(-4)=E_4/2=5/2.
        chi4 = (0, 1, 0, -1)
        assert dirichlet_l_negative_integer(1, chi4).contains(0.5)
        assert dirichlet_l_negative_integer(3, chi4).contains(-0.5)
        assert dirichlet_l_negative_integer(5, chi4).contains(2.5)

    def test_principal_recovers_zeta_negatives(self) -> None:
        # trivial character mod 1 -> L(1-n) = zeta(1-n) for n >= 2.
        from omnibias.core.verified.dirichlet import zeta_negative_odd

        assert dirichlet_l_negative_integer(2, (1,)).contains(zeta_negative_odd(1).mid)

    def test_n_guard(self) -> None:
        with pytest.raises(ValueError):
            dirichlet_l_negative_integer(0, (0, 1, 0, -1))
