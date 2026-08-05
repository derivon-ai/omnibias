# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Basic hypergeometric series: float baseline vs mpmath, and certified enclosures."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.qcalculus import (
    basic_hypergeometric,
    basic_hypergeometric_enclosure,
    q_exp,
    q_exp_enclosure,
)


def test_basic_hypergeometric_matches_mpmath() -> None:
    mp = pytest.importorskip("mpmath")
    cases = [
        ([0.2, 0.3], [0.4], 0.5, 0.1),
        ([0.1], [0.6], 0.5, 0.2),
        ([0.25, 0.5], [0.75], 0.3, 0.15),
    ]
    for a, b, q, z in cases:
        got = basic_hypergeometric(a, b, q, z, terms=80)
        ref = float(mp.qhyper([mp.mpf(x) for x in a], [mp.mpf(x) for x in b], mp.mpf(q), mp.mpf(z)))
        assert got == pytest.approx(ref, rel=1e-10, abs=1e-12)


def test_enclosure_contains_mpmath_oracle_across_seeds() -> None:
    mp = pytest.importorskip("mpmath")
    # 2phi1 over K = 8 (q, z) seeds: certified enclosure must contain the oracle.
    a = [Fraction(1, 5), Fraction(3, 10)]
    b = [Fraction(2, 5)]
    for seed in range(8):
        q = Fraction(3, 10) + Fraction(seed, 40)  # 0.30 .. 0.475
        z = Fraction(1, 20) + Fraction(seed, 100)  # 0.05 .. 0.12
        iv = basic_hypergeometric_enclosure(a, b, q, z, terms=40)
        ref = float(
            mp.qhyper(
                [mp.mpf(x.numerator) / x.denominator for x in a],
                [mp.mpf(x.numerator) / x.denominator for x in b],
                mp.mpf(q.numerator) / q.denominator,
                mp.mpf(z.numerator) / z.denominator,
            )
        )
        assert iv.lo <= ref <= iv.hi, f"seed {seed}: {ref} not in [{iv.lo}, {iv.hi}]"


def test_enclosure_contains_float_baseline() -> None:
    a = [Fraction(1, 4)]
    b = [Fraction(3, 5)]
    q, z = Fraction(1, 2), Fraction(1, 10)
    iv = basic_hypergeometric_enclosure(a, b, q, z, terms=40)
    base = basic_hypergeometric([0.25], [0.6], 0.5, 0.1, terms=200)
    assert iv.lo <= base <= iv.hi


def test_e_q_as_1phi0() -> None:
    # e_q(z) = _1phi_0(0; ; q, (1-q) z).
    for q in (0.3, 0.5, 0.7):
        for z in (0.2, 0.5, 1.0):
            viaphi = basic_hypergeometric([0.0], [], q, (1 - q) * z, terms=120)
            assert viaphi == pytest.approx(q_exp(z, q), rel=1e-9, abs=1e-12)


def test_q_exp_enclosure_sound_and_matches_baseline() -> None:
    for seed in range(8):
        q = Fraction(1, 4) + Fraction(seed, 40)  # 0.25 .. 0.425
        z = Fraction(1, 10) + Fraction(seed, 50)  # 0.1 .. 0.24
        iv = q_exp_enclosure(z, q, terms=48)
        base = q_exp(float(z), float(q), terms=400)
        assert iv.lo <= base <= iv.hi, f"seed {seed}: {base} not in [{iv.lo}, {iv.hi}]"


def test_enclosure_tightness_beats_single_term() -> None:
    # The certified width is far tighter than a trivial "first term +/- crude bound".
    a, b = [Fraction(1, 5)], [Fraction(1, 2)]
    q, z = Fraction(1, 3), Fraction(1, 10)
    iv = basic_hypergeometric_enclosure(a, b, q, z, terms=30)
    assert (iv.hi - iv.lo) < 1e-12


def test_divergent_regime_raises() -> None:
    # r > s + 1 (here r = 2, s = 0) is not enclosable: must refuse, not fake a bound.
    with pytest.raises(ValueError):
        basic_hypergeometric_enclosure([Fraction(1, 5), Fraction(1, 4)], [], Fraction(1, 2), Fraction(1, 10))


def test_uncertifiable_ratio_raises() -> None:
    # |z| too large for the retained-term count -> ratio not provably < 1.
    with pytest.raises(ValueError):
        q_exp_enclosure(Fraction(5), Fraction(1, 2), terms=2)
