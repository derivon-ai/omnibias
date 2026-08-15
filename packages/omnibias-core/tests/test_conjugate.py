# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Conjugate Hilbert tower G1–G4 (theory 01-12)."""

from __future__ import annotations

import random

import pytest
from omnibias.core.conjugate import (
    HardyAtom,
    HardyDictionary,
    evaluate,
    hardy_p_deriv_n,
    hilbert,
)
from omnibias.core.verified.hardy_line import (
    hardy_even_deriv,
    hardy_even_deriv_n,
    hardy_even_deriv_n_iv,
    hardy_odd_deriv,
    hardy_odd_deriv_n,
    hardy_odd_deriv_n_iv,
    hilbert_of_hardy_even_deriv_n,
    pochhammer,
)
from omnibias.core.verified.interval import Interval


def test_g1_n1_interval_equal() -> None:
    y, a, alpha = 1.0, 1.0, 0.5
    assert hardy_even_deriv_n(y, a, alpha, 1) == hardy_even_deriv(y, a, alpha)
    assert hardy_odd_deriv_n(y, a, alpha, 1) == hardy_odd_deriv(y, a, alpha)


def test_g1_n1_interval_equal_iv() -> None:
    box = Interval(0.9, 1.1)
    from omnibias.core.verified.hardy_line import hardy_even_deriv_iv, hardy_odd_deriv_iv

    assert hardy_even_deriv_n_iv(box, 1.0, 0.5, 1) == hardy_even_deriv_iv(box, 1.0, 0.5)
    assert hardy_odd_deriv_n_iv(box, 1.0, 0.5, 1) == hardy_odd_deriv_iv(box, 1.0, 0.5)


def test_worked_example_n1_n2() -> None:
    y, a, alpha = 1.0, 1.0, 0.5
    p1 = hardy_even_deriv_n(y, a, alpha, 1)
    assert p1.mid == pytest.approx(-0.27467102, rel=1e-6)
    p2 = hardy_even_deriv_n(y, a, alpha, 2)
    q2 = hardy_odd_deriv_n(y, a, alpha, 2)
    assert p2.mid == pytest.approx(0.12067393, rel=1e-5)
    assert q2.mid == pytest.approx(-0.29133254, rel=1e-5)
    assert hilbert_of_hardy_even_deriv_n(y, a, alpha, 2) == q2


def test_g2_high_precision_mpmath() -> None:
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 80
    rng = random.Random(2)

    def mp_p(yy: object, aa: object, al: object) -> object:
        r = mpmath.sqrt(aa**2 + yy**2)
        phi = mpmath.atan(yy / aa)
        return (r ** (-al)) * mpmath.cos(al * phi)

    def mp_table(yy: object, aa: object, al: object, n: int) -> object:
        poch = mpmath.rf(al, n)
        beta = al + n
        r = mpmath.sqrt(aa**2 + yy**2)
        phi = mpmath.atan(yy / aa)
        p = (r ** (-beta)) * mpmath.cos(beta * phi)
        q = (r ** (-beta)) * mpmath.sin(beta * phi)
        return (poch * p, -poch * q, -poch * p, poch * q)[n % 4]

    worst = mpmath.mpf(0)
    for n in range(0, 9):
        a = mpmath.mpf("1.0")
        alpha = mpmath.mpf("0.5")
        y = mpmath.mpf("0.4")
        table = mp_table(y, a, alpha, n)
        if n == 0:
            ref = mp_p(y, a, alpha)
        else:
            ref = mpmath.diff(lambda t, aa=a, al=alpha: mp_p(t, aa, al), y, n)
        rel = abs(table - ref) / max(abs(ref), mpmath.mpf("1e-80"))
        if rel > worst:
            worst = rel
        # a couple of random (a, alpha, y) at modest order
        if n <= 4:
            aa = mpmath.mpf(0.5 + rng.random())
            al = mpmath.mpf(0.4 + rng.random())
            yy = mpmath.mpf(rng.uniform(-0.8, 0.8))
            table_r = mp_table(yy, aa, al, n)
            ref_r = mp_p(yy, aa, al) if n == 0 else mpmath.diff(
                lambda t, aaa=aa, all_=al: mp_p(t, aaa, all_), yy, n
            )
            rel_r = abs(table_r - ref_r) / max(abs(ref_r), mpmath.mpf("1e-80"))
            if rel_r > worst:
                worst = rel_r
    assert worst <= mpmath.mpf("1e-30")


def test_g3_enclosure_contains_truth() -> None:
    rng = random.Random(3)
    grid = [i * 0.1 for i in range(-20, 21)]
    samples = [rng.uniform(-2.0, 2.0) for _ in range(40)]
    for y in grid + samples:
        box = Interval(y - 1e-8, y + 1e-8)
        for n in range(0, 6):
            iv = hardy_even_deriv_n_iv(box, 1.0, 0.7, n)
            truth = hardy_p_deriv_n(y, 1.0, 0.7, n)
            assert iv.contains(truth)
            iv_q = hardy_odd_deriv_n_iv(box, 1.0, 0.7, n)
            from omnibias.core.conjugate import hardy_q_deriv_n

            assert iv_q.contains(hardy_q_deriv_n(y, 1.0, 0.7, n))


def test_g4_commutation_by_construction() -> None:
    y, a, alpha = 0.3, 1.2, 0.8
    for n in range(0, 6):
        hp = hilbert_of_hardy_even_deriv_n(y, a, alpha, n)
        qn = hardy_odd_deriv_n(y, a, alpha, n)
        assert hp == qn


def test_dictionary_signed_permutation() -> None:
    atoms = (
        HardyAtom(1.0, 0.5, 0, "even"),
        HardyAtom(1.0, 0.5, 0, "odd"),
        HardyAtom(1.0, 0.5, 1, "even"),
        HardyAtom(1.0, 0.5, 1, "odd"),
    )
    d = HardyDictionary(atoms)
    coeffs = (1.0, 0.2, -0.3, 0.4)
    h = hilbert(d, coeffs)
    # H[P]=Q (even -> odd, +), H[Q]=-P (odd -> even, -)
    assert h == pytest.approx((-0.2, 1.0, -0.4, -0.3))
    vals = evaluate(d, 0.4)
    assert len(vals) == 4
    assert pochhammer(0.5, 2) == pytest.approx(0.75)


def test_alpha_nonpositive_refused() -> None:
    d = HardyDictionary((HardyAtom(1.0, 0.0, 0, "even"), HardyAtom(1.0, 0.0, 0, "odd")))
    with pytest.raises(ValueError, match="alpha"):
        d.hilbert_permutation()
