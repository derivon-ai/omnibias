# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified (interval) Walsh / multilinear spectra and certified bias bounds.

Soundness is checked two ways: the interval result must **contain** the
float-core result (parity), and it must contain an *independent exact-integer*
oracle computed straight from the truth table.
"""

from __future__ import annotations

import random

from omnibias.boolean._core.multilinear import multilinear_coeffs
from omnibias.boolean._core.truth_table import truth_table_from_callable
from omnibias.boolean._core.verified import (
    absolute_indicator_iv,
    autocorrelation_iv,
    differential_bias_iv,
    fourier_coeffs_iv,
    linear_bias_iv,
    linearity_iv,
    max_linear_bias_iv,
    mobius_iv,
    nonlinearity_iv,
    parseval_defect_iv,
    walsh_hadamard_iv,
    walsh_spectrum_iv,
)
from omnibias.boolean._core.walsh import (
    fourier_coeffs,
    walsh_hadamard,
)
from omnibias.core.verified.interval import Interval

XOR = truth_table_from_callable(lambda a, b: a ^ b, 2)
AND = truth_table_from_callable(lambda a, b: a & b, 2)
# 4-variable bent function f = x0 x1 xor x2 x3: |Walsh| == 4 everywhere.
BENT4 = truth_table_from_callable(lambda a, b, c, d: (a & b) ^ (c & d), 4)


# ---- exact integer oracles ------------------------------------------------- #
def _exact_wht(vals: list[int]) -> list[int]:
    a = list(vals)
    size = len(a)
    step = 1
    while step < size:
        for i in range(0, size, step << 1):
            for j in range(i, i + step):
                u, v = a[j], a[j + step]
                a[j] = u + v
                a[j + step] = u - v
        step <<= 1
    return a


def _parity(v: int) -> int:
    return bin(v).count("1") & 1


def _rand_table(rng: random.Random, n: int) -> tuple[int, ...]:
    return tuple(rng.randint(0, 1) for _ in range(1 << n))


# ---- transform parity (interval contains float core) ----------------------- #
def test_wht_iv_contains_float_core() -> None:
    rng = random.Random(0)
    for n in (1, 2, 3, 4):
        for _ in range(20):
            tt = _rand_table(rng, n)
            spin = [float(1 - 2 * v) for v in tt]
            iv = walsh_hadamard_iv(spin)
            fl = walsh_hadamard(spin)
            for a, b in zip(iv, fl, strict=True):
                assert a.contains(b)


def test_mobius_iv_contains_float_core() -> None:
    rng = random.Random(1)
    for n in (1, 2, 3, 4):
        for _ in range(20):
            vals = [rng.uniform(-2.0, 2.0) for _ in range(1 << n)]
            iv = mobius_iv(vals)
            fl = multilinear_coeffs(vals)
            for a, b in zip(iv, fl, strict=True):
                assert a.contains(float(b))


def test_fourier_coeffs_iv_contains_float_core() -> None:
    rng = random.Random(2)
    for n in (1, 2, 3, 4):
        for _ in range(20):
            tt = _rand_table(rng, n)
            iv = fourier_coeffs_iv(tt, encoding="pm1")
            fl = fourier_coeffs(tt, encoding="pm1")
            for a, b in zip(iv, fl, strict=True):
                assert a.contains(b)


# ---- certified figures of merit vs exact-integer oracle -------------------- #
def test_linearity_nonlinearity_oracle() -> None:
    rng = random.Random(3)
    for n in (2, 3, 4):
        for _ in range(30):
            tt = _rand_table(rng, n)
            spin = [1 - 2 * v for v in tt]
            wht = _exact_wht(spin)
            lin = max(abs(c) for c in wht)
            nl = (1 << (n - 1)) - lin // 2
            assert linearity_iv(tt).contains(float(lin))
            assert nonlinearity_iv(tt).contains(float(nl))


def test_linear_bias_oracle() -> None:
    rng = random.Random(4)
    for n in (2, 3, 4):
        for _ in range(20):
            tt = _rand_table(rng, n)
            for a in range(1 << n):
                agree = sum(1 for x in range(1 << n) if tt[x] == _parity(a & x))
                bias = agree / (1 << n) - 0.5
                assert linear_bias_iv(tt, a).contains(bias)


def test_autocorrelation_and_differential_bias_oracle() -> None:
    rng = random.Random(5)
    for n in (2, 3, 4):
        for _ in range(20):
            tt = _rand_table(rng, n)
            spin = [1 - 2 * v for v in tt]
            for a in range(1 << n):
                c = sum(spin[x] * spin[x ^ a] for x in range(1 << n))
                assert autocorrelation_iv(tt, a).contains(float(c))
                assert differential_bias_iv(tt, a).contains(c / (1 << (n + 1)))


def test_parseval_defect_contains_zero() -> None:
    rng = random.Random(6)
    for n in (1, 2, 3, 4):
        for _ in range(20):
            tt = _rand_table(rng, n)
            assert parseval_defect_iv(tt).contains(0.0)


# ---- known functions ------------------------------------------------------- #
def test_xor_linear_bias_is_extremal() -> None:
    # XOR equals chi_{0,1}, so the linear approximation by mask {0,1} = 0b11 is
    # exact: f(x) == <mask, x> for every x, giving the extremal bias +1/2.
    b = linear_bias_iv(XOR, 0b11)
    assert b.contains(0.5)
    # and the best linear bias over all nonzero masks is 1/2.
    assert max_linear_bias_iv(XOR).contains(0.5)


def test_bent_function_certificates() -> None:
    # Bent on n=4: linearity 4, nonlinearity 6, perfect (zero) autocorrelation.
    assert linearity_iv(BENT4).contains(4.0)
    assert nonlinearity_iv(BENT4).contains(6.0)
    ai = absolute_indicator_iv(BENT4)
    assert ai.contains(0.0)
    assert ai.hi < 1e-9  # near-zero up to defensive outward rounding
    # every nonzero linear bias has magnitude exactly 1/8 = 4 / 2^(4+1).
    assert max_linear_bias_iv(BENT4).contains(0.125)


# ---- verified spectrum of a differentiable surrogate ----------------------- #
def test_surrogate_interval_values_enclose_exact() -> None:
    # Model a tanh(beta x) gate whose outputs are only known to a tolerance:
    # each spin value is the interval [s - d, s + d].  The verified Fourier
    # coefficients must still enclose the exact Boolean spectrum.
    d = 1e-3
    spin_iv = [Interval(float(1 - 2 * v) - d, float(1 - 2 * v) + d) for v in AND]
    iv = fourier_coeffs_iv(values=spin_iv)
    exact = fourier_coeffs(AND, encoding="pm1")
    for a, b in zip(iv, exact, strict=True):
        assert a.contains(b)


def test_walsh_spectrum_iv_keys_match_float() -> None:
    spec_iv = walsh_spectrum_iv(AND, encoding="pm1")
    spec_fl = fourier_coeffs(AND, encoding="pm1")
    # mapping is keyed by variable subset; spot-check the empty set = mean.
    assert spec_iv[frozenset()].contains(spec_fl[0])
