# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Walsh / Fourier spectrum, Parseval, and influence consistency."""

from __future__ import annotations

import math
import random

from omnibias.boolean._core.truth_table import truth_table_from_callable
from omnibias.boolean._core.walsh import (
    fourier_coeffs,
    fourier_influences,
    influences,
    parseval_defect,
    total_influence,
    walsh_spectrum,
)

XOR = truth_table_from_callable(lambda a, b: a ^ b, 2)
AND = truth_table_from_callable(lambda a, b: a & b, 2)
DICTATOR = truth_table_from_callable(lambda a, b, c: a, 3)


def test_xor_is_pure_high_frequency() -> None:
    # x0 ^ x1 in +/-1 output == chi_{0,1} exactly, so hat f({0,1}) = +1.
    spec = walsh_spectrum(XOR)
    assert math.isclose(spec[frozenset({0, 1})], 1.0, abs_tol=1e-12)
    for s, c in spec.items():
        if s != frozenset({0, 1}):
            assert abs(c) < 1e-12


def test_parseval() -> None:
    rng = random.Random(1)
    for n in (1, 2, 3, 4):
        for _ in range(15):
            tt = tuple(rng.randint(0, 1) for _ in range(1 << n))
            assert parseval_defect(tt) < 1e-9


def test_influence_definitions_agree() -> None:
    rng = random.Random(2)
    for n in (1, 2, 3, 4):
        for _ in range(15):
            tt = tuple(rng.randint(0, 1) for _ in range(1 << n))
            comb = influences(tt)
            four = fourier_influences(tt)
            assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(comb, four, strict=False))


def test_dictator_influences() -> None:
    inf = influences(DICTATOR)
    assert math.isclose(inf[0], 1.0, abs_tol=1e-12)
    assert math.isclose(inf[1], 0.0, abs_tol=1e-12)
    assert math.isclose(inf[2], 0.0, abs_tol=1e-12)


def test_and_total_influence() -> None:
    # AND on 2 bits: each input flips the output on 1 of 2 settings of the other.
    assert math.isclose(total_influence(AND), 1.0, abs_tol=1e-12)


def test_fourier_coeffs_indexing() -> None:
    coeffs = fourier_coeffs(AND, encoding="pm1")
    assert len(coeffs) == 4
    # E[f] for +/-1 AND = mean of (1,1,1,-1) = 0.5.
    assert math.isclose(coeffs[0], 0.5, abs_tol=1e-12)
