# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multilinear extension and the discrete <-> continuous derivative bridge."""

from __future__ import annotations

import math
import random

from omnibias.boolean._core.anf import anf_from_truth_table
from omnibias.boolean._core.multilinear import (
    anf_from_multilinear_coeffs,
    mixed_partial,
    multilinear_coeffs,
    multilinear_eval,
    values_from_multilinear_coeffs,
)
from omnibias.boolean._core.truth_table import assignment, truth_table_from_callable

AND = truth_table_from_callable(lambda a, b: a & b, 2)
OR = truth_table_from_callable(lambda a, b: a | b, 2)
XOR = truth_table_from_callable(lambda a, b: a ^ b, 2)


def test_known_multilinear_coeffs() -> None:
    assert multilinear_coeffs(AND) == (0, 0, 0, 1)  # x0*x1
    assert multilinear_coeffs(OR) == (0, 1, 1, -1)  # x0 + x1 - x0*x1
    assert multilinear_coeffs(XOR) == (0, 1, 1, -2)  # x0 + x1 - 2*x0*x1


def test_extension_agrees_on_cube_corners() -> None:
    rng = random.Random(6)
    for n in (1, 2, 3, 4):
        for _ in range(10):
            tt = tuple(rng.randint(0, 1) for _ in range(1 << n))
            for idx, val in enumerate(tt):
                x = [float(b) for b in assignment(idx, n)]
                assert math.isclose(multilinear_eval(tt, x), val, abs_tol=1e-12)


def test_coeffs_round_trip() -> None:
    rng = random.Random(7)
    for n in (1, 2, 3, 4):
        for _ in range(10):
            tt = tuple(rng.randint(0, 1) for _ in range(1 << n))
            coeffs = multilinear_coeffs(tt)
            assert values_from_multilinear_coeffs(coeffs) == tt


def test_mobius_coeff_is_mixed_partial_at_zero() -> None:
    rng = random.Random(8)
    for n in (1, 2, 3):
        for _ in range(10):
            tt = tuple(rng.randint(0, 1) for _ in range(1 << n))
            coeffs = multilinear_coeffs(tt)
            zero = [0.0] * n
            for subset in range(1 << n):
                assert math.isclose(
                    mixed_partial(tt, subset, zero), float(coeffs[subset]), abs_tol=1e-9
                )


def test_anf_is_multilinear_mod_two() -> None:
    rng = random.Random(9)
    for n in (1, 2, 3, 4):
        for _ in range(20):
            tt = tuple(rng.randint(0, 1) for _ in range(1 << n))
            coeffs = multilinear_coeffs(tt)
            assert anf_from_multilinear_coeffs(coeffs) == anf_from_truth_table(tt)
