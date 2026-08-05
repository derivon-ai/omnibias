# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""ANF / Reed-Muller (GF(2) Mobius) transform: round-trips and known forms."""

from __future__ import annotations

import random

from omnibias.boolean._core.anf import (
    algebraic_degree,
    anf_from_truth_table,
    anf_monomials,
    anf_to_string,
    truth_table_from_anf,
)
from omnibias.boolean._core.truth_table import truth_table_from_callable

AND = truth_table_from_callable(lambda a, b: a & b, 2)
OR = truth_table_from_callable(lambda a, b: a | b, 2)
XOR = truth_table_from_callable(lambda a, b: a ^ b, 2)


def test_anf_and_is_product() -> None:
    # AND -> x0*x1  (monomial mask 0b11 == index 3).
    assert anf_from_truth_table(AND) == (0, 0, 0, 1)
    assert anf_to_string(anf_from_truth_table(AND)) == "x0*x1"


def test_anf_xor_is_sum() -> None:
    assert anf_from_truth_table(XOR) == (0, 1, 1, 0)
    assert anf_to_string(anf_from_truth_table(XOR)) == "x0 + x1"


def test_anf_or() -> None:
    assert anf_from_truth_table(OR) == (0, 1, 1, 1)
    assert anf_to_string(anf_from_truth_table(OR)) == "x0 + x1 + x0*x1"


def test_algebraic_degree() -> None:
    assert algebraic_degree(anf_from_truth_table(AND)) == 2
    assert algebraic_degree(anf_from_truth_table(XOR)) == 1
    assert algebraic_degree((1, 0, 0, 0)) == 0  # constant 1
    assert algebraic_degree((0, 0, 0, 0)) == -1  # zero function


def test_monomials() -> None:
    mons = anf_monomials(anf_from_truth_table(OR))
    assert frozenset({0}) in mons
    assert frozenset({1}) in mons
    assert frozenset({0, 1}) in mons


def test_anf_is_involution() -> None:
    rng = random.Random(0)
    for n in (1, 2, 3, 4):
        for _ in range(20):
            tt = tuple(rng.randint(0, 1) for _ in range(1 << n))
            anf = anf_from_truth_table(tt)
            assert truth_table_from_anf(anf) == tt
            # Transform is its own inverse.
            assert anf_from_truth_table(anf) == tt
