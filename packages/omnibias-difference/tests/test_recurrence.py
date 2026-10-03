# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite-prefix recurrence fits and independent held-out sequences."""
from __future__ import annotations

import random
from fractions import Fraction
from math import comb, factorial

import pytest
from omnibias.difference.recurrence import RecurrenceRelation, discover_recurrence

_PARTITION = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231]

def _fibonacci(n: int) -> list[int]:
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def _catalan(n: int) -> list[int]:
    return [comb(2 * k, k) // (k + 1) for k in range(n)]


def _bell(n: int) -> list[int]:
    row = [1]
    out = [1]
    for _ in range(n - 1):
        nxt = [row[-1]]
        for value in row:
            nxt.append(nxt[-1] + value)
        row = nxt
        out.append(row[0])
    return out


def test_discover_fibonacci_c_finite() -> None:
    rel = discover_recurrence(_fibonacci(15))
    assert rel is not None
    assert (rel.order, rel.index_degree) == (2, 0)
    # a_n - a_{n-1} - a_{n-2} = 0
    assert rel.coefficients == ((Fraction(1),), (Fraction(-1),), (Fraction(-1),))
    assert rel.max_abs_residual(_fibonacci(20)) == 0


def test_discover_catalan_p_recursive() -> None:
    rel = discover_recurrence(_catalan(13))
    assert rel is not None
    assert (rel.order, rel.index_degree) == (1, 1)
    # (n + 1) C_n - (4n - 2) C_{n-1} = 0  ->  p0 = 1 + n, p1 = 2 - 4n
    assert rel.coefficients == ((Fraction(1), Fraction(1)), (Fraction(2), Fraction(-4)))
    assert rel.max_abs_residual(_catalan(18)) == 0


def test_discover_factorial_p_recursive() -> None:
    rel = discover_recurrence([factorial(n) for n in range(11)])
    assert rel is not None
    assert (rel.order, rel.index_degree) == (1, 1)
    # a_n - n a_{n-1} = 0  ->  p0 = 1, p1 = -n
    assert rel.coefficients == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(-1)))


def test_discover_returns_none_for_non_holonomic() -> None:
    # Bell numbers and the partition function are not finitely P-recursive.
    assert discover_recurrence(_bell(17), max_order=4, max_index_degree=3) is None
    assert discover_recurrence(_PARTITION, max_order=4, max_index_degree=3) is None


def test_discover_random_c_finite_across_seeds() -> None:
    for seed in range(8):
        rng = random.Random(seed)
        p, q = rng.randint(1, 4), rng.randint(1, 4)
        seq = [rng.randint(0, 3), rng.randint(1, 4)]
        for _ in range(14):
            seq.append(p * seq[-1] + q * seq[-2])
        rel = discover_recurrence(seq)
        assert rel is not None
        assert rel.max_abs_residual(seq) == 0
        assert rel.order == 2 and rel.index_degree == 0


def test_discover_recurrence_validation() -> None:
    with pytest.raises(ValueError):
        discover_recurrence(_fibonacci(15), max_order=0)
    with pytest.raises(ValueError):
        discover_recurrence(_fibonacci(15), max_index_degree=-1)


def test_recurrence_relation_evaluation_and_pretty() -> None:
    # (n + 1) a_n + (2 - 4n) a_{n-1} = 0  (Catalan)
    rel = RecurrenceRelation(
        order=1,
        index_degree=1,
        coefficients=((Fraction(1), Fraction(1)), (Fraction(2), Fraction(-4))),
    )
    cat = _catalan(10)
    assert rel.coefficient_poly(0, 3) == 4  # 1 + 3
    assert rel.coefficient_poly(1, 3) == -10  # 2 - 12
    assert rel.is_satisfied_by(cat)
    assert rel.max_abs_residual(cat) == 0
    assert rel.pretty() == "(1 + n) a[n] + (2 - 4 n) a[n-1] = 0"
    assert rel.pretty(symbol="C").startswith("(1 + n) C[n]")


def test_recurrence_relation_detects_violation() -> None:
    rel = RecurrenceRelation(
        order=2, index_degree=0, coefficients=((Fraction(1),), (Fraction(-1),), (Fraction(-1),))
    )
    assert rel.is_satisfied_by(_fibonacci(12))
    assert not rel.is_satisfied_by([0, 1, 2, 4, 8, 16])  # geometric, not Fibonacci
