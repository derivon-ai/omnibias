# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for discrete equation / recurrence discovery (the ``delta = 1`` twin).

Covers the exact-rational P-recursive finder, its least-squares baseline (whose
documented failure on Catalan is the refinement finding this locks in), the
closed-form Faulhaber polynomial extraction, and the honest ``None`` for
non-holonomic sequences (Bell / partition).
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb, factorial

import numpy as np
import pytest
from omnibias.symbolic.discrete import (
    DifferenceJets,
    RecurrenceRelation,
    build_difference_relation_library,
    discover_recurrence,
    discover_recurrence_least_squares,
    extract_difference_jets,
    polynomial_from_samples,
    verify_binomial_recurrence,
)


# --------------------------------------------------------------------------- #
# Canonical sequences                                                         #
# --------------------------------------------------------------------------- #
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


_PARTITION = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231]


# --------------------------------------------------------------------------- #
# extract_difference_jets                                                     #
# --------------------------------------------------------------------------- #
def test_difference_jets_rectangular_and_matches_forward_difference() -> None:
    samples = [1, 4, 9, 16, 25, 36, 49]  # squares (degree 2)
    jets = extract_difference_jets(samples, 3)
    assert isinstance(jets, DifferenceJets)
    assert jets.max_order == 3
    assert len(jets.index) == len(samples) - 3
    # column 0 is the raw sequence over the anchor range
    assert list(jets.columns[0]) == [Fraction(v) for v in samples[: len(jets.index)]]
    # squares: Delta^1 = 2n+1, Delta^2 = 2 (const), Delta^3 = 0
    assert all(v == 2 for v in jets.columns[2])
    assert all(v == 0 for v in jets.columns[3])


def test_difference_jets_degree_polynomial_vanishes() -> None:
    poly = [2 + 3 * n - n**2 + 4 * n**3 for n in range(9)]  # degree 3
    jets = extract_difference_jets(poly, 5)
    assert len({v for v in jets.columns[3]}) == 1  # Delta^3 constant (= 24)
    assert all(v == 0 for v in jets.columns[4])
    assert all(v == 0 for v in jets.columns[5])


def test_difference_jets_float_design_shape() -> None:
    jets = extract_difference_jets(list(range(10)), 2)
    design = jets.as_float_design()
    assert design.shape == (len(jets.index), 3)
    assert jets.column_name(0) == "a" and jets.column_name(2) == "D2a"


def test_difference_jets_validation() -> None:
    with pytest.raises(ValueError):
        extract_difference_jets([1, 2, 3], -1)
    with pytest.raises(ValueError):
        extract_difference_jets([1, 2, 3], 3)  # max_order >= len


# --------------------------------------------------------------------------- #
# discover_recurrence -- exact rational P-recursive finder                    #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# RecurrenceRelation                                                          #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Least-squares baseline (the contrasted, weaker path)                        #
# --------------------------------------------------------------------------- #
def test_least_squares_recovers_fibonacci() -> None:
    eq = discover_recurrence_least_squares(_fibonacci(15), order=2, index_degree=0)
    active = {row["name"]: row["coefficient"] for row in eq.active_terms()}
    assert active["a[n-1]"] == pytest.approx(1.0, abs=1e-6)
    assert active["a[n-2]"] == pytest.approx(1.0, abs=1e-6)


def test_least_squares_recovers_factorial_with_index_degree() -> None:
    eq = discover_recurrence_least_squares(
        [factorial(n) for n in range(11)], order=1, index_degree=1
    )
    active = {row["name"]: row["coefficient"] for row in eq.active_terms()}
    assert active["n^1*a[n-1]"] == pytest.approx(1.0, abs=1e-4)


def test_least_squares_fails_on_catalan_but_exact_finder_wins() -> None:
    # The refinement finding, locked in: a monic float fit cannot represent the
    # non-constant leading coefficient (n + 1) of the Catalan recurrence, while
    # the exact rational null-space finder recovers it with zero residual.
    catalan = _catalan(13)
    design, _names, target = build_difference_relation_library(catalan, order=1, index_degree=1)
    eq = discover_recurrence_least_squares(catalan, order=1, index_degree=1)
    ls_rel_resid = float(np.max(np.abs(eq.predict(design) - target)) / np.max(np.abs(target)))
    assert ls_rel_resid > 1e-6  # baseline is NOT exact

    rel = discover_recurrence(catalan)
    assert rel is not None and rel.max_abs_residual(catalan) == 0  # exact finder is


# --------------------------------------------------------------------------- #
# build_difference_relation_library                                           #
# --------------------------------------------------------------------------- #
def test_build_library_shapes_and_names() -> None:
    samples = list(range(2, 12))
    design, names, target = build_difference_relation_library(samples, order=2, index_degree=1)
    assert names == ["a[n-1]", "n^1*a[n-1]", "a[n-2]", "n^1*a[n-2]"]
    assert design.shape == (len(samples) - 2, 4)
    assert target.shape == (len(samples) - 2,)
    # first row: n = 2, a_1 = 3, a_0 = 2  ->  [3, 6, 2, 4], target a_2 = 4
    assert list(design[0]) == [3.0, 6.0, 2.0, 4.0]
    assert target[0] == 4.0


def test_build_library_validation() -> None:
    with pytest.raises(ValueError):
        build_difference_relation_library([1, 2, 3], order=0)
    with pytest.raises(ValueError):
        build_difference_relation_library([1, 2, 3], order=1, index_degree=-1)
    with pytest.raises(ValueError):
        build_difference_relation_library([1, 2], order=2)  # too few samples


# --------------------------------------------------------------------------- #
# polynomial_from_samples -- closed-form Faulhaber                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("power", "expected"),
    [
        (1, (Fraction(0), Fraction(1, 2), Fraction(1, 2))),
        (2, (Fraction(0), Fraction(1, 6), Fraction(1, 2), Fraction(1, 3))),
        (3, (Fraction(0), Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))),
    ],
)
def test_faulhaber_polynomial_exact(power: int, expected: tuple[Fraction, ...]) -> None:
    samples = [sum(k**power for k in range(1, i + 1)) for i in range(power + 6)]
    assert polynomial_from_samples(samples) == expected


def test_polynomial_from_samples_generic() -> None:
    poly = [2 - 3 * n + 5 * n**2 for n in range(6)]
    assert polynomial_from_samples(poly) == (Fraction(2), Fraction(-3), Fraction(5))


def test_polynomial_from_samples_constant_and_errors() -> None:
    assert polynomial_from_samples([7, 7, 7, 7]) == (Fraction(7),)
    with pytest.raises(ValueError):
        polynomial_from_samples([])


# --------------------------------------------------------------------------- #
# verify_binomial_recurrence -- Bell's full-history law                       #
# --------------------------------------------------------------------------- #
def test_bell_binomial_recurrence() -> None:
    assert verify_binomial_recurrence(_bell(12))
    # a sequence that does not satisfy B_{n+1} = sum_k C(n,k) B_k
    assert not verify_binomial_recurrence([1, 1, 2, 5, 15, 999])
    with pytest.raises(ValueError):
        verify_binomial_recurrence([1])
