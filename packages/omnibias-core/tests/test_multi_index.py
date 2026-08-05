# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-Python regression tests for multi-index combinatorics.

These run with no torch / jax / numpy dependency; they validate the bookkeeping
contracts (canonical ordering, Cauchy-product table, multi-index factorial)
consumed by the multivariate jet kernels in ``omnibias-jax`` / ``omnibias-torch``.
"""

from __future__ import annotations

from math import comb, factorial, prod

import pytest
from omnibias.core.multi_index import (
    index_position,
    multi_index_factorial,
    multi_indices,
    multiply_table,
    num_multi_indices,
)

# ----- multi_indices -----


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_count_matches_binomial(dim: int, order: int) -> None:
    idx = multi_indices(dim, order)
    assert len(idx) == comb(dim + order, dim)
    assert len(idx) == num_multi_indices(dim, order)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_all_distinct_and_within_order(dim: int, order: int) -> None:
    idx = multi_indices(dim, order)
    assert len(set(idx)) == len(idx)
    for alpha in idx:
        assert len(alpha) == dim
        assert all(a >= 0 for a in alpha)
        assert sum(alpha) <= order


def test_first_row_is_zero_index() -> None:
    assert multi_indices(3, 2)[0] == (0, 0, 0)


def test_gradient_block_is_unit_vectors() -> None:
    # After the zero index, the next ``dim`` entries are the unit vectors (the
    # gradient block), in canonical (degree, lexicographic) order.
    idx = multi_indices(3, 2)
    assert sorted(idx[1:4]) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    assert all(sum(a) == 1 for a in idx[1:4])


def test_sorted_by_total_degree() -> None:
    idx = multi_indices(3, 3)
    degrees = [sum(a) for a in idx]
    assert degrees == sorted(degrees)


# ----- index_position -----


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_index_position_is_inverse(dim: int, order: int) -> None:
    idx = multi_indices(dim, order)
    pos = index_position(dim, order)
    for i, alpha in enumerate(idx):
        assert pos[alpha] == i


# ----- multiply_table -----


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_multiply_table_consistency(dim: int, order: int) -> None:
    idx = multi_indices(dim, order)
    for g, a, b in multiply_table(dim, order):
        gamma, alpha, beta = idx[g], idx[a], idx[b]
        assert tuple(alpha[i] + beta[i] for i in range(dim)) == gamma


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_multiply_table_complete(dim: int, order: int) -> None:
    # Each gamma must receive exactly prod_i (gamma_i + 1) divisor pairs.
    idx = multi_indices(dim, order)
    counts: dict[int, int] = {}
    for g, _a, _b in multiply_table(dim, order):
        counts[g] = counts.get(g, 0) + 1
    for g, gamma in enumerate(idx):
        assert counts[g] == prod(gi + 1 for gi in gamma)


def test_multiply_table_reproduces_polynomial_product() -> None:
    # Multiply two explicit truncated polynomials via the table and compare to a
    # brute-force dict product, in 2 variables truncated at order 3.
    dim, order = 2, 3
    idx = multi_indices(dim, order)
    a = {alpha: float(i + 1) for i, alpha in enumerate(idx)}
    b = {alpha: float(2 * i + 3) for i, alpha in enumerate(idx)}
    a_vec = [a[alpha] for alpha in idx]
    b_vec = [b[alpha] for alpha in idx]

    out = [0.0] * len(idx)
    for g, ai, bi in multiply_table(dim, order):
        out[g] += a_vec[ai] * b_vec[bi]

    # Brute force, independent of the table.
    pos = index_position(dim, order)
    ref = [0.0] * len(idx)
    for alpha in idx:
        for beta in idx:
            gamma = tuple(alpha[i] + beta[i] for i in range(dim))
            if sum(gamma) <= order:
                ref[pos[gamma]] += a[alpha] * b[beta]

    assert out == pytest.approx(ref, rel=0, abs=0)


# ----- multi_index_factorial -----


def test_multi_index_factorial() -> None:
    assert multi_index_factorial((0, 0)) == 1
    assert multi_index_factorial((1, 0)) == 1
    assert multi_index_factorial((2, 3)) == factorial(2) * factorial(3)
    assert multi_index_factorial((4,)) == 24


def test_multi_index_factorial_rejects_negative() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        multi_index_factorial((1, -1))


# ----- error paths -----


def test_multi_indices_rejects_bad_dim() -> None:
    with pytest.raises(ValueError, match="dim must be >= 1"):
        multi_indices(0, 2)


def test_multi_indices_rejects_bad_order() -> None:
    with pytest.raises(ValueError, match="order must be >= 0"):
        multi_indices(2, -1)
