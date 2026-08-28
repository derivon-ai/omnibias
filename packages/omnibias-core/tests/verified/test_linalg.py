# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soundness tests for verified interval linear solves."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.verified.eig_operator import interval_ldlt_factor
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import (
    interval_solve,
    interval_triangular_solve,
)


def _sample(
    lower: np.ndarray, upper: np.ndarray, fraction: float
) -> np.ndarray:
    """Grid sample clipped to its source interval after floating interpolation."""
    return np.clip(lower + fraction * (upper - lower), lower, upper)


def _assert_contains(solution: list[Interval], expected: np.ndarray) -> None:
    assert len(solution) == expected.size
    for enclosure, value in zip(solution, expected, strict=True):
        assert enclosure.contains(float(value))


def test_interval_solve_krawczyk_encloses_grid_and_random_systems() -> None:
    """A Krawczyk comparison enclosure contains every sampled point-system solve."""
    matrix_lo = np.asarray([[3.8, -0.4], [0.1, 2.7]])
    matrix_hi = np.asarray([[4.2, -0.2], [0.3, 3.1]])
    rhs_lo = np.asarray([-1.2, 0.5])
    rhs_hi = np.asarray([-0.8, 0.9])
    matrix = [
        [Interval(float(matrix_lo[i, j]), float(matrix_hi[i, j])) for j in range(2)]
        for i in range(2)
    ]
    rhs = [Interval(float(rhs_lo[i]), float(rhs_hi[i])) for i in range(2)]
    solution = interval_solve(matrix, rhs, max_iter=12)

    # Dense deterministic samples.
    for matrix_fraction in np.linspace(0.0, 1.0, 17):
        matrix_sample = _sample(matrix_lo, matrix_hi, float(matrix_fraction))
        for rhs_fraction in np.linspace(0.0, 1.0, 17):
            rhs_sample = _sample(rhs_lo, rhs_hi, float(rhs_fraction))
            _assert_contains(solution, np.linalg.solve(matrix_sample, rhs_sample))

    # Independent random samples vary every entry separately.
    rng = np.random.default_rng(20260827)
    for _ in range(128):
        matrix_sample = rng.uniform(matrix_lo, matrix_hi)
        rhs_sample = rng.uniform(rhs_lo, rhs_hi)
        _assert_contains(solution, np.linalg.solve(matrix_sample, rhs_sample))


def test_interval_solve_symmetric_ldlt_encloses_grid_and_random_systems() -> None:
    """The exposed LDLT factors and triangular route enclose an SPD matrix box."""
    diagonal_lo = np.asarray([3.8, 2.9])
    diagonal_hi = np.asarray([4.2, 3.3])
    off_lo, off_hi = -0.3, -0.1
    rhs_lo = np.asarray([-1.0, 0.4])
    rhs_hi = np.asarray([-0.6, 0.8])
    matrix = [
        [Interval(float(diagonal_lo[0]), float(diagonal_hi[0])), Interval(off_lo, off_hi)],
        [Interval(off_lo, off_hi), Interval(float(diagonal_lo[1]), float(diagonal_hi[1]))],
    ]
    rhs = [Interval(float(rhs_lo[i]), float(rhs_hi[i])) for i in range(2)]
    factor = interval_ldlt_factor(matrix)
    assert factor is not None
    lower, diagonal = factor
    assert all(lower[i][i] == Interval.point(1.0) for i in range(2))
    assert all(not pivot.contains_zero() for pivot in diagonal)
    solution = interval_solve(matrix, rhs, symmetric=True)

    for fraction in np.linspace(0.0, 1.0, 17):
        diagonal_sample = _sample(diagonal_lo, diagonal_hi, float(fraction))
        rhs_sample = _sample(rhs_lo, rhs_hi, float(1.0 - fraction))
        matrix_sample = np.asarray(
            [
                [diagonal_sample[0], off_lo + fraction * (off_hi - off_lo)],
                [off_lo + fraction * (off_hi - off_lo), diagonal_sample[1]],
            ]
        )
        _assert_contains(solution, np.linalg.solve(matrix_sample, rhs_sample))

    rng = np.random.default_rng(20260827)
    for _ in range(128):
        diagonal_sample = rng.uniform(diagonal_lo, diagonal_hi)
        off_diagonal = rng.uniform(off_lo, off_hi)
        matrix_sample = np.asarray(
            [[diagonal_sample[0], off_diagonal], [off_diagonal, diagonal_sample[1]]]
        )
        rhs_sample = rng.uniform(rhs_lo, rhs_hi)
        _assert_contains(solution, np.linalg.solve(matrix_sample, rhs_sample))


def test_interval_triangular_solve_and_inconclusive_inputs() -> None:
    lower = [
        [Interval.point(2.0), Interval.point(0.0)],
        [Interval.point(-1.0), Interval.point(3.0)],
    ]
    rhs = [Interval.point(4.0), Interval.point(5.0)]
    solution = interval_triangular_solve(lower, rhs, lower=True)
    assert solution[0].contains(2.0)
    assert solution[1].contains(7.0 / 3.0)

    upper = [
        [Interval.point(2.0), Interval.point(-1.0)],
        [Interval.point(0.0), Interval.point(3.0)],
    ]
    upper_solution = interval_triangular_solve(upper, rhs, lower=False)
    assert upper_solution[0].contains(17.0 / 6.0)
    assert upper_solution[1].contains(5.0 / 3.0)

    with pytest.raises(ValueError, match="singular|must be < 1"):
        interval_solve([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])
