# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soundness tests for vectorized outward-rounded interval arithmetic."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.interval_array import IntervalArray, sparse_matvec


def _scalar_at(values: IntervalArray, index: tuple[int, ...]) -> Interval:
    return Interval(float(values.lo[index]), float(values.hi[index]))


def _grid_sample(values: IntervalArray, fraction: float) -> np.ndarray:
    """Interpolate then clip, avoiding an endpoint roundoff escape."""
    sampled = values.lo + fraction * (values.hi - values.lo)
    return np.clip(sampled, values.lo, values.hi)


def _assert_encloses_scalar(
    result: IntervalArray,
    scalar_operation: Callable[[Interval, Interval], Interval],
    left: IntervalArray,
    right: IntervalArray,
) -> None:
    """Cross-check every vectorized component against scalar Interval arithmetic."""
    for index in np.ndindex(result.shape):
        scalar = scalar_operation(_scalar_at(left, index), _scalar_at(right, index))
        assert result.lo[index] <= scalar.lo <= scalar.hi <= result.hi[index]


def test_elementwise_operations_enclose_grid_and_random_samples() -> None:
    """All scalar operations remain sound in the vectorized representation."""
    left = IntervalArray(
        [[-1.5, 0.25, 2.0], [-0.8, -3.0, 0.4]],
        [[0.5, 1.75, 3.5], [1.2, -0.4, 2.2]],
    )
    right = IntervalArray(
        [[0.2, 1.1, 0.5], [0.7, 0.25, 1.5]],
        [[1.4, 2.3, 1.8], [1.9, 0.75, 3.0]],
    )
    positive = IntervalArray(
        [[0.0, 0.25, 2.0], [0.8, 3.0, 0.4]],
        [[0.5, 1.75, 3.5], [1.2, 4.0, 2.2]],
    )

    summed = left + right
    differenced = left - right
    multiplied = left * right
    divided = left / right
    squared = left.pow_int(2)
    cubed = left.pow_int(3)
    absolute = left.abs()
    rooted = positive.sqrt()
    reciprocal = right.reciprocal()
    _assert_encloses_scalar(summed, lambda x, y: x + y, left, right)
    _assert_encloses_scalar(differenced, lambda x, y: x - y, left, right)
    _assert_encloses_scalar(multiplied, lambda x, y: x * y, left, right)
    _assert_encloses_scalar(divided, lambda x, y: x / y, left, right)
    for index in np.ndindex(left.shape):
        scalar = _scalar_at(left, index)
        scalar_positive = _scalar_at(positive, index)
        scalar_right = _scalar_at(right, index)
        assert squared.lo[index] <= scalar.pow_int(2).lo <= scalar.pow_int(2).hi <= squared.hi[index]
        assert cubed.lo[index] <= scalar.pow_int(3).lo <= scalar.pow_int(3).hi <= cubed.hi[index]
        assert absolute.lo[index] <= scalar.abs().lo <= scalar.abs().hi <= absolute.hi[index]
        assert rooted.lo[index] <= scalar_positive.sqrt().lo <= scalar_positive.sqrt().hi <= rooted.hi[index]
        assert (
            reciprocal.lo[index]
            <= scalar_right.reciprocal().lo
            <= scalar_right.reciprocal().hi
            <= reciprocal.hi[index]
        )

    # Dense deterministic interior grid.
    for left_t in np.linspace(0.0, 1.0, 17):
        sample_left = _grid_sample(left, float(left_t))
        for right_t in np.linspace(0.0, 1.0, 17):
            sample_right = _grid_sample(right, float(right_t))
            assert np.all(summed.contains(sample_left + sample_right))
            assert np.all(differenced.contains(sample_left - sample_right))
            assert np.all(multiplied.contains(sample_left * sample_right))
            assert np.all(divided.contains(sample_left / sample_right))
            assert np.all(reciprocal.contains(1.0 / sample_right))
        assert np.all(squared.contains(sample_left**2))
        assert np.all(cubed.contains(sample_left**3))
        assert np.all(absolute.contains(np.abs(sample_left)))
        sample_positive = _grid_sample(positive, float(left_t))
        assert np.all(rooted.contains(np.sqrt(sample_positive)))

    # Independent random samples exercise distinct values in every array entry.
    rng = np.random.default_rng(20260827)
    for _ in range(128):
        sample_left = rng.uniform(left.lo, left.hi)
        sample_right = rng.uniform(right.lo, right.hi)
        sample_positive = rng.uniform(positive.lo, positive.hi)
        assert np.all(summed.contains(sample_left + sample_right))
        assert np.all(differenced.contains(sample_left - sample_right))
        assert np.all(multiplied.contains(sample_left * sample_right))
        assert np.all(divided.contains(sample_left / sample_right))
        assert np.all(squared.contains(sample_left**2))
        assert np.all(cubed.contains(sample_left**3))
        assert np.all(absolute.contains(np.abs(sample_left)))
        assert np.all(rooted.contains(np.sqrt(sample_positive)))
        assert np.all(reciprocal.contains(1.0 / sample_right))


def test_hull_properties_and_domain_checks() -> None:
    left = IntervalArray([-2.0, 1.0], [-1.0, 3.0])
    right = IntervalArray([-1.5, -4.0], [0.5, 2.0])
    combined = IntervalArray.hull(left, right, 0.0)
    assert np.all(combined.lo <= left.lo)
    assert np.all(left.hi <= combined.hi)
    assert np.all(combined.lo <= right.lo)
    assert np.all(right.hi <= combined.hi)
    assert np.all(combined.contains(0.0))
    assert np.all(combined.mid >= combined.lo)
    assert np.all(combined.mid <= combined.hi)
    assert np.all(combined.rad >= 0.0)
    assert np.all(combined.width >= 0.0)
    assert np.all(combined.mag >= combined.mig)
    with pytest.raises(ZeroDivisionError, match="zero outside"):
        IntervalArray([-1.0], [1.0]).reciprocal()
    with pytest.raises(ValueError, match="non-negative"):
        left.sqrt()


def test_dense_and_sparse_matvec_and_dot_are_sound() -> None:
    """Products use outward accumulation and agree componentwise with Interval."""
    matrix = IntervalArray(
        [[0.5, -1.3, 0.2], [1.1, -0.7, 0.4]],
        [[1.2, -0.3, 0.9], [1.8, 0.2, 1.1]],
    )
    vector = IntervalArray([-0.8, 0.5, 1.2], [0.6, 1.5, 2.0])
    dense = matrix.matvec(vector)
    rows = np.asarray([0, 0, 0, 1, 1, 1])
    columns = np.asarray([0, 1, 2, 0, 1, 2])
    entries = IntervalArray(matrix.lo[rows, columns], matrix.hi[rows, columns])
    sparse = sparse_matvec(rows, columns, entries, vector, matrix.shape)
    dot = vector.dot(vector)

    for row in range(matrix.shape[0]):
        scalar_sum = Interval.point(0.0)
        for column in range(matrix.shape[1]):
            scalar_sum = scalar_sum + _scalar_at(matrix, (row, column)) * _scalar_at(
                vector, (column,)
            )
        assert dense.lo[row] <= scalar_sum.lo <= scalar_sum.hi <= dense.hi[row]
        assert sparse.lo[row] <= scalar_sum.lo <= scalar_sum.hi <= sparse.hi[row]
    scalar_dot = Interval.point(0.0)
    for value in range(vector.shape[0]):
        scalar_dot = scalar_dot + _scalar_at(vector, (value,)) * _scalar_at(vector, (value,))
    assert dot.lo <= scalar_dot.lo <= scalar_dot.hi <= dot.hi

    # Dense deterministic samples and independent random samples.
    rng = np.random.default_rng(20260827)
    samples = [
        (
            _grid_sample(matrix, float(t)),
            _grid_sample(vector, float(1.0 - t)),
        )
        for t in np.linspace(0.0, 1.0, 17)
    ]
    samples.extend(
        (rng.uniform(matrix.lo, matrix.hi), rng.uniform(vector.lo, vector.hi))
        for _ in range(128)
    )
    for matrix_sample, vector_sample in samples:
        expected = matrix_sample @ vector_sample
        assert np.all(dense.contains(expected))
        assert np.all(sparse.contains(expected))
        assert dot.contains(float(vector_sample @ vector_sample))


def _entrywise_sparse_matvec(rows, columns, data, vector, shape):
    """The definition: one outward-rounded addition per COO entry, in entry order."""
    from omnibias.core.verified.interval_array import _mul_bounds

    lower, upper = np.zeros(shape[0]), np.zeros(shape[0])
    for row, column, lo, hi in zip(rows, columns, data.lo, data.hi, strict=True):
        plo, phi = _mul_bounds(
            np.asarray(lo, dtype=np.float64),
            np.asarray(hi, dtype=np.float64),
            np.asarray(vector.lo[column], dtype=np.float64),
            np.asarray(vector.hi[column], dtype=np.float64),
        )
        lower[row] = np.nextafter(lower[row] + plo.item(), -np.inf)
        upper[row] = np.nextafter(upper[row] + phi.item(), np.inf)
    return lower, upper


def test_sparse_matvec_is_bit_identical_to_the_entry_by_entry_accumulation() -> None:
    """Rounding after every addition makes the order of a row's entries part of the result; the vectorized
    form adds the k-th entry of every row at once and so performs the same IEEE operations per row."""
    rng = np.random.default_rng(7)
    for trial in range(60):
        n_rows, n_cols = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        nnz = int(rng.integers(0, 40))
        rows = rng.integers(0, n_rows, nnz)
        columns = rng.integers(0, n_cols, nnz)  # duplicates are separate entries
        lo = rng.standard_normal(nnz) * 10.0 ** rng.integers(-6, 6, nnz)
        hi = lo + np.abs(rng.standard_normal(nnz)) * 10.0 ** rng.integers(-9, 1, nnz)
        if trial % 5 == 0 and nnz:
            lo[0], hi[0] = -np.inf, np.inf
        v_lo = rng.standard_normal(n_cols)
        vector = IntervalArray(v_lo, v_lo + np.abs(rng.standard_normal(n_cols)) * 1e-3)
        data = IntervalArray(lo, hi)
        with np.errstate(invalid="ignore"):
            got = sparse_matvec(rows, columns, data, vector, (n_rows, n_cols))
            lower, upper = _entrywise_sparse_matvec(rows, columns, data, vector, (n_rows, n_cols))
        assert np.array_equal(got.lo.view(np.uint64), lower.view(np.uint64))
        assert np.array_equal(got.hi.view(np.uint64), upper.view(np.uint64))
