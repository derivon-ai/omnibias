# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soundness and parity tests for the vectorized interval ``LDL^T`` path.

Two oracles guard :mod:`omnibias.core.verified.linalg_array`:

1. the scalar reference path in :mod:`omnibias.core.verified.eig_operator`,
   which the vectorized twin must reproduce endpoint for endpoint (both apply
   the same outward-rounded primitives in the same order), and
2. ``numpy.linalg.eigvalsh`` on point matrices drawn from each box -- a
   certified inertia must equal the true eigenvalue sign counts of every
   symmetric point matrix in the box.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.verified.eig_operator import (
    interval_ldlt_factor,
    interval_ldlt_inertia,
    interval_ldlt_pivots,
    is_positive_definite,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.interval_array import IntervalArray
from omnibias.core.verified.linalg_array import (
    interval_ldlt_factor_array,
    interval_ldlt_inertia_array,
    interval_ldlt_pivots_array,
    is_positive_definite_array,
)

#: Deterministic midpoints: definite, indefinite, and singular-pivot matrices.
DETERMINISTIC_MIDPOINTS = (
    pytest.param([[2.5]], id="one_by_one_positive"),
    pytest.param([[-0.75]], id="one_by_one_negative"),
    pytest.param(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], id="diagonal_positive"
    ),
    pytest.param(
        [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0]], id="diagonal_indefinite"
    ),
    pytest.param(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]], id="tridiagonal_laplacian"
    ),
    pytest.param([[-2.0, 0.5], [0.5, -3.0]], id="negative_definite"),
    pytest.param([[1.0, 2.0], [2.0, 1.0]], id="saddle"),
    pytest.param(
        [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.0, 0.25, 0.5],
            [0.5, 0.25, 2.0, 1.0],
            [0.0, 0.5, 1.0, 5.0],
        ],
        id="gram_matrix",
    ),
    pytest.param([[0.0, 1.0], [1.0, 0.0]], id="zero_first_pivot"),
    pytest.param([[1.0, 1.0], [1.0, 1.0]], id="rank_deficient"),
)

#: Box radii applied to every midpoint: exact points through a wide box.
RADII: tuple[float, ...] = (0.0, 1e-12, 1e-6, 1e-2)


def _scalar_matrix(box: IntervalArray) -> list[list[Interval]]:
    """The same box as scalar :class:`Interval` rows, for the reference path."""
    rows, columns = box.shape
    return [
        [Interval(float(box.lo[i, j]), float(box.hi[i, j])) for j in range(columns)]
        for i in range(rows)
    ]


def _symmetric_box(midpoint: np.ndarray, radius: float) -> IntervalArray:
    return IntervalArray(midpoint - radius, midpoint + radius)


def _symmetrize_lower(sample: np.ndarray) -> np.ndarray:
    """Mirror the lower triangle up, the half the factorization actually reads."""
    lower = np.tril(sample)
    return lower + lower.T - np.diag(np.diag(lower))


def _float_ldlt_pivots(matrix: np.ndarray) -> np.ndarray | None:
    """Plain float ``LDL^T`` pivots of a point matrix, or ``None`` if a pivot dies."""
    n = matrix.shape[0]
    lower = np.zeros((n, n), dtype=np.float64)
    pivots = np.zeros(n, dtype=np.float64)
    for j in range(n):
        lower[j, j] = 1.0
        value = matrix[j, j]
        for k in range(j):
            value -= lower[j, k] * lower[j, k] * pivots[k]
        if value == 0.0:
            return None
        pivots[j] = value
        for i in range(j + 1, n):
            entry = matrix[i, j]
            for k in range(j):
                entry -= lower[i, k] * lower[j, k] * pivots[k]
            lower[i, j] = entry / value
    return pivots


def _assert_matches_scalar_reference(box: IntervalArray) -> None:
    """The vectorized path must agree with the scalar path endpoint for endpoint."""
    scalar_rows = _scalar_matrix(box)
    scalar_inertia = interval_ldlt_inertia(scalar_rows)
    array_inertia = interval_ldlt_inertia_array(box)
    assert (scalar_inertia is None) == (array_inertia is None)
    assert is_positive_definite(scalar_rows) == is_positive_definite_array(box)
    assert (interval_ldlt_pivots(scalar_rows) is None) == (
        interval_ldlt_pivots_array(box) is None
    )
    if scalar_inertia is None or array_inertia is None:
        assert interval_ldlt_factor_array(box) is None
        return

    assert array_inertia == scalar_inertia
    assert array_inertia.negative + array_inertia.positive == box.shape[0]

    scalar_pivots = interval_ldlt_pivots(scalar_rows)
    array_pivots = interval_ldlt_pivots_array(box)
    assert scalar_pivots is not None and array_pivots is not None
    for index, pivot in enumerate(scalar_pivots):
        assert float(array_pivots.lo[index]) == pivot.lo
        assert float(array_pivots.hi[index]) == pivot.hi

    scalar_factor = interval_ldlt_factor(scalar_rows)
    array_factor = interval_ldlt_factor_array(box)
    assert scalar_factor is not None and array_factor is not None
    assert array_factor.size == box.shape[0]
    scalar_lower, _ = scalar_factor
    for i, row in enumerate(scalar_lower):
        for j, entry in enumerate(row):
            assert float(array_factor.lower.lo[i, j]) == entry.lo
            assert float(array_factor.lower.hi[i, j]) == entry.hi
    diagonal = np.diag(array_factor.lower.lo)
    assert np.all(diagonal == 1.0)
    assert np.all(np.triu(array_factor.lower.lo, 1) == 0.0)
    assert np.all(np.triu(array_factor.lower.hi, 1) == 0.0)


def _assert_matches_numpy_spectrum(box: IntervalArray, samples: list[np.ndarray]) -> None:
    """A certified inertia must equal the eigenvalue signs of every sampled point."""
    inertia = interval_ldlt_inertia_array(box)
    pivots = interval_ldlt_pivots_array(box)
    if inertia is None:
        assert pivots is None
        return
    assert pivots is not None
    certified_definite = is_positive_definite_array(box)
    for sample in samples:
        assert np.all(sample >= box.lo) and np.all(sample <= box.hi)
        eigenvalues = np.linalg.eigvalsh(sample)
        assert int(np.count_nonzero(eigenvalues < 0.0)) == inertia.negative
        assert int(np.count_nonzero(eigenvalues > 0.0)) == inertia.positive
        if certified_definite:
            assert float(eigenvalues.min()) > 0.0
        sample_pivots = _float_ldlt_pivots(sample)
        assert sample_pivots is not None
        assert np.all(pivots.contains(sample_pivots))


@pytest.mark.parametrize("midpoint", DETERMINISTIC_MIDPOINTS)
@pytest.mark.parametrize("radius", RADII)
def test_deterministic_boxes_match_scalar_reference_and_numpy(
    midpoint: list[list[float]], radius: float
) -> None:
    """Named matrices: parity with the scalar path plus a true-spectrum check."""
    center = np.asarray(midpoint, dtype=np.float64)
    box = _symmetric_box(center, radius)
    _assert_matches_scalar_reference(box)

    # A dense deterministic interior grid, symmetrized on the lower triangle.
    samples = [
        _symmetrize_lower(box.lo + fraction * (box.hi - box.lo))
        for fraction in np.linspace(0.0, 1.0, 9)
    ]
    _assert_matches_numpy_spectrum(box, [np.clip(s, box.lo, box.hi) for s in samples])


def test_random_symmetric_boxes_match_scalar_reference() -> None:
    """Random symmetric boxes, indefinite and definite, agree with the scalar path."""
    rng = np.random.default_rng(20260827)
    certified = 0
    for trial in range(160):
        n = int(rng.integers(1, 7))
        center = rng.normal(size=(n, n))
        center = 0.5 * (center + center.T)
        if trial % 3 == 0:  # a definite arm, so both branches are exercised
            center = center @ center.T + 0.5 * np.eye(n)
        box = _symmetric_box(center, float(rng.choice(RADII)))
        _assert_matches_scalar_reference(box)
        if interval_ldlt_inertia_array(box) is not None:
            certified += 1
    assert certified > 100, "expected most random boxes to certify an inertia"


def test_random_symmetric_boxes_match_numpy_eigenspectra() -> None:
    """Certified inertia equals the numpy eigenvalue signs of random box samples."""
    rng = np.random.default_rng(20260828)
    definite = 0
    indefinite = 0
    for trial in range(120):
        n = int(rng.integers(2, 7))
        center = rng.normal(size=(n, n))
        center = 0.5 * (center + center.T)
        if trial % 2 == 0:
            center = center @ center.T + 0.25 * np.eye(n)
        box = _symmetric_box(center, float(rng.choice((0.0, 1e-10, 1e-6))))
        inertia = interval_ldlt_inertia_array(box)
        if inertia is None:
            continue
        samples = [
            _symmetrize_lower(np.clip(rng.uniform(box.lo, box.hi), box.lo, box.hi))
            for _ in range(8)
        ]
        _assert_matches_numpy_spectrum(box, samples)
        if inertia.negative == 0:
            definite += 1
        else:
            indefinite += 1
    assert definite > 0 and indefinite > 0, "expected both definite and indefinite draws"


def test_straddling_pivot_is_reported_as_uncertified() -> None:
    """A box that contains a singular matrix cannot certify an inertia."""
    box = IntervalArray([[-0.5, 1.0], [1.0, 3.0]], [[0.5, 1.0], [1.0, 3.0]])
    assert interval_ldlt_inertia_array(box) is None
    assert interval_ldlt_pivots_array(box) is None
    assert interval_ldlt_factor_array(box) is None
    assert is_positive_definite_array(box) is False
    assert interval_ldlt_inertia(_scalar_matrix(box)) is None


def test_wide_box_around_a_definite_matrix_loses_definiteness() -> None:
    """Widening a certified box eventually returns an honest inconclusive."""
    center = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert is_positive_definite_array(_symmetric_box(center, 0.25)) is True
    assert is_positive_definite_array(_symmetric_box(center, 4.0)) is False


def test_domain_errors() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        interval_ldlt_inertia_array(IntervalArray.point([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-empty square"):
        interval_ldlt_inertia_array(IntervalArray.point([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="non-empty square"):
        interval_ldlt_pivots_array(IntervalArray.point(np.zeros((0, 0))))
    with pytest.raises(ValueError, match="two-dimensional"):
        interval_ldlt_factor_array(IntervalArray.point(np.zeros((2, 2, 2))))
