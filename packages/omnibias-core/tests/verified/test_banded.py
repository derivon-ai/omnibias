# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soundness tests for the banded tail-inverse-norm bound."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.verified.banded import (
    banded_tail_inverse_bound,
    finite_band_row_sum_bound,
    geometric_band_row_sum_bound,
)
from omnibias.core.verified.interval import Interval


def test_diagonal_case_matches_radii_spectral_mu() -> None:
    """With zero off-diagonal mass, the bound collapses to 1/d_min exactly."""
    bound = banded_tail_inverse_bound(diag_lower=5.0, off_diagonal_row_sum_upper=0.0)
    assert isinstance(bound, Interval)
    assert bound.contains(1.0 / 5.0)
    assert bound.hi == pytest.approx(0.2, abs=1e-12)


def test_hand_worked_example() -> None:
    """d_min=4, s=1 => bound is exactly 1/3, matching the closed-form derivation."""
    bound = banded_tail_inverse_bound(diag_lower=4.0, off_diagonal_row_sum_upper=1.0)
    assert bound.contains(1.0 / 3.0)
    assert bound.lo <= 1.0 / 3.0 <= bound.hi
    assert bound.hi < 1.0 / 3.0 + 1e-9


def test_rejects_non_dominant_hypothesis() -> None:
    """off_diagonal_row_sum_upper >= diag_lower must refuse, never under-certify."""
    with pytest.raises(ValueError, match="diagonal dominance"):
        banded_tail_inverse_bound(diag_lower=2.0, off_diagonal_row_sum_upper=2.0)
    with pytest.raises(ValueError, match="diagonal dominance"):
        banded_tail_inverse_bound(diag_lower=2.0, off_diagonal_row_sum_upper=3.0)


def test_rejects_nonpositive_diag_lower() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        banded_tail_inverse_bound(diag_lower=0.0, off_diagonal_row_sum_upper=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        banded_tail_inverse_bound(diag_lower=-1.0, off_diagonal_row_sum_upper=0.0)


def test_rejects_negative_row_sum() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        banded_tail_inverse_bound(diag_lower=1.0, off_diagonal_row_sum_upper=-0.1)


def test_rejects_negative_bandwidth() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        banded_tail_inverse_bound(diag_lower=1.0, off_diagonal_row_sum_upper=0.1, bandwidth=-1)


def test_bandwidth_is_record_only() -> None:
    """bandwidth never changes the arithmetic; it is provenance only."""
    a = banded_tail_inverse_bound(diag_lower=4.0, off_diagonal_row_sum_upper=1.0)
    b = banded_tail_inverse_bound(
        diag_lower=4.0, off_diagonal_row_sum_upper=1.0, bandwidth=7
    )
    assert a == b


def _truncated_inf_norm_inverse(matrix: np.ndarray) -> float:
    """Dense oracle: ``||M^{-1}||_infty`` (max absolute row sum) for a finite matrix."""
    inv = np.linalg.inv(matrix)
    return float(np.max(np.sum(np.abs(inv), axis=1)))


def test_randomized_containment_against_dense_truncated_inverse() -> None:
    """The certified bound never falls below the true inf-norm of a truncated banded block.

    We build many random finite banded matrices satisfying row-wise diagonal
    dominance (``|M_ii| >= d_min``, ``sum_{j!=i} |M_ij| <= s < d_min``) at varying
    sizes and bandwidths, and check the certified ``1/(d_min - s)`` bound always
    dominates the true (dense) inf-norm of the finite block's inverse -- exactly
    the finite-dimensional restriction of what the infinite-tail claim asserts.
    """
    rng = np.random.default_rng(20260828)
    for _ in range(200):
        n = int(rng.integers(3, 12))
        bandwidth = int(rng.integers(1, min(4, n)))
        d_min = float(rng.uniform(2.0, 6.0))
        # Choose s comfortably below d_min so random per-row realizations
        # (each row sum independently <= s) still respect the hypothesis.
        s = float(rng.uniform(0.0, 0.9 * d_min))

        matrix = np.zeros((n, n))
        for i in range(n):
            sign = rng.choice([-1.0, 1.0])
            matrix[i, i] = sign * rng.uniform(d_min, d_min + 1.0)
            offsets = [d for d in range(-bandwidth, bandwidth + 1) if d != 0]
            weights = rng.uniform(0.0, 1.0, size=len(offsets))
            weights *= s / max(weights.sum(), 1e-12)  # row sum exactly <= s
            for offset, w in zip(offsets, weights, strict=True):
                j = i + offset
                if 0 <= j < n:
                    matrix[i, j] = rng.choice([-1.0, 1.0]) * w

        bound = banded_tail_inverse_bound(diag_lower=d_min, off_diagonal_row_sum_upper=s)
        true_norm = _truncated_inf_norm_inverse(matrix)
        assert true_norm <= bound.hi + 1e-9, (
            f"certified bound {bound.hi} violated by true inf-norm {true_norm} "
            f"(n={n}, bandwidth={bandwidth}, d_min={d_min}, s={s})"
        )


def test_finite_band_row_sum_bound_matches_hand_sum() -> None:
    total = finite_band_row_sum_bound([0.3, 0.2, 0.1])
    assert total == pytest.approx(0.6, abs=1e-12)
    assert finite_band_row_sum_bound([]) == 0.0


def test_finite_band_row_sum_bound_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        finite_band_row_sum_bound([0.1, -0.2])


def test_geometric_band_row_sum_bound_matches_closed_form() -> None:
    # sum_{d != 0} coeff * ratio^|d| = 2 * coeff * ratio / (1 - ratio)
    coeff, ratio = 0.5, 0.25
    expected = 2.0 * coeff * ratio / (1.0 - ratio)
    got = geometric_band_row_sum_bound(coeff, ratio)
    assert got == pytest.approx(expected, rel=1e-12)


def test_geometric_band_row_sum_bound_matches_truncated_series() -> None:
    """The closed form matches a directly summed long-but-finite truncation."""
    coeff, ratio = 0.7, 0.4
    truncated = sum(coeff * ratio**d for d in range(1, 200)) * 2.0
    got = geometric_band_row_sum_bound(coeff, ratio)
    assert got == pytest.approx(truncated, rel=1e-9)


def test_geometric_band_row_sum_bound_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError, match="ratio"):
        geometric_band_row_sum_bound(1.0, 1.0)
    with pytest.raises(ValueError, match="ratio"):
        geometric_band_row_sum_bound(1.0, -0.1)


def test_geometric_band_row_sum_bound_rejects_negative_coeff() -> None:
    with pytest.raises(ValueError, match="coeff"):
        geometric_band_row_sum_bound(-1.0, 0.5)


def test_end_to_end_with_finite_band_helper_and_dense_oracle() -> None:
    """Chain finite_band_row_sum_bound -> banded_tail_inverse_bound and verify vs a dense block."""
    d_min = 4.0
    band_bounds = [0.6, 0.6]  # two neighbours, e.g. a nearest-neighbour x.grad coupling
    s = finite_band_row_sum_bound(band_bounds)
    bound = banded_tail_inverse_bound(diag_lower=d_min, off_diagonal_row_sum_upper=s)

    n = 9
    matrix = np.diag([d_min] * n)
    for i in range(n):
        for offset in (-1, 1):
            j = i + offset
            if 0 <= j < n:
                matrix[i, j] = 0.6
    true_norm = _truncated_inf_norm_inverse(matrix)
    assert true_norm <= bound.hi + 1e-9
