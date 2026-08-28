# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite-matrix non-self-adjoint contour-count soundness tests."""

from __future__ import annotations

import math

import numpy as np
from omnibias.core.verified import (
    characteristic_polynomial_enclosure,
    count_eigenvalues_in_contour,
)


def _block_inverse_norm(matrix: np.ndarray, z: complex) -> float:
    inverse = np.linalg.inv(z * np.eye(matrix.shape[0]) - matrix)
    block = np.block([[inverse.real, -inverse.imag], [inverse.imag, inverse.real]])
    return float(np.max(np.sum(np.abs(block), axis=1)))


def _circle_points(segments: int) -> list[complex]:
    return [
        complex(math.cos(2.0 * math.pi * index / segments), math.sin(2.0 * math.pi * index / segments))
        for index in range(segments)
    ]


def test_exact_triangular_oracle_counts_algebraic_multiplicity() -> None:
    # The spectrum is exactly the diagonal {1/4, 3/2, 2}; only 1/4 is inside.
    matrix = [[0.25, 0.2, -0.1], [0.0, 1.5, 0.15], [0.0, 0.0, 2.0]]
    certificate = count_eigenvalues_in_contour(matrix, radius=1.0)

    assert certificate.certified
    assert certificate.count == 1
    assert certificate.winding is not None and certificate.winding.contains(1.0)
    assert len(certificate.resolvent_inf_norm_bounds) == certificate.segments


def test_characteristic_polynomial_enclosure_contains_exact_oracle() -> None:
    # det(z I - A) = z² - 3z + 2 for this non-symmetric triangular matrix.
    coeffs = characteristic_polynomial_enclosure([[1.0, 5.0], [0.0, 2.0]])
    for coefficient, expected in zip(coeffs, (2.0, -3.0, 1.0), strict=True):
        assert coefficient.contains(expected)


def test_complex_pair_dense_and_random_resolvent_oracles() -> None:
    matrix = np.asarray([[0.1, -0.5], [0.5, 0.1]])
    certificate = count_eigenvalues_in_contour(matrix.tolist(), radius=0.8, segments=64)

    assert certificate.certified
    assert certificate.count == 2
    assert certificate.winding is not None and certificate.winding.contains(2.0)

    # Every deterministic perimeter point lies in one of the certified contour
    # boxes; use the adjacent segment's bound at its shared endpoint.
    for index, z in enumerate(_circle_points(certificate.segments)):
        exact = _block_inverse_norm(matrix, 0.8 * z)
        assert exact <= certificate.resolvent_inf_norm_bounds[index]

    rng = np.random.default_rng(20260827)
    for _ in range(128):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        index = min(int(theta / (2.0 * math.pi) * certificate.segments), certificate.segments - 1)
        exact = _block_inverse_norm(matrix, 0.8 * complex(math.cos(theta), math.sin(theta)))
        assert exact <= certificate.resolvent_inf_norm_bounds[index]


def test_boundary_eigenvalue_is_inconclusive() -> None:
    # The eigenvalue 1 lies on |z|=1, so a sound counter must refuse the count.
    certificate = count_eigenvalues_in_contour([[1.0, 2.0], [0.0, 3.0]], radius=1.0)
    assert not certificate.certified
    assert certificate.count is None
