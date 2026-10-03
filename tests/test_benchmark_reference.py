# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Benchmark accuracy comes from independent mathematical references."""

import mpmath as mp
import numpy as np
import pytest
from _reference import derivative_reference, error_metrics, tanh_network_reference


def test_reference_recovers_polynomial_derivatives():
    points = np.array([-0.3, 0.0, 0.8])
    np.testing.assert_allclose(
        derivative_reference(lambda x: x**5, points, 3), 60 * points**2, atol=1e-15
    )
    with pytest.raises(ValueError, match="non-negative"):
        derivative_reference(mp.tanh, points, -1)


def test_network_oracle_accounts_for_affine_chain_rule():
    points = np.array([-0.3, 0.0, 0.8])
    layers = [(np.array([[2.0]]), np.array([0.1])), (np.array([[3.0]]), np.array([0.2]))]
    expected = -24 * np.tanh(2 * points + 0.1) / np.cosh(2 * points + 0.1) ** 2
    np.testing.assert_allclose(tanh_network_reference(layers, points, 2), expected, atol=2e-14)


def test_error_metric_detects_a_wrong_candidate_including_zero_reference():
    result = error_metrics(np.array([0.01, 1.1]), np.array([0.0, 1.0]))
    assert result["max_abs"] == pytest.approx(0.1)
    assert result["median_scaled_error"] == pytest.approx(5.05)
