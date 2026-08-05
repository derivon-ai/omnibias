# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the backend-agnostic quadrature rules (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.fields._core.quadrature import (
    QuadratureSpec,
    gauss_hermite,
    gauss_legendre,
    monte_carlo,
    tensor_product,
)


def test_gauss_legendre_exact_for_polynomials() -> None:
    # n nodes integrate polynomials up to degree 2n-1 exactly.
    rule = gauss_legendre([(0.0, 1.0)], 4)
    x = rule.nodes[:, 0]
    for p in range(7):  # 2*4 - 1 = 7
        approx = float(rule.weights @ (x ** p))
        exact = 1.0 / (p + 1)
        assert np.isclose(approx, exact, rtol=1e-12, atol=1e-12), p


def test_gauss_legendre_2d_volume_and_polynomial() -> None:
    rule = gauss_legendre([(0.0, 2.0), (-1.0, 1.0)], (5, 5))
    assert rule.dim == 2
    # Integral of 1 over [0,2]x[-1,1] = 4.
    assert np.isclose(rule.weights.sum(), 4.0, rtol=1e-12)
    # Integral of x*y^2 = (int_0^2 x)(int_-1^1 y^2) = 2 * (2/3).
    f = rule.nodes[:, 0] * rule.nodes[:, 1] ** 2
    assert np.isclose(rule.weights @ f, 2.0 * (2.0 / 3.0), rtol=1e-12)


def test_gauss_hermite_is_normalised_expectation() -> None:
    rule = gauss_hermite(6, mean=0.5, scale=2.0)
    # Weights sum to 1 (it is an expectation rule).
    assert np.isclose(rule.weights.sum(), 1.0, rtol=1e-12)
    x = rule.nodes[:, 0]
    # E[x] = mean.
    assert np.isclose(rule.weights @ x, 0.5, rtol=1e-12)
    # E[x^2] = mean^2 + scale^2 = 0.25 + 4.
    assert np.isclose(rule.weights @ (x ** 2), 0.25 + 4.0, rtol=1e-12)


def test_monte_carlo_is_seeded_and_converges() -> None:
    r1 = monte_carlo([(0.0, 1.0)], 4096, seed=7)
    r2 = monte_carlo([(0.0, 1.0)], 4096, seed=7)
    assert np.array_equal(r1.nodes, r2.nodes)
    # Integral of x over [0,1] ~ 0.5 (loose MC tolerance).
    est = float(r1.weights @ r1.nodes[:, 0])
    assert np.isclose(est, 0.5, atol=2e-2)


def test_tensor_product_matches_direct_2d() -> None:
    rx = gauss_legendre([(0.0, 1.0)], 4)
    ry = gauss_legendre([(2.0, 3.0)], 4)
    tp = tensor_product(rx, ry)
    direct = gauss_legendre([(0.0, 1.0), (2.0, 3.0)], (4, 4))
    assert tp.n_nodes == direct.n_nodes
    # Both integrate the same polynomial identically.
    for rule in (tp, direct):
        f = rule.nodes[:, 0] ** 2 * rule.nodes[:, 1]
        val = float(rule.weights @ f)
        # int_0^1 x^2 dx * int_2^3 y dy = (1/3) * (5/2).
        assert np.isclose(val, (1.0 / 3.0) * 2.5, rtol=1e-12)


def test_quadrature_spec_validation() -> None:
    with pytest.raises(ValueError):
        QuadratureSpec("bad", np.zeros((3, 2)), np.zeros((4,)))
    with pytest.raises(ValueError):
        gauss_legendre([(1.0, 0.0)], 3)  # lo >= hi
