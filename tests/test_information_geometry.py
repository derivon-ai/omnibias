# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression tests for the 04-01 G2 Fisher-degeneracy falsifier."""

from __future__ import annotations

import numpy as np
import pytest
from information_geometry import (
    PREDICTED_PREFACTOR,
    fisher_delta_delta,
    fisher_delta_delta_mc,
    pack_density,
    pack_density_ddelta,
    pack_density_naive,
)
from omnibias.core.polynomials import sigmoid_polynomial_coeffs


def _quad_integral(delta: float, fn, *, nodes: int = 200) -> float:
    """Integrate ``fn(p, dp)`` against ``dx`` via ``t = sigma(x)``."""
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    t = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    u = (1.0 - t) / t
    p = pack_density(u, delta)
    dp = pack_density_ddelta(u, delta)
    return float(np.sum(w * fn(p, dp) / (t * (1.0 - t))))


@pytest.mark.parametrize("delta", [1e-3, 0.1, 1.0, 5.0])
def test_density_normalizes(delta: float) -> None:
    mass = _quad_integral(delta, lambda p, dp: p)
    assert abs(mass - 1.0) < 1e-12


@pytest.mark.parametrize("delta", [1.0, 0.1])
def test_stable_form_matches_definition(delta: float) -> None:
    xs = np.linspace(-4.0, 4.0, 401)
    u = np.exp(-xs)
    stable = pack_density(u, delta)
    naive = pack_density_naive(xs, delta)
    assert np.max(np.abs(stable - naive)) < 1e-12


def test_ddelta_matches_finite_difference_in_delta() -> None:
    delta = 0.5
    eps = 1e-6
    xs = np.linspace(-3.0, 3.0, 201)
    u = np.exp(-xs)
    analytic = pack_density_ddelta(u, delta)
    fd = (pack_density(u, delta + eps) - pack_density(u, delta - eps)) / (2.0 * eps)
    assert np.max(np.abs(analytic - fd)) < 1e-8


@pytest.mark.parametrize("delta", [1e-3, 0.1, 1.0])
def test_score_has_zero_mean(delta: float) -> None:
    mean_score = _quad_integral(delta, lambda p, dp: dp)
    assert abs(mean_score) < 1e-12


def test_leading_order_uses_core_coefficients() -> None:
    coeffs = sigmoid_polynomial_coeffs(3)
    assert coeffs == (0.0, 1.0, -7.0, 12.0, -6.0)
    delta = 1e-4
    for x in (-2.0, -0.5, 0.0, 1.3):
        t = 1.0 / (1.0 + np.exp(-x))
        sigma3 = sum(a * t**i for i, a in enumerate(coeffs))
        dp = float(pack_density_ddelta(np.exp(-x), delta))
        rel = abs((dp / delta) / (sigma3 / 12.0) - 1.0)
        assert rel < 1e-7, f"x={x}: rel={rel}"


def test_prefactor_is_one_over_720() -> None:
    delta = 1e-4
    g = fisher_delta_delta(delta, nodes=200)
    expected = PREDICTED_PREFACTOR * delta**2
    rel = abs(g - expected) / expected
    assert rel <= 1e-6


def test_quadrature_is_node_independent() -> None:
    delta = 1e-3
    g100 = fisher_delta_delta(delta, nodes=100)
    g400 = fisher_delta_delta(delta, nodes=400)
    assert abs(g100 - g400) / abs(g400) < 1e-12


def test_monte_carlo_agrees_with_closed_form() -> None:
    delta = 0.1
    g_closed = fisher_delta_delta(delta, nodes=200)
    g_mc, se = fisher_delta_delta_mc(delta, n=200_000, seed=0)
    assert abs(g_closed - g_mc) <= 3.0 * se
