# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the numpy reference measure substrate."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.measure import (
    Measure,
    importance_expectation,
    layer_cake_integral,
    lebesgue_integral,
    simple_function_approx,
    superlevel_measure,
)
from omnibias.measure._core import measure as m


def test_lebesgue_box_integral_matches_analytic() -> None:
    # int_0^1 int_0^1 (x^2 + y^2) dx dy = 2/3
    mu = m.lebesgue([(0.0, 1.0), (0.0, 1.0)], 8)
    val = lebesgue_integral(lambda p: p[:, 0] ** 2 + p[:, 1] ** 2, mu)
    assert float(val) == pytest.approx(2.0 / 3.0, rel=1e-10)
    assert mu.total_mass == pytest.approx(1.0, rel=1e-12)  # box volume


def test_gaussian_measure_is_expectation() -> None:
    # E[x^2] = 1 under N(0, 1); E[1] = 1 (probability measure).
    mu = m.gaussian(16)
    assert mu.total_mass == pytest.approx(1.0, rel=1e-12)
    assert float(lebesgue_integral(lambda p: p[:, 0] ** 2, mu)) == pytest.approx(1.0, rel=1e-8)
    # E[x^4] = 3 for a standard normal.
    assert float(lebesgue_integral(lambda p: p[:, 0] ** 4, mu)) == pytest.approx(3.0, rel=1e-6)


def test_counting_measure_is_a_sum() -> None:
    mu = m.counting([0.0, 1.0, 2.0, 3.0])
    val = lebesgue_integral(lambda p: p[:, 0], mu)
    assert float(val) == pytest.approx(6.0)


def test_vector_integrand_returns_per_component() -> None:
    mu = m.lebesgue([(0.0, 1.0)], 8)
    out = lebesgue_integral(lambda p: np.stack([p[:, 0], p[:, 0] ** 2], axis=1), mu)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.5, rel=1e-10)
    assert out[1] == pytest.approx(1.0 / 3.0, rel=1e-10)


def test_pushforward_change_of_variables() -> None:
    # T(x) = 2x + 1 on [0,1] Lebesgue -> int_[1,3] g dy where the pushed mass is
    # kept: int (g o T) dmu == int g d(T# mu).
    mu = m.lebesgue([(0.0, 1.0)], 16)
    push = mu.pushforward(lambda p: 2.0 * p + 1.0)
    lhs = lebesgue_integral(lambda p: (2.0 * p[:, 0] + 1.0) ** 2, mu)
    rhs = lebesgue_integral(lambda p: p[:, 0] ** 2, push)
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-12)


def test_product_measure_factorizes() -> None:
    mu = m.gaussian(12)  # N(0,1) on R
    nu = m.lebesgue([(0.0, 1.0)], 8)
    prod = mu.product(nu)
    assert prod.dim == 2
    # int (x^2 * y) d(mu (x) nu) = E[x^2] * int_0^1 y dy = 1 * 0.5
    val = lebesgue_integral(lambda p: p[:, 0] ** 2 * p[:, 1], prod)
    assert float(val) == pytest.approx(0.5, rel=1e-7)


def test_reweight_is_radon_nikodym() -> None:
    mu = m.lebesgue([(0.0, 1.0)], 32)
    # dnu/dmu = 3 x^2 -> int_0^1 x d nu = int_0^1 x * 3x^2 dx = 3/4
    nu = mu.reweight(lambda p: 3.0 * p[:, 0] ** 2)
    assert float(lebesgue_integral(lambda p: p[:, 0], nu)) == pytest.approx(0.75, rel=1e-10)


def test_normalize_gives_probability_measure() -> None:
    mu = m.lebesgue([(0.0, 2.0)], 8)  # mass 2
    p = mu.normalize()
    assert p.total_mass == pytest.approx(1.0, rel=1e-12)


def test_self_normalized_importance_expectation() -> None:
    # Proposal q = N(0,1); target p ∝ exp(-(x-1)^2/2) = N(1,1). E_p[x] = 1.
    q = m.gaussian(40)

    def log_ratio(p: np.ndarray) -> np.ndarray:
        return -0.5 * (p[:, 0] - 1.0) ** 2 + 0.5 * p[:, 0] ** 2

    est = importance_expectation(lambda p: p[:, 0], q, log_ratio, self_normalized=True)
    assert float(est) == pytest.approx(1.0, abs=1e-3)


def test_layer_cake_matches_direct_integral_on_nonneg() -> None:
    mu = m.lebesgue([(0.0, 1.0)], 64)

    def f(p: np.ndarray) -> np.ndarray:  # strictly positive on [0,1]
        return p[:, 0] ** 2 + 0.25

    direct = float(lebesgue_integral(f, mu))
    cake = float(layer_cake_integral(f, mu, beta=400.0, num_t=1200, signed=False))
    assert cake == pytest.approx(direct, rel=2e-2)


def test_layer_cake_signed() -> None:
    mu = m.lebesgue([(-1.0, 1.0)], 64)
    cake = float(
        layer_cake_integral(lambda p: p[:, 0], mu, beta=400.0, num_t=1200, signed=True)
    )
    assert cake == pytest.approx(0.0, abs=2e-2)


def test_simple_function_approx_converges_from_below() -> None:
    mu = m.lebesgue([(0.0, 1.0)], 64)
    levels = np.linspace(0.0, 1.0, 200)
    res = simple_function_approx(lambda p: p[:, 0], mu, levels=levels, beta=600.0)
    assert float(res.integral) == pytest.approx(0.5, abs=1e-2)
    # telescoping invariant: band masses sum to G_0 = mu({f > level_0}).
    assert float(np.sum(res.band_masses)) == pytest.approx(
        float(res.superlevel_measures[0]), rel=1e-9
    )


def test_superlevel_measure_monotone_decreasing() -> None:
    mu = m.gaussian(32)
    levels = np.linspace(-2.0, 2.0, 10)
    g = superlevel_measure(lambda p: p[:, 0], mu, levels, beta=50.0)
    assert np.all(np.diff(g) <= 1e-9)  # G(t) is non-increasing in t


def test_measure_validation() -> None:
    with pytest.raises(ValueError):
        Measure(np.zeros((3, 2)), np.zeros((2,)))
    with pytest.raises(ValueError):
        m.lebesgue([(0.0, 1.0)], 4).reweight(lambda p: -np.ones(p.shape[0]))
