# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contract tests for :mod:`omnibias.curvature.information` (theory 04-01)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.curvature.information import (
    DEGENERACY_ABS_FLOOR,
    PINV_RCOND_RULE,
    NotADensityError,
    collapse_degeneracy,
    distinguishability_samples,
    finite_difference_family,
    fisher_delta_delta,
    fisher_distance,
    fisher_metric,
    fisher_metric_mc,
    fisher_pinv,
    honesty_payload,
    logistic_location_family,
    logistic_mixture_family,
    pinv_rcond,
    randomized_mixture_suite,
    two_bias_density_naive,
    two_bias_density_u,
    two_bias_family,
    two_bias_located_family,
    worked_example,
)


def test_two_bias_prefactor_matches_g2() -> None:
    delta = 1e-4
    g = fisher_delta_delta(delta, nodes=200)
    expected = (1.0 / 720.0) * delta * delta
    assert abs(g - expected) / expected <= 1e-6


def test_logistic_location_fisher_is_one_third() -> None:
    g = fisher_metric(logistic_location_family(), np.array([0.3]), nodes=96)
    assert float(g[0, 0]) == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_location_distance_is_euclidean_over_sqrt_i() -> None:
    family = logistic_location_family()
    dist = fisher_distance(family, np.array([0.0]), np.array([math.sqrt(3.0)]))
    assert dist == pytest.approx(1.0, rel=1e-12)


def test_distinguishability_wald_count() -> None:
    family = logistic_location_family()
    n = distinguishability_samples(family, np.array([0.0]), np.array([0.6]))
    assert n == 66


def test_k_ge_3_finite_difference_is_not_a_density() -> None:
    family = finite_difference_family(n_biases=3)
    with pytest.raises(NotADensityError, match="inapplicable"):
        fisher_metric(family, np.array([0.2]))


def test_non_sigmoid_activation_is_refused() -> None:
    with pytest.raises(NotADensityError, match="logistic"):
        from omnibias.curvature.information import PackFamily

        PackFamily(kind="logistic_location", activation="tanh")


def test_metric_is_symmetric_pspd() -> None:
    family = logistic_mixture_family()
    for theta in randomized_mixture_suite(n=4, seed=1):
        g = fisher_metric(family, theta, nodes=64)
        assert np.max(np.abs(g - g.T)) <= 1e-14
        eig = np.linalg.eigvalsh(0.5 * (g + g.T))
        assert float(np.min(eig)) >= 1e-3


def test_two_bias_near_collapse_is_degenerate() -> None:
    report = collapse_degeneracy(two_bias_family(), np.array([1e-4]))
    assert report.exponent == pytest.approx(2.0, abs=0.05)
    assert report.eigenvalue < 1e-9
    assert report.recommendation == "reparameterize by order"
    assert report.damping == DEGENERACY_ABS_FLOOR


def test_mixture_mc_agrees_with_quadrature() -> None:
    family = logistic_mixture_family()
    theta = randomized_mixture_suite(n=1, seed=2)[0]
    closed = fisher_metric(family, theta, nodes=64)
    mc, se = fisher_metric_mc(family, theta, n=40_000, seed=0)
    sig = np.abs(closed - mc) / np.maximum(se, 1e-18)
    assert float(np.max(sig)) <= 3.0


def test_pinv_uses_numpy_rcond_rule() -> None:
    g = np.diag([1.0, 1e-20])
    rc = pinv_rcond(g)
    assert rc == pytest.approx(2.0 * np.finfo(g.dtype).eps)
    pin = fisher_pinv(g)
    assert pin[0, 0] == pytest.approx(1.0)
    assert pin[1, 1] == pytest.approx(0.0)
    assert "numpy.linalg.pinv" in PINV_RCOND_RULE


def test_stable_two_bias_matches_definition() -> None:
    xs = np.linspace(-3.0, 3.0, 81)
    u = np.exp(-xs)
    assert np.max(np.abs(two_bias_density_u(u, 0.4) - two_bias_density_naive(xs, 0.4))) < 1e-12


def test_located_pack_translation() -> None:
    family = two_bias_located_family()
    g0 = fisher_metric(family, np.array([0.0, 0.8]), nodes=96)
    g1 = fisher_metric(family, np.array([1.3, 0.8]), nodes=96)
    assert g0 == pytest.approx(g1, rel=1e-9, abs=1e-12)


def test_worked_example_and_honesty() -> None:
    ex = worked_example()
    assert abs(float(ex["g_over_delta2"]) - 1.0 / 720.0) / (1.0 / 720.0) < 1e-6
    assert ex["recommendation"] == "reparameterize by order"
    payload = honesty_payload()
    assert payload["bias_collapse"] is True
    assert payload["temperature_collapse"] is False
    assert payload["k_ge_3_fisher"] == "inapplicable_not_a_density"
    assert payload["theorem_prover_verified"] is False


def test_as_manifold_spec_optional() -> None:
    pytest.importorskip("omnibias.geometry")
    from omnibias.curvature.information import as_manifold_spec

    spec = as_manifold_spec(logistic_location_family())
    assert spec.dim == 1
    assert spec.name.startswith("pack_fisher_")
