# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for controlled omnibias feature discovery."""

from __future__ import annotations

import numpy as np

from examples.symbolic_discovery.synthetic_feature_discovery.benchmark import (
    DiscoveredFeature,
    build_design_matrix,
    discover_features_from_derivatives,
    evaluate_benchmark,
    make_dataset,
    true_law,
)


def test_dataset_uses_hidden_law_with_noise_free_target() -> None:
    data = make_dataset(n_samples=100, noise_std=0.0, seed=1)
    assert np.allclose(data.y_train, true_law(data.x_train))
    assert data.x_train.shape[1] == 4
    assert data.x_test.size > 0


def test_design_matrix_builds_discovered_terms() -> None:
    x = np.asarray([[2.0, 3.0, 4.0, np.pi / 2]])
    features = [
        DiscoveredFeature("x1^2", "square", (0,), 1.0),
        DiscoveredFeature("x2*x3", "product", (1, 2), 1.0),
        DiscoveredFeature("sin(x4)", "sin", (3,), 1.0),
    ]
    design, names = build_design_matrix(x, features)
    assert names[-3:] == ["x1^2", "x2*x3", "sin(x4)"]
    assert np.allclose(design[0, -3:], [4.0, 12.0, 1.0])


def test_derivative_discovery_selects_true_terms_from_exact_derivatives() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, size=(500, 4))
    x[:, 3] = rng.uniform(-np.pi, np.pi, size=500)
    grad = np.zeros_like(x)
    grad[:, 0] = 6.0 * x[:, 0]
    grad[:, 1] = -2.0 * x[:, 2]
    grad[:, 2] = -2.0 * x[:, 1]
    grad[:, 3] = np.cos(x[:, 3])
    hess = np.zeros((x.shape[0], 4, 4))
    hess[:, 0, 0] = 6.0
    hess[:, 1, 2] = -2.0
    hess[:, 2, 1] = -2.0
    hess[:, 3, 3] = -np.sin(x[:, 3])
    features = discover_features_from_derivatives(x, grad, hess)
    names = {feature.name for feature in features}
    assert "x1^2" in names
    assert "x2*x3" in names
    assert "sin(x4)" in names


def test_benchmark_fairness_metadata() -> None:
    results = evaluate_benchmark(n_samples=300, hidden=64, noise_std=0.05, seed=2)
    protocol = results["fairness_protocol"]
    assert protocol["feature_discovery_split"] == "train only"
    assert protocol["final_scoring_split"] == "test only"
    assert protocol["omnibias_knows_formula"] is False
