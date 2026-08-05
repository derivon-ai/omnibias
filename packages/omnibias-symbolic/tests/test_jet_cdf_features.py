# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""NeuralJet probability-operator features: CDF columns in jet discovery."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic.discovery import (
    JetBundle,
    NeuralJetDiscoverer,
    build_jet_cdf_features,
    build_jet_relation_library,
    fit_jet_cdf_plan,
)


def _sigmoid(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def _sigmoidal_bundles() -> tuple[JetBundle, JetBundle, JetBundle, float, float]:
    """Three splits whose target law is ``dy = sigmoid((x - loc0)/s0)`` exactly."""
    x_tr = np.linspace(-2.0, 2.0, 200)
    x_va = np.linspace(-1.8, 1.8, 150)
    x_te = np.linspace(-1.9, 1.9, 150)
    loc0 = float(np.median(x_tr))  # the 0.5 quantile lands on the n_locations=5 grid
    s0 = float(np.std(x_tr))  # scale_mult 1.0 lands on the grid

    def bundle(x: np.ndarray) -> JetBundle:
        y = np.cos(x)  # order-0 jet, distinct from x
        dy = _sigmoid((x - loc0) / s0)  # target relation
        return JetBundle(x=x, jets=np.stack([y, dy], axis=1))

    return bundle(x_tr), bundle(x_va), bundle(x_te), loc0, s0


# ----- plan + feature builders ----------------------------------------------


def test_fit_jet_cdf_plan_shapes_and_train_quantiles() -> None:
    train, _, _, _, _ = _sigmoidal_bundles()
    per_locs, per_scales = fit_jet_cdf_plan(
        train, lhs_order=1, bases=("sigmoid",), n_locations=5, scale_mults=(0.5, 1.0, 2.0)
    )
    assert len(per_locs) == 2 and len(per_scales) == 2  # variables = [x, y]
    assert per_locs[0].shape == (5,) and per_scales[0].shape == (3,)
    # the middle of 5 linspace(0.1, 0.9) quantiles is the median
    assert per_locs[0][2] == pytest.approx(float(np.median(train.x)))
    assert per_scales[0][1] == pytest.approx(float(np.std(train.x)))


def test_build_jet_cdf_features_are_bounded_and_named() -> None:
    train, val, _, _, _ = _sigmoidal_bundles()
    per_locs, per_scales = fit_jet_cdf_plan(train, lhs_order=1, bases=("sigmoid", "tanh"))
    design, names = build_jet_cdf_features(
        val,
        lhs_order=1,
        bases=("sigmoid", "tanh"),
        per_variable_locations=per_locs,
        per_variable_scales=per_scales,
    )
    assert design.shape[0] == val.x.shape[0]
    assert design.shape[1] == len(names)
    assert np.all((design >= 0.0) & (design <= 1.0))  # CDF transforms
    assert any(n.startswith("sigmoid((x") for n in names)
    assert any(n.startswith("tanh((y") for n in names)


def test_plan_uses_fixed_grid_across_splits_no_leakage() -> None:
    train, val, _, _, _ = _sigmoidal_bundles()
    per_locs, per_scales = fit_jet_cdf_plan(train, lhs_order=1, bases=("sigmoid",))
    # Manually transform val.x with the train-fit grid: must match the builder.
    col = val.x
    locs, scales = per_locs[0], per_scales[0]
    expected_first = _sigmoid((col - locs[0]) / scales[0])
    design, names = build_jet_cdf_features(
        val,
        lhs_order=1,
        bases=("sigmoid",),
        per_variable_locations=per_locs,
        per_variable_scales=per_scales,
    )
    # first column is base sigmoid, first scale, first location of variable x
    assert names[0] == f"sigmoid((x-{locs[0]:.4g})/{scales[0]:.4g})"
    assert np.allclose(design[:, 0], expected_first)


# ----- end-to-end discovery -------------------------------------------------


def test_cdf_features_recover_sigmoidal_law() -> None:
    train, val, test, _, _ = _sigmoidal_bundles()
    with_cdf = NeuralJetDiscoverer(
        max_library_degree=2, cdf_feature_bases=("sigmoid",)
    ).discover(train, val, test, candidate_lhs_orders=(1,))
    assert with_cdf.test_rmse < 1e-5
    assert "sigmoid(" in with_cdf.formula()


def test_cdf_features_beat_polynomial_only_on_saturating_law() -> None:
    train, val, test, _, _ = _sigmoidal_bundles()
    with_cdf = NeuralJetDiscoverer(
        max_library_degree=2, cdf_feature_bases=("sigmoid",)
    ).discover(train, val, test, candidate_lhs_orders=(1,))
    poly_only = NeuralJetDiscoverer(max_library_degree=2).discover(
        train, val, test, candidate_lhs_orders=(1,)
    )
    assert poly_only.test_rmse > 50.0 * with_cdf.test_rmse


def test_empty_cdf_bases_is_pure_polynomial() -> None:
    # Default (no CDF bases) must leave the jet design untouched.
    train, _, _, _, _ = _sigmoidal_bundles()
    disc = NeuralJetDiscoverer()
    assert disc.cdf_feature_bases == ()
    design, names = build_jet_relation_library(train, lhs_order=1, max_degree=2)
    assert all("sigmoid" not in n and "tanh" not in n for n in names)
