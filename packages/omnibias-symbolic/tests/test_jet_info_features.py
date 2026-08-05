# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""NeuralJet information-operator features: surprisal columns in jet discovery.

The information-theoretic twin of the CDF jet features: every right-hand-side jet
coordinate is mapped through the surprisal ``-ln f((v - loc)/scale)`` of a base
density, letting :class:`NeuralJetDiscoverer` recover differential laws with
log-likelihood / energy structure that a polynomial or CDF jet library cannot.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic.discovery import (
    JetBundle,
    NeuralJetDiscoverer,
    build_jet_info_features,
    build_jet_relation_library,
    fit_jet_info_plan,
)


def _surprisal_sigmoid(u: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, u) + np.logaddexp(0.0, -u)


def _surprisal_bundles() -> tuple[JetBundle, JetBundle, JetBundle, float, float]:
    """Three splits whose target law is ``dy = surprisal_sigmoid((x - loc0)/s0)``."""
    x_tr = np.linspace(-2.0, 2.0, 200)
    x_va = np.linspace(-1.8, 1.8, 150)
    x_te = np.linspace(-1.9, 1.9, 150)
    loc0 = float(np.median(x_tr))  # the 0.5 quantile lands on the n_locations=5 grid
    s0 = float(np.std(x_tr))  # scale_mult 1.0 lands on the grid

    def bundle(x: np.ndarray) -> JetBundle:
        y = np.cos(x)  # order-0 jet, distinct from x
        dy = _surprisal_sigmoid((x - loc0) / s0)  # target relation
        return JetBundle(x=x, jets=np.stack([y, dy], axis=1))

    return bundle(x_tr), bundle(x_va), bundle(x_te), loc0, s0


# ----- plan + feature builders ----------------------------------------------


def test_fit_jet_info_plan_shapes_and_train_quantiles() -> None:
    train, _, _, _, _ = _surprisal_bundles()
    per_locs, per_scales = fit_jet_info_plan(
        train, lhs_order=1, bases=("sigmoid",), n_locations=5, scale_mults=(0.5, 1.0, 2.0)
    )
    assert len(per_locs) == 2 and len(per_scales) == 2  # variables = [x, y]
    assert per_locs[0].shape == (5,) and per_scales[0].shape == (3,)
    assert per_locs[0][2] == pytest.approx(float(np.median(train.x)))
    assert per_scales[0][1] == pytest.approx(float(np.std(train.x)))


def test_build_jet_info_features_match_closed_form_and_names() -> None:
    train, val, _, _, _ = _surprisal_bundles()
    per_locs, per_scales = fit_jet_info_plan(train, lhs_order=1, bases=("sigmoid", "arctan"))
    design, names = build_jet_info_features(
        val,
        lhs_order=1,
        bases=("sigmoid", "arctan"),
        per_variable_locations=per_locs,
        per_variable_scales=per_scales,
    )
    assert design.shape[0] == val.x.shape[0]
    assert design.shape[1] == len(names)
    assert np.all(design >= 0.0)  # surprisal of a sub-unit density is non-negative here
    assert any(n.startswith("surprisal_sigmoid((x") for n in names)
    assert any(n.startswith("surprisal_arctan((y") for n in names)
    # first column == surprisal_sigmoid of variable x at (loc0, scale0)
    locs, scales = per_locs[0], per_scales[0]
    expected = _surprisal_sigmoid((val.x - locs[0]) / scales[0])
    assert names[0] == f"surprisal_sigmoid((x-{locs[0]:.4g})/{scales[0]:.4g})"
    assert np.allclose(design[:, 0], expected, atol=1e-12)


def test_plan_uses_fixed_grid_across_splits_no_leakage() -> None:
    train, _, test, _, _ = _surprisal_bundles()
    per_locs, per_scales = fit_jet_info_plan(train, lhs_order=1, bases=("sigmoid",))
    _, names_train = build_jet_info_features(
        train,
        lhs_order=1,
        bases=("sigmoid",),
        per_variable_locations=per_locs,
        per_variable_scales=per_scales,
    )
    _, names_test = build_jet_info_features(
        test,
        lhs_order=1,
        bases=("sigmoid",),
        per_variable_locations=per_locs,
        per_variable_scales=per_scales,
    )
    assert names_train == names_test


# ----- end-to-end discovery -------------------------------------------------


def test_info_features_recover_surprisal_law() -> None:
    train, val, test, _, _ = _surprisal_bundles()
    result = NeuralJetDiscoverer(
        max_library_degree=2, info_feature_bases=("sigmoid",)
    ).discover(train, val, test, candidate_lhs_orders=(1,))
    assert result.test_rmse < 1e-5
    assert "surprisal_sigmoid(" in result.formula()


def test_info_features_beat_polynomial_only_on_surprisal_law() -> None:
    train, val, test, _, _ = _surprisal_bundles()
    with_info = NeuralJetDiscoverer(
        max_library_degree=2, info_feature_bases=("sigmoid",)
    ).discover(train, val, test, candidate_lhs_orders=(1,))
    poly_only = NeuralJetDiscoverer(max_library_degree=2).discover(
        train, val, test, candidate_lhs_orders=(1,)
    )
    assert poly_only.test_rmse > 20.0 * with_info.test_rmse


def test_info_and_cdf_features_compose() -> None:
    train, val, test, _, _ = _surprisal_bundles()
    result = NeuralJetDiscoverer(
        max_library_degree=1,
        cdf_feature_bases=("sigmoid",),
        info_feature_bases=("sigmoid",),
    ).discover(train, val, test, candidate_lhs_orders=(1,))
    # the surprisal column is the exact law, so it should still be recovered
    assert result.test_rmse < 1e-5
    assert "surprisal_sigmoid(" in result.formula()


def test_empty_info_bases_is_pure_polynomial() -> None:
    train, _, _, _, _ = _surprisal_bundles()
    disc = NeuralJetDiscoverer()
    assert disc.info_feature_bases == ()
    _, names = build_jet_relation_library(train, lhs_order=1, max_degree=2)
    assert all("surprisal" not in n for n in names)


# ----- guards ---------------------------------------------------------------


def test_fit_jet_info_plan_rejects_unknown_base() -> None:
    train, _, _, _, _ = _surprisal_bundles()
    with pytest.raises(ValueError, match="unknown information base"):
        fit_jet_info_plan(train, lhs_order=1, bases=("not_a_density",))


def test_fit_jet_info_plan_rejects_bad_scale_mults() -> None:
    train, _, _, _, _ = _surprisal_bundles()
    with pytest.raises(ValueError, match="scale_mults"):
        fit_jet_info_plan(train, lhs_order=1, scale_mults=(0.0, 1.0))
