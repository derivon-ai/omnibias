# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Synthetic smoke tests for battery law discovery."""

from __future__ import annotations

import numpy as np

from examples.symbolic_discovery.battery_law_discovery.features import (
    build_feature_bundle,
    fit_feature_stats,
)
from examples.symbolic_discovery.battery_law_discovery.omnibias_law_model import (
    build_operator_library,
    build_physics_library,
    fit_physics_constrained_law,
    fit_sparse_law,
    predict_law,
    rollout_law,
    select_physics_operator_law,
)
from examples.symbolic_discovery.battery_law_discovery.severson_loader import (
    make_synthetic_cycle_table,
    train_test_protocol_split,
)


def test_synthetic_table_has_expected_columns() -> None:
    table = make_synthetic_cycle_table(n_cells=3, n_cycles=20, seed=7)
    for name in ["cell_id", "cycle_norm", "capacity_norm", "c_rate", "temperature_c"]:
        assert name in table.columns
    assert len(table) == 60


def test_sparse_law_recovers_exponential_decay_sign() -> None:
    n = np.linspace(0.0, 1.0, 200)
    k = 0.17
    q = np.exp(-k * n)
    dqdn = -k * q
    d2qdn2 = k * k * q
    x = np.zeros((n.size, 2))
    x[:, 0] = n
    library, names = build_operator_library(x, q=q, d2qdn2=d2qdn2)
    law = fit_sparse_law(library, dqdn, names, threshold=1e-4)
    pred = predict_law(law, library)
    assert np.sqrt(np.mean((pred - dqdn) ** 2)) < 1e-4
    assert law.coef[names.index("q")] < 0.0


def test_feature_bundle_shapes() -> None:
    table = make_synthetic_cycle_table(n_cells=4, n_cycles=12, seed=1)
    stats = fit_feature_stats(table)
    bundle = build_feature_bundle(table, stats)
    assert bundle.x.shape == (48, 5)
    assert bundle.y.shape == (48,)
    assert bundle.feature_names[0] == "cycle_norm"


def test_physics_law_rollout_is_monotone() -> None:
    n = np.linspace(0.0, 1.0, 200)
    k = 0.17
    q = np.exp(-k * n)
    dqdn = -k * q
    x = np.zeros((n.size, 2))
    x[:, 0] = n
    library, names = build_physics_library(x, q=q)
    law = fit_physics_constrained_law(library, dqdn, q, names, threshold=1e-6)
    pred_dqdn = predict_law(law, library, q=q)
    rolled = rollout_law(law, x[0], float(q[0]), n)
    assert np.all(np.diff(rolled) <= 1e-12)
    assert np.sqrt(np.mean((pred_dqdn - dqdn) ** 2)) < 1e-3


def test_physics_operator_auto_selects_known_operator_set() -> None:
    n = np.linspace(0.0, 1.0, 160)
    q = np.exp(-0.2 * n)
    dqdn = -0.2 * q
    x = np.zeros((n.size, 3))
    x[:, 0] = n
    law = select_physics_operator_law(x, q, dqdn, threshold=1e-6, seed=4)
    assert law.operator_set in {"minimal", "poly", "capacity", "stress", "stress_interactions"}
    assert law.kind == "degradation"


def test_protocol_split_holds_out_high_c_rates() -> None:
    table = make_synthetic_cycle_table(n_cells=10, n_cycles=8, seed=3)
    train, test = train_test_protocol_split(table, test_fraction=0.3)
    train_rate = np.nanmedian(train.require("c_rate").astype(float))
    test_rate = np.nanmedian(test.require("c_rate").astype(float))
    assert len(train) > 0
    assert len(test) > 0
    assert test_rate >= train_rate
