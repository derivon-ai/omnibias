# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the C-MAPSS feature-discovery benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from examples.symbolic_discovery.cmapss_feature_discovery.benchmark import (
    CandidateFeature,
    _feature_gain,
    _gumbel_topk_selection,
    _nasa_rul_score,
    _operator_order,
    _past_trajectory_matrix,
    _postprocess_rul_predictions,
    _pretty_feature_name,
    _pretty_feature_names,
    _standard_last_cycle_metrics,
)
from examples.symbolic_discovery.joint_operator_regressor.cmapss_benchmark import (
    _sequence_window_operator_matrix,
)


def test_pretty_feature_name_maps_indices_without_partial_replacement() -> None:
    cols = [f"f{i}" for i in range(1, 12)]
    assert _pretty_feature_name("x1^2", cols) == "f1^2"
    assert _pretty_feature_name("x10*x11", cols) == "f10*f11"


def test_pretty_feature_names_maps_list() -> None:
    assert _pretty_feature_names(["x1", "x2*x3"], ["a", "b", "c"]) == ["a", "b*c"]


def test_feature_gain_reports_delta() -> None:
    models = {
        "raw_ridge": {"rmse_cycles": 10.0},
        "omnibias_functional_selected_ridge": {"rmse_cycles": 9.0},
        "generic_dictionary_ridge": {"rmse_cycles": 9.5},
    }
    gain = _feature_gain(models)
    assert np.isclose(gain["ridge"]["delta_rmse_cycles"], -1.0)
    assert np.isclose(gain["ridge"]["delta_vs_generic_dictionary_cycles"], -0.5)


def test_nasa_rul_score_penalizes_late_predictions_more() -> None:
    early = _nasa_rul_score(np.array([100.0]), np.array([90.0]))
    late = _nasa_rul_score(np.array([100.0]), np.array([110.0]))
    assert late > early


def test_standard_last_cycle_metrics_use_one_row_per_engine() -> None:
    y_true = np.array([100.0, 80.0, 40.0, 30.0])
    y_pred = np.array([100.0, 70.0, 40.0, 50.0])
    unit = np.array([1, 1, 2, 2])
    time_cycles = np.array([1, 2, 1, 2])
    metrics = _standard_last_cycle_metrics(y_true, y_pred, unit, time_cycles)
    assert metrics["n_engines"] == 2
    assert np.isclose(metrics["rmse_cycles"], np.sqrt((10.0**2 + 20.0**2) / 2.0))


def test_postprocess_rul_predictions_enforces_physical_range() -> None:
    pred = _postprocess_rul_predictions(np.array([-5.0, 50.0, 200.0]))
    assert pred.tolist() == [0.0, 50.0, 125.0]


def test_past_trajectory_matrix_preserves_row_order() -> None:
    df = pd.DataFrame(
        {
            "unit_number": [2, 1, 1, 2],
            "time_cycles": [2, 1, 2, 1],
            "s_2": [12.0, 10.0, 11.0, 10.0],
        }
    )
    matrix, names = _past_trajectory_matrix(df, ["time_cycles", "s_2"])
    delta_idx = names.index("s_2_delta0")
    diff_idx = names.index("s_2_diff1")
    assert matrix[:, delta_idx].tolist() == [2.0, 0.0, 1.0, 0.0]
    assert matrix[:, diff_idx].tolist() == [2.0, 0.0, 1.0, 0.0]


def test_past_trajectory_matrix_adds_high_order_derivatives() -> None:
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4, 5, 6, 7],
            "s_2": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0],
        }
    )
    _, names = _past_trajectory_matrix(df, ["time_cycles", "s_2"])
    assert "s_2_diff6" in names
    assert _operator_order("s_2_diff6") == 6


def test_past_trajectory_features_do_not_use_future_rows() -> None:
    base = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4],
            "s_2": [10.0, 11.0, 12.0, 13.0],
        }
    )
    changed_future = base.copy()
    changed_future.loc[3, "s_2"] = 1000.0
    base_matrix, names = _past_trajectory_matrix(base, ["time_cycles", "s_2"])
    changed_matrix, _ = _past_trajectory_matrix(changed_future, ["time_cycles", "s_2"])
    delta_idx = names.index("s_2_delta0")
    roll_idx = names.index("s_2_roll5")
    np.testing.assert_allclose(base_matrix[:3, [delta_idx, roll_idx]], changed_matrix[:3, [delta_idx, roll_idx]])


def test_sequence_window_operator_matrix_uses_only_past_rows() -> None:
    base = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4],
            "RUL": [3.0, 2.0, 1.0, 0.0],
            "s_2": [10.0, 11.0, 12.0, 13.0],
        }
    )
    changed_future = base.copy()
    changed_future.loc[3, "s_2"] = 1000.0
    base_matrix, base_y, _, names = _sequence_window_operator_matrix(
        base,
        ["s_2"],
        window=3,
        endpoint_mode="sliding",
        seed=0,
    )
    changed_matrix, changed_y, _, _ = _sequence_window_operator_matrix(
        changed_future,
        ["s_2"],
        window=3,
        endpoint_mode="sliding",
        seed=0,
    )
    last_idx = names.index("s_2_win_last")
    slope_idx = names.index("s_2_win_slope")
    np.testing.assert_allclose(base_matrix[0, [last_idx, slope_idx]], changed_matrix[0, [last_idx, slope_idx]])
    assert base_y[0] == changed_y[0] == 1.0


def test_gumbel_selector_prefers_informative_candidate() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 2))
    base = x[:, 1:2]
    y = x[:, 0] + rng.normal(scale=0.01, size=80)
    noise = rng.normal(size=80)
    candidates = [
        CandidateFeature("signal", "test", x[:, 0], x[:, 0], x[:, 0], complexity=1.0),
        CandidateFeature("noise", "test", noise, noise, noise, complexity=1.0),
    ]
    selected, details = _gumbel_topk_selection(base, y, base, y, candidates, max_selected_features=1, prefilter=2, seed=0, n_draws=4)
    assert details["selector"] == "gumbel"
    assert selected[0].name == "signal"
