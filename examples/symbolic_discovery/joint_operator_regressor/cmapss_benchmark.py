# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""C-MAPSS refinement loop for the joint operator regressor."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omnibias.torch.architectures import fit_joint_operator_regressor
from torch import Tensor

from examples.symbolic_discovery.cmapss_feature_discovery.benchmark import (
    TARGET_CAP,
    _even_sample_indices,
    _fair_operator_dictionary_matrices,
    _fit_optional_hgb_cycles,
    _past_trajectory_matrices,
    _postprocess_rul_predictions,
    _standard_last_cycle_metrics,
    discover_features_from_derivatives,
    field_value_grad_hessian,
    fit_omnibias_field,
    load_splits,
)


def evaluate_cmapss_joint_benchmark(
    data_dir: Path,
    out_dir: Path,
    *,
    seed: int = 0,
    max_screened_features: int = 36,
) -> dict[str, object]:
    split = load_splits(data_dir, seed=seed)
    field_train_df = split["field_train_df"]  # type: ignore[assignment]
    train_df = split["model_train_df"]  # type: ignore[assignment]
    val_df = split["tune_val_df"]  # type: ignore[assignment]
    test_df = split["test_df"]  # type: ignore[assignment]
    feature_cols = split["feature_cols"]  # type: ignore[assignment]

    raw_train = train_df[feature_cols].to_numpy(float)
    raw_val = val_df[feature_cols].to_numpy(float)
    raw_test = test_df[feature_cols].to_numpy(float)
    phm_train, phm_val, phm_test, phm_names = _past_trajectory_matrices(train_df, val_df, test_df, feature_cols)
    (
        seq_train,
        seq_y_train,
        seq_train_rows,
        seq_val,
        seq_y_val,
        seq_y_val_standard,
        seq_val_rows,
        seq_test,
        seq_y_test,
        seq_y_test_standard,
        seq_test_rows,
        seq_test_df,
        seq_names,
    ) = _sequence_window_operator_matrices(
        train_df,
        val_df,
        test_df,
        feature_cols,
        window=30,
        val_endpoints_per_engine=5,
        seed=seed,
    )
    seq_raw_train, _, _ = _sequence_raw_window_array(
        train_df,
        feature_cols,
        window=30,
        endpoint_mode="sliding",
        seed=seed,
    )
    seq_raw_val, _, _ = _sequence_raw_window_array(
        val_df,
        feature_cols,
        window=30,
        endpoint_mode="pseudo",
        seed=seed,
        endpoints_per_engine=5,
    )
    seq_raw_test, _, _ = _sequence_raw_window_array(
        test_df,
        feature_cols,
        window=30,
        endpoint_mode="last",
        seed=seed,
    )

    y_train = np.minimum(train_df["RUL"].to_numpy(float), TARGET_CAP)
    y_val = np.minimum(val_df["RUL"].to_numpy(float), TARGET_CAP)
    y_val_standard = val_df["RUL"].to_numpy(float)
    y_test = np.minimum(test_df["RUL"].to_numpy(float), TARGET_CAP)
    y_test_standard = test_df["RUL"].to_numpy(float)

    fair_train, fair_val, fair_test, fair_names = _build_fair_operator_matrices(
        field_train_df,
        train_df,
        val_df,
        test_df,
        feature_cols,
        seed=seed,
    )
    sequence_fair_train = np.concatenate([fair_train[seq_train_rows], seq_train], axis=1)
    sequence_fair_val = np.concatenate([fair_val[seq_val_rows], seq_val], axis=1)
    sequence_fair_test = np.concatenate([fair_test[seq_test_rows], seq_test], axis=1)
    sequence_fair_names = fair_names + seq_names

    selected_idx = _screen_features(phm_train, y_train, phm_val, y_val, max_screened_features)
    screened_train = phm_train[:, selected_idx]
    screened_val = phm_val[:, selected_idx]
    screened_test = phm_test[:, selected_idx]
    screened_names = [phm_names[int(idx)] for idx in selected_idx]

    raw_pred, raw_details = _fit_tuned_ridge_cycles(raw_train, y_train, raw_val, y_val, raw_test)
    raw_val_pred, _ = _fit_tuned_ridge_cycles(raw_train, y_train, raw_val, y_val, raw_val, refit=False)
    raw_calibrator = _fit_endpoint_rul_calibrator(val_df, y_val_standard, raw_val_pred, seed=seed)
    raw_pred = _apply_rul_calibrator(raw_pred, raw_calibrator)

    phm_pred, phm_details = _fit_tuned_ridge_cycles(screened_train, y_train, screened_val, y_val, screened_test)
    phm_val_pred, _ = _fit_tuned_ridge_cycles(screened_train, y_train, screened_val, y_val, screened_val, refit=False)
    phm_calibrator = _fit_endpoint_rul_calibrator(val_df, y_val_standard, phm_val_pred, seed=seed)
    phm_pred = _apply_rul_calibrator(phm_pred, phm_calibrator)

    fair_pred, fair_details = _fit_tuned_ridge_cycles(fair_train, y_train, fair_val, y_val, fair_test)
    fair_val_pred, _ = _fit_tuned_ridge_cycles(fair_train, y_train, fair_val, y_val, fair_val, refit=False)
    fair_calibrator = _fit_endpoint_rul_calibrator(val_df, y_val_standard, fair_val_pred, seed=seed)
    fair_pred = _apply_rul_calibrator(fair_pred, fair_calibrator)
    fair_endpoint_pred, fair_endpoint_details = _fit_endpoint_tuned_ridge_cycles(
        fair_train,
        y_train,
        fair_val,
        y_val,
        y_val_standard,
        val_df,
        fair_test,
        seed=seed,
    )
    sequence_pred, sequence_details = _fit_endpoint_tuned_array_ridge_cycles(
        seq_train,
        seq_y_train,
        seq_val,
        seq_y_val,
        seq_y_val_standard,
        seq_test,
    )
    sequence_fair_pred, sequence_fair_details = _fit_endpoint_tuned_array_ridge_cycles(
        sequence_fair_train,
        seq_y_train,
        sequence_fair_val,
        seq_y_val,
        seq_y_val_standard,
        sequence_fair_test,
    )
    hgb = _fit_optional_hgb_cycles(
        screened_train,
        _zscore(y_train, y_train),
        screened_val,
        _zscore(y_val, y_train),
        screened_test,
        y_mean=float(y_train.mean()),
        y_scale=float(y_train.std()),
    )

    train_weight = _training_weight(train_df, y_train)
    val_weight = _training_weight(val_df, y_val)
    endpoint_selection_metric = _endpoint_nasa_selection_metric(val_df, y_val_standard, seed=seed)
    sequence_selected_idx = _screen_features(seq_train, seq_y_train, seq_val, seq_y_val, min(64, seq_train.shape[1]))
    sequence_train_screened = seq_train[:, sequence_selected_idx]
    sequence_val_screened = seq_val[:, sequence_selected_idx]
    sequence_test_screened = seq_test[:, sequence_selected_idx]
    sequence_screened_names = [seq_names[int(idx)] for idx in sequence_selected_idx]
    sequence_selection_metric = _array_endpoint_nasa_selection_metric(seq_y_val_standard)
    sequence_joint = fit_joint_operator_regressor(
        sequence_train_screened,
        seq_y_train,
        sequence_val_screened,
        seq_y_val,
        seed=seed + 303,
        epochs=180,
        lr=8e-3,
        batch_size=4096,
        patience=55,
        sparsity_weight=5e-2,
        asymmetric_weight=2e-2,
        validation_asymmetric_weight=5e-2,
        train_sample_weight=_late_life_weight(seq_y_train),
        val_sample_weight=_late_life_weight(seq_y_val),
        standardize_x=True,
        polish_readout=False,
        validation_selection_metric=sequence_selection_metric,
        validation_selection_complexity_weight=0.0,
        model_kwargs={
            "include_raw": True,
            "include_unary": True,
            "include_pairwise": True,
            "include_nested": False,
            "max_pairwise": 256,
            "ombu_channels": 2,
            "stochastic_gates": False,
            "initial_gate_logit": -1.5,
        },
    )
    sequence_joint_val_pred = _postprocess_rul_predictions(sequence_joint.predict(sequence_val_screened))
    sequence_joint_calibrator = _fit_array_endpoint_rul_calibrator(seq_y_val_standard, sequence_joint_val_pred)
    sequence_joint_pred = _apply_rul_calibrator(
        sequence_joint.predict(sequence_test_screened),
        sequence_joint_calibrator,
    )
    sequence_joint_selected = _rename_selected_operators(
        sequence_joint.selected_operators(top_k=20),
        sequence_screened_names,
    )
    sequence_attention_pred, sequence_attention_details, sequence_attention_selected = _fit_sequence_attention_regressor(
        seq_raw_train,
        seq_y_train,
        seq_raw_val,
        seq_y_val,
        seq_y_val_standard,
        seq_raw_test,
        feature_cols,
        seed=seed,
    )
    sequence_tcn_pred, sequence_tcn_details, sequence_tcn_selected = _fit_sequence_tcn_hybrid_regressor(
        seq_raw_train,
        fair_train[seq_train_rows],
        seq_y_train,
        seq_raw_val,
        fair_val[seq_val_rows],
        seq_y_val,
        seq_y_val_standard,
        seq_raw_test,
        fair_test[seq_test_rows],
        feature_cols,
        fair_names,
        seed=seed,
    )
    fitted = fit_joint_operator_regressor(
        screened_train,
        y_train,
        screened_val,
        y_val,
        seed=seed,
        epochs=240,
        lr=1e-2,
        batch_size=2048,
        patience=50,
        sparsity_weight=8e-2,
        asymmetric_weight=2e-2,
        validation_asymmetric_weight=5e-2,
        train_sample_weight=train_weight,
        val_sample_weight=val_weight,
        standardize_x=True,
        polish_ridge=1e-4,
        model_kwargs={
            "include_raw": True,
            "include_unary": True,
            "include_pairwise": True,
            "include_nested": False,
            "max_pairwise": 128,
            "ombu_channels": 4,
            "stochastic_gates": False,
            "initial_gate_logit": -1.25,
        },
    )
    joint_val_pred = _postprocess_rul_predictions(fitted.predict(screened_val))
    joint_calibrator = _fit_endpoint_rul_calibrator(val_df, y_val_standard, joint_val_pred, seed=seed)
    joint_pred = _apply_rul_calibrator(fitted.predict(screened_test), joint_calibrator)
    selected = _rename_selected_operators(fitted.selected_operators(top_k=20), screened_names)

    fair_joint = fit_joint_operator_regressor(
        fair_train,
        y_train,
        fair_val,
        y_val,
        seed=seed + 101,
        epochs=180,
        lr=8e-3,
        batch_size=4096,
        patience=40,
        sparsity_weight=3e-2,
        asymmetric_weight=2e-2,
        validation_asymmetric_weight=5e-2,
        train_sample_weight=train_weight,
        val_sample_weight=val_weight,
        standardize_x=True,
        polish_ridge=1e-4,
        model_kwargs={
            "include_raw": True,
            "include_unary": False,
            "include_pairwise": False,
            "include_nested": False,
            "ombu_channels": 0,
            "stochastic_gates": False,
            "initial_gate_logit": -1.8,
        },
    )
    fair_joint_val_pred = _postprocess_rul_predictions(fair_joint.predict(fair_val))
    fair_joint_calibrator = _fit_endpoint_rul_calibrator(val_df, y_val_standard, fair_joint_val_pred, seed=seed)
    fair_joint_pred = _apply_rul_calibrator(fair_joint.predict(fair_test), fair_joint_calibrator)
    fair_joint_selected = _rename_selected_operators(fair_joint.selected_operators(top_k=20), fair_names)
    fair_joint_endpoint = fit_joint_operator_regressor(
        fair_train,
        y_train,
        fair_val,
        y_val,
        seed=seed + 101,
        epochs=240,
        lr=8e-3,
        batch_size=4096,
        patience=80,
        sparsity_weight=3e-2,
        asymmetric_weight=2e-2,
        validation_asymmetric_weight=5e-2,
        train_sample_weight=train_weight,
        val_sample_weight=val_weight,
        standardize_x=True,
        polish_ridge=1e-4,
        validation_selection_metric=endpoint_selection_metric,
        validation_selection_complexity_weight=0.0,
        model_kwargs={
            "include_raw": True,
            "include_unary": False,
            "include_pairwise": False,
            "include_nested": False,
            "ombu_channels": 0,
            "stochastic_gates": False,
            "initial_gate_logit": -1.8,
        },
    )
    fair_joint_endpoint_val_pred = _postprocess_rul_predictions(fair_joint_endpoint.predict(fair_val))
    fair_joint_endpoint_calibrator = _fit_multi_endpoint_rul_calibrator(
        val_df,
        y_val_standard,
        fair_joint_endpoint_val_pred,
        seed=seed,
    )
    fair_joint_endpoint_single_calibrator = _fit_endpoint_rul_calibrator(
        val_df,
        y_val_standard,
        fair_joint_endpoint_val_pred,
        seed=seed,
    )
    fair_joint_endpoint_pred = _apply_rul_calibrator(
        fair_joint_endpoint.predict(fair_test),
        fair_joint_endpoint_calibrator,
    )
    fair_joint_endpoint_single_pred = _apply_rul_calibrator(
        fair_joint_endpoint.predict(fair_test),
        fair_joint_endpoint_single_calibrator,
    )
    fair_joint_endpoint_selected = _rename_selected_operators(
        fair_joint_endpoint.selected_operators(top_k=20),
        fair_names,
    )
    fair_ranked_indices = [
        int(item["index"])
        for item in fair_joint.selected_operators(threshold=0.0, top_k=len(fair_names))
    ]
    endpoint_selected_pred, endpoint_selected_details = _fit_endpoint_selected_ridge(
        fair_train,
        y_train,
        fair_val,
        y_val,
        y_val_standard,
        val_df,
        fair_test,
        fair_ranked_indices,
        seed=seed,
    )

    models: dict[str, object] = {
        "raw_ridge": _row_and_standard_metrics(y_test, y_test_standard, raw_pred, test_df) | {"n_features": len(feature_cols), "details": raw_details},
        "screened_phm_ridge": _row_and_standard_metrics(y_test, y_test_standard, phm_pred, test_df)
        | {"n_features": len(screened_names), "details": phm_details},
        "fair_operator_dictionary_ridge": _row_and_standard_metrics(y_test, y_test_standard, fair_pred, test_df)
        | {"n_features": len(fair_names), "details": fair_details | {"calibrator": fair_calibrator}},
        "fair_operator_dictionary_endpoint_ridge": _row_and_standard_metrics(
            y_test,
            y_test_standard,
            fair_endpoint_pred,
            test_df,
        )
        | {"n_features": len(fair_names), "details": fair_endpoint_details},
        "sequence_window_operator_ridge": _row_and_standard_metrics(
            seq_y_test,
            seq_y_test_standard,
            sequence_pred,
            seq_test_df,
        )
        | {
            "n_features": len(seq_names),
            "details": sequence_details | {"window": 30, "protocol": "sliding train windows; final test window per engine"},
        },
        "sequence_fair_hybrid_endpoint_ridge": _row_and_standard_metrics(
            seq_y_test,
            seq_y_test_standard,
            sequence_fair_pred,
            seq_test_df,
        )
        | {
            "n_features": len(sequence_fair_names),
            "details": sequence_fair_details
            | {
                "window": 30,
                "protocol": "fair endpoint operators plus sequence-window trajectory operators",
            },
        },
        "sequence_window_joint_operator_regressor": _row_and_standard_metrics(
            seq_y_test,
            seq_y_test_standard,
            sequence_joint_pred,
            seq_test_df,
        )
        | {
            "n_input_features": len(sequence_screened_names),
            "n_operators": sequence_joint.model.n_operators,
            "active_operator_count_0.2": sequence_joint.model.active_operator_count(0.2),
            "details": {
                "best_validation_endpoint_selection": min(sequence_joint.history["val_selection_score"]),
                "best_weighted_val_rmse_z": min(sequence_joint.history["val_rmse_z"]),
                "best_val_asymmetric": min(sequence_joint.history["val_asymmetric"]),
                "epochs_ran": len(sequence_joint.history["val_rmse_z"]),
                "calibrator": sequence_joint_calibrator,
                "window": 30,
                "screened_window_features": sequence_screened_names,
                "selection_objective": "sequence endpoint validation NASA score",
            },
        },
        "sequence_attention_window_regressor": _row_and_standard_metrics(
            seq_y_test,
            seq_y_test_standard,
            sequence_attention_pred,
            seq_test_df,
        )
        | {
            "n_input_features": len(feature_cols),
            "details": sequence_attention_details,
        },
        "sequence_tcn_fair_hybrid_regressor": _row_and_standard_metrics(
            seq_y_test,
            seq_y_test_standard,
            sequence_tcn_pred,
            seq_test_df,
        )
        | {
            "n_input_features": len(feature_cols),
            "n_side_features": len(sequence_tcn_details["selected_side_features"]),
            "details": sequence_tcn_details,
        },
        "fair_dictionary_endpoint_selected_ridge": _row_and_standard_metrics(
            y_test,
            y_test_standard,
            endpoint_selected_pred,
            test_df,
        )
        | {"n_features": endpoint_selected_details["selected_count"], "details": endpoint_selected_details},
        "joint_operator_regressor": _row_and_standard_metrics(y_test, y_test_standard, joint_pred, test_df)
        | {
            "n_input_features": len(screened_names),
            "n_operators": fitted.model.n_operators,
            "active_operator_count_0.2": fitted.model.active_operator_count(0.2),
            "details": {
                "best_weighted_val_rmse_z": min(fitted.history["val_rmse_z"]),
                "best_val_asymmetric": min(fitted.history["val_asymmetric"]),
                "epochs_ran": len(fitted.history["val_rmse_z"]),
                "calibrator": joint_calibrator,
            },
        },
        "fair_dictionary_joint_operator_regressor": _row_and_standard_metrics(
            y_test,
            y_test_standard,
            fair_joint_pred,
            test_df,
        )
        | {
            "n_input_features": len(fair_names),
            "n_operators": fair_joint.model.n_operators,
            "active_operator_count_0.2": fair_joint.model.active_operator_count(0.2),
            "details": {
                "best_weighted_val_rmse_z": min(fair_joint.history["val_rmse_z"]),
                "best_val_asymmetric": min(fair_joint.history["val_asymmetric"]),
                "epochs_ran": len(fair_joint.history["val_rmse_z"]),
                "calibrator": fair_joint_calibrator,
            },
        },
        "fair_dictionary_joint_endpoint_regressor": _row_and_standard_metrics(
            y_test,
            y_test_standard,
            fair_joint_endpoint_pred,
            test_df,
        )
        | {
            "n_input_features": len(fair_names),
            "n_operators": fair_joint_endpoint.model.n_operators,
            "active_operator_count_0.2": fair_joint_endpoint.model.active_operator_count(0.2),
            "details": {
                "best_validation_endpoint_selection": min(fair_joint_endpoint.history["val_selection_score"]),
                "best_weighted_val_rmse_z": min(fair_joint_endpoint.history["val_rmse_z"]),
                "best_val_asymmetric": min(fair_joint_endpoint.history["val_asymmetric"]),
                "epochs_ran": len(fair_joint_endpoint.history["val_rmse_z"]),
                "calibrator": fair_joint_endpoint_calibrator,
                "selection_objective": "early stopping on validation pseudo-endpoint NASA score",
            },
        },
        "fair_dictionary_joint_endpoint_single_calibrated": _row_and_standard_metrics(
            y_test,
            y_test_standard,
            fair_joint_endpoint_single_pred,
            test_df,
        )
        | {
            "n_input_features": len(fair_names),
            "n_operators": fair_joint_endpoint.model.n_operators,
            "active_operator_count_0.2": fair_joint_endpoint.model.active_operator_count(0.2),
            "details": {
                "best_validation_endpoint_selection": min(fair_joint_endpoint.history["val_selection_score"]),
                "best_weighted_val_rmse_z": min(fair_joint_endpoint.history["val_rmse_z"]),
                "best_val_asymmetric": min(fair_joint_endpoint.history["val_asymmetric"]),
                "epochs_ran": len(fair_joint_endpoint.history["val_rmse_z"]),
                "calibrator": fair_joint_endpoint_single_calibrator,
                "selection_objective": "same endpoint-aware joint model; single pseudo-endpoint calibration",
            },
        },
    }
    if hgb is not None:
        hgb_pred, hgb_details = hgb
        hgb_val = _fit_optional_hgb_cycles(
            screened_train,
            _zscore(y_train, y_train),
            screened_val,
            _zscore(y_val, y_train),
            screened_val,
            y_mean=float(y_train.mean()),
            y_scale=float(y_train.std()),
        )
        if hgb_val is not None:
            hgb_val_pred, _ = hgb_val
            hgb_calibrator = _fit_endpoint_rul_calibrator(val_df, y_val_standard, hgb_val_pred, seed=seed)
            hgb_pred = _apply_rul_calibrator(hgb_pred, hgb_calibrator)
        models["screened_phm_hist_gradient_boosting"] = _row_and_standard_metrics(
            y_test,
            y_test_standard,
            hgb_pred,
            test_df,
        ) | {"n_features": len(screened_names), "details": hgb_details}

    results = {
        "dataset": "NASA C-MAPSS FD001 turbofan degradation",
        "target": f"RUL clipped at {TARGET_CAP:g} cycles for row metrics; uncapped RUL for standard last-cycle metrics",
        "models": models,
        "screened_features": screened_names,
        "selected_operators": selected,
        "fair_dictionary_selected_operators": fair_joint_selected,
        "fair_dictionary_endpoint_selected_operators": fair_joint_endpoint_selected,
        "sequence_window_selected_operators": sequence_joint_selected,
        "sequence_attention_selected_features": sequence_attention_selected,
        "sequence_tcn_selected_features": sequence_tcn_selected,
        "fairness_protocol": {
            "feature_screening": "model-train and tune-validation engines only",
            "fair_dictionary": "same rich operator universe used by the prior fair-dictionary baseline",
            "calibration": "scale/offset tuned on pseudo-test endpoints from validation engines using NASA score",
            "joint_gate_training": "model-train engines with tune-validation early stopping",
            "endpoint_joint_gate_training": "fair-dictionary joint model early-stopped on validation pseudo-endpoint NASA score",
            "sequence_window_protocol": "30-cycle sliding train windows, validation pseudo-endpoint windows, final test window per engine",
            "sequence_attention_protocol": "temporal convolutions plus attention over raw 30-cycle sensor windows",
            "final_scoring": "held-out validation parquet only",
            "test_leakage": "no future rows are used in trajectory operators",
        },
        "seed": seed,
    }
    write_artifacts(results, out_dir)
    return results


def write_artifacts(results: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    report = [
        "# C-MAPSS Joint Operator Regressor",
        "",
        f"Dataset: `{results['dataset']}`",
        f"Target: `{results['target']}`",
        "",
        "## Test Metrics",
        "",
    ]
    models = results["models"]
    assert isinstance(models, dict)
    for name, metrics in sorted(models.items(), key=lambda item: item[1]["standard_last_cycle"]["rmse_cycles"]):
        row = metrics["row_metrics"]
        standard = metrics["standard_last_cycle"]
        report.append(
            f"- `{name}`: row RMSE `{row['rmse_cycles']:.4f}`, "
            f"last-cycle RMSE `{standard['rmse_cycles']:.4f}`, "
            f"NASA Score `{standard['nasa_score']:.2f}`"
        )
    report.extend(["", "## Selected Joint Operators", ""])
    for item in results["selected_operators"]:  # type: ignore[index]
        report.append(
            f"- `{item['name']}` ({item['family']}): gate `{item['gate_probability']:.3f}`, "
            f"importance `{item['importance']:.3f}`"
        )
    report.extend(["", "## Selected Fair-Dictionary Joint Operators", ""])
    for item in results.get("fair_dictionary_selected_operators", []):  # type: ignore[union-attr]
        report.append(
            f"- `{item['name']}` ({item['family']}): gate `{item['gate_probability']:.3f}`, "
            f"importance `{item['importance']:.3f}`"
        )
    report.extend(["", "## Endpoint-Aware Fair-Dictionary Joint Operators", ""])
    for item in results.get("fair_dictionary_endpoint_selected_operators", []):  # type: ignore[union-attr]
        report.append(
            f"- `{item['name']}` ({item['family']}): gate `{item['gate_probability']:.3f}`, "
            f"importance `{item['importance']:.3f}`"
        )
    report.extend(["", "## Sequence-Window Joint Operators", ""])
    for item in results.get("sequence_window_selected_operators", []):  # type: ignore[union-attr]
        report.append(
            f"- `{item['name']}` ({item['family']}): gate `{item['gate_probability']:.3f}`, "
            f"importance `{item['importance']:.3f}`"
        )
    report.extend(["", "## Sequence Attention Features", ""])
    for item in results.get("sequence_attention_selected_features", []):  # type: ignore[union-attr]
        report.append(f"- `{item['name']}`: attention `{item['attention']:.3f}`")
    report.extend(["", "## Sequence TCN Hybrid Features", ""])
    for item in results.get("sequence_tcn_selected_features", []):  # type: ignore[union-attr]
        report.append(f"- `{item['name']}`: importance `{item['importance']:.3f}`")
    report.extend(["", "## Fairness", ""])
    fairness = results["fairness_protocol"]
    assert isinstance(fairness, dict)
    for key, value in fairness.items():
        report.append(f"- `{key}`: {value}")
    (out_dir / "report.md").write_text("\n".join(report) + "\n")


def _build_fair_operator_matrices(
    field_train_df: object,
    train_df: object,
    val_df: object,
    test_df: object,
    feature_cols: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_field = field_train_df[feature_cols].to_numpy(float)  # type: ignore[index]
    y_field = np.minimum(field_train_df["RUL"].to_numpy(float), TARGET_CAP)  # type: ignore[index]
    y_scale = float(y_field.std()) or 1.0
    z_field = (y_field - float(y_field.mean())) / y_scale
    field_idx = _even_sample_indices(x_field.shape[0], 6000)
    field = fit_omnibias_field(x_field[field_idx], z_field[field_idx], hidden=512, ridge=1e-5, seed=seed)
    _, grad, hess = field_value_grad_hessian(field, x_field[field_idx])
    discovered = discover_features_from_derivatives(
        x_field[field_idx],
        grad,
        hess,
        curvature_threshold=0.5,
        interaction_threshold=0.5,
        periodic_threshold=0.85,
    )[:20]
    return _fair_operator_dictionary_matrices(train_df, val_df, test_df, feature_cols, discovered, field)


def _sequence_window_operator_matrices(
    train_df: object,
    val_df: object,
    test_df: object,
    feature_cols: list[str],
    *,
    window: int,
    val_endpoints_per_engine: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    object,
    list[str],
]:
    train_x, train_y, train_rows, names = _sequence_window_operator_matrix(
        train_df,
        feature_cols,
        window=window,
        endpoint_mode="sliding",
        seed=seed,
    )
    val_x, val_y_standard, val_rows, _ = _sequence_window_operator_matrix(
        val_df,
        feature_cols,
        window=window,
        endpoint_mode="pseudo",
        seed=seed,
        endpoints_per_engine=val_endpoints_per_engine,
    )
    test_x, test_y_standard, test_rows, _ = _sequence_window_operator_matrix(
        test_df,
        feature_cols,
        window=window,
        endpoint_mode="last",
        seed=seed,
    )
    return (
        train_x,
        np.minimum(train_y, TARGET_CAP),
        train_rows,
        val_x,
        np.minimum(val_y_standard, TARGET_CAP),
        val_y_standard,
        val_rows,
        test_x,
        np.minimum(test_y_standard, TARGET_CAP),
        test_y_standard,
        test_rows,
        test_df.iloc[test_rows].copy(),  # type: ignore[attr-defined]
        names,
    )


def _sequence_window_operator_matrix(
    df: object,
    feature_cols: list[str],
    *,
    window: int,
    endpoint_mode: str,
    seed: int,
    endpoints_per_engine: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    unit = df["unit_number"].to_numpy()  # type: ignore[index]
    time = df["time_cycles"].to_numpy()  # type: ignore[index]
    rul = df["RUL"].to_numpy(float)  # type: ignore[index]
    values = df[feature_cols].to_numpy(float)  # type: ignore[index]
    endpoints: list[int] = []
    for engine in np.unique(unit):
        idx = np.flatnonzero(unit == engine)
        idx = idx[np.argsort(time[idx])]
        if endpoint_mode == "sliding":
            endpoints.extend(int(row) for row in idx[window - 1 :])
        elif endpoint_mode == "last":
            endpoints.append(int(idx[-1]))
        elif endpoint_mode == "pseudo":
            used: set[int] = set()
            for slot in range(endpoints_per_engine):
                target_rul = 10.0 + float((int(engine) * 37 + seed * 11 + slot * 29) % 125)
                row = int(idx[np.argmin(np.abs(rul[idx] - target_rul))])
                if row not in used:
                    endpoints.append(row)
                    used.add(row)
        else:
            raise ValueError(f"unknown endpoint_mode: {endpoint_mode}")
    names = _sequence_window_operator_names(feature_cols)
    matrix = np.vstack([_sequence_window_features(values, unit, time, row, window) for row in endpoints])
    return matrix, rul[np.asarray(endpoints, dtype=int)], np.asarray(endpoints, dtype=int), names


def _sequence_raw_window_array(
    df: object,
    feature_cols: list[str],
    *,
    window: int,
    endpoint_mode: str,
    seed: int,
    endpoints_per_engine: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit = df["unit_number"].to_numpy()  # type: ignore[index]
    time = df["time_cycles"].to_numpy()  # type: ignore[index]
    rul = df["RUL"].to_numpy(float)  # type: ignore[index]
    values = df[feature_cols].to_numpy(float)  # type: ignore[index]
    endpoints = _sequence_endpoint_rows(unit, time, rul, window, endpoint_mode, seed, endpoints_per_engine)
    windows = np.stack([_raw_sequence_window(values, unit, time, int(row), window) for row in endpoints])
    return windows, rul[endpoints], endpoints


def _sequence_endpoint_rows(
    unit: np.ndarray,
    time: np.ndarray,
    rul: np.ndarray,
    window: int,
    endpoint_mode: str,
    seed: int,
    endpoints_per_engine: int,
) -> np.ndarray:
    endpoints: list[int] = []
    for engine in np.unique(unit):
        idx = np.flatnonzero(unit == engine)
        idx = idx[np.argsort(time[idx])]
        if endpoint_mode == "sliding":
            endpoints.extend(int(row) for row in idx[window - 1 :])
        elif endpoint_mode == "last":
            endpoints.append(int(idx[-1]))
        elif endpoint_mode == "pseudo":
            used: set[int] = set()
            for slot in range(endpoints_per_engine):
                target_rul = 10.0 + float((int(engine) * 37 + seed * 11 + slot * 29) % 125)
                row = int(idx[np.argmin(np.abs(rul[idx] - target_rul))])
                if row not in used:
                    endpoints.append(row)
                    used.add(row)
        else:
            raise ValueError(f"unknown endpoint_mode: {endpoint_mode}")
    return np.asarray(endpoints, dtype=int)


def _raw_sequence_window(
    values: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    endpoint_row: int,
    window: int,
) -> np.ndarray:
    engine_idx = np.flatnonzero(unit == unit[endpoint_row])
    engine_idx = engine_idx[np.argsort(time[engine_idx])]
    local_pos = int(np.flatnonzero(engine_idx == endpoint_row)[0])
    start = max(0, local_pos - window + 1)
    rows = engine_idx[start : local_pos + 1]
    segment = values[rows]
    if segment.shape[0] < window:
        pad = np.repeat(segment[:1], window - segment.shape[0], axis=0)
        segment = np.vstack([pad, segment])
    return segment.astype(np.float32)


def _sequence_window_operator_names(feature_cols: list[str]) -> list[str]:
    suffixes = [
        "last",
        "mean",
        "std",
        "min",
        "max",
        "delta",
        "slope",
        "recent_slope",
        "curvature",
        "tail_minus_head",
        "diff_energy",
        "last_minus_mean",
    ]
    return [f"{name}_win_{suffix}" for name in feature_cols for suffix in suffixes]


def _sequence_window_features(
    values: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    endpoint_row: int,
    window: int,
) -> np.ndarray:
    engine_idx = np.flatnonzero(unit == unit[endpoint_row])
    engine_idx = engine_idx[np.argsort(time[engine_idx])]
    local_pos = int(np.flatnonzero(engine_idx == endpoint_row)[0])
    start = max(0, local_pos - window + 1)
    rows = engine_idx[start : local_pos + 1]
    segment = values[rows]
    if segment.shape[0] < window:
        pad = np.repeat(segment[:1], window - segment.shape[0], axis=0)
        segment = np.vstack([pad, segment])
    first = segment[0]
    last = segment[-1]
    mean = segment.mean(axis=0)
    std = segment.std(axis=0)
    min_value = segment.min(axis=0)
    max_value = segment.max(axis=0)
    delta = last - first
    slope = _window_slope(segment)
    recent_slope = _window_slope(segment[-min(10, window) :])
    curvature = np.diff(segment, n=2, axis=0).mean(axis=0) if window >= 3 else np.zeros(segment.shape[1])
    half = max(1, window // 2)
    tail_minus_head = segment[-half:].mean(axis=0) - segment[:half].mean(axis=0)
    diff = np.diff(segment, axis=0)
    diff_energy = np.sqrt(np.mean(diff**2, axis=0))
    last_minus_mean = last - mean
    return np.column_stack(
        [
            last,
            mean,
            std,
            min_value,
            max_value,
            delta,
            slope,
            recent_slope,
            curvature,
            tail_minus_head,
            diff_energy,
            last_minus_mean,
        ]
    ).ravel()


def _window_slope(segment: np.ndarray) -> np.ndarray:
    t = np.arange(segment.shape[0], dtype=float)
    t = t - t.mean()
    denom = float(np.sum(t**2)) or 1.0
    return (t[:, None] * (segment - segment.mean(axis=0))).sum(axis=0) / denom


class _TemporalOperatorAttentionRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        hidden: int = 48,
        conv_channels: int = 4,
        kernels: tuple[int, ...] = (3, 7, 11),
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.conv_channels = conv_channels
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    n_features,
                    n_features * conv_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=n_features,
                )
                for kernel in kernels
            ]
        )
        token_in = 6 + 2 * conv_channels * len(kernels)
        self.token_proj = nn.Sequential(nn.Linear(token_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.self_attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.token_norm = nn.LayerNorm(hidden)
        self.query = nn.Parameter(torch.randn(hidden) / np.sqrt(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, 96),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(96, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        tokens = self._tokens(x)
        attended, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.token_norm(tokens + attended)
        weights = self._attention_from_tokens(tokens)
        context = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        pooled = torch.cat([context, tokens.mean(dim=1), tokens.amax(dim=1)], dim=1)
        return self.head(pooled).squeeze(-1)

    def feature_attention(self, x: Tensor) -> Tensor:
        tokens = self._tokens(x)
        attended, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.token_norm(tokens + attended)
        return self._attention_from_tokens(tokens)

    def _tokens(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.n_features:
            raise ValueError(f"x must have shape (batch, window, {self.n_features}), got {tuple(x.shape)}")
        bsz, window, _ = x.shape
        last = x[:, -1, :].transpose(0, 1).transpose(0, 1)
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        delta = x[:, -1, :] - x[:, 0, :]
        t = torch.arange(window, dtype=x.dtype, device=x.device)
        t = t - t.mean()
        denom = torch.clamp(torch.sum(t.square()), min=1e-12)
        slope = torch.sum(t.reshape(1, window, 1) * (x - mean.unsqueeze(1)), dim=1) / denom
        recent = x[:, -min(10, window) :, :]
        recent_t = torch.arange(recent.shape[1], dtype=x.dtype, device=x.device)
        recent_t = recent_t - recent_t.mean()
        recent_denom = torch.clamp(torch.sum(recent_t.square()), min=1e-12)
        recent_slope = torch.sum(
            recent_t.reshape(1, recent.shape[1], 1) * (recent - recent.mean(dim=1).unsqueeze(1)),
            dim=1,
        ) / recent_denom
        pieces = [torch.stack([last, mean, std, delta, slope, recent_slope], dim=-1)]
        xf = x.transpose(1, 2)
        for conv in self.convs:
            conv_out = torch.relu(conv(xf)).reshape(bsz, self.n_features, self.conv_channels, window)
            pieces.extend([conv_out.mean(dim=-1), conv_out.amax(dim=-1)])
        return self.token_proj(torch.cat(pieces, dim=-1))

    def _attention_from_tokens(self, tokens: Tensor) -> Tensor:
        score = torch.sum(tokens * self.query.reshape(1, 1, -1), dim=-1) / np.sqrt(tokens.shape[-1])
        return torch.softmax(score / 0.7, dim=1)


class _TemporalConvFairHybridRegressor(nn.Module):
    def __init__(self, n_window_features: int, n_side_features: int, *, channels: int = 64) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_window_features, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.side = nn.Sequential(
            nn.Linear(n_side_features, 96),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(channels + 64, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(channels + 64 + 1, 96),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(96, 1),
        )

    def forward(self, x_window: Tensor, x_side: Tensor) -> Tensor:
        temporal_tokens = self.temporal(x_window.transpose(1, 2))
        pooled = torch.cat([temporal_tokens[:, :, -1], temporal_tokens.mean(dim=-1)], dim=1)
        temporal = 0.5 * (pooled[:, : temporal_tokens.shape[1]] + pooled[:, temporal_tokens.shape[1] :])
        side = self.side(x_side)
        gate = self.fusion_gate(torch.cat([temporal, side], dim=1))
        fused_side = gate * side
        return self.head(torch.cat([temporal, fused_side, gate], dim=1)).squeeze(-1)


def _fit_sequence_tcn_hybrid_regressor(
    x_train: np.ndarray,
    side_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    side_val: np.ndarray,
    y_val: np.ndarray,
    y_val_standard: np.ndarray,
    x_test: np.ndarray,
    side_test: np.ndarray,
    feature_cols: list[str],
    side_names: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, float | str]]]:
    torch.manual_seed(seed + 404)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    side_train = np.asarray(side_train, dtype=np.float32)
    side_val = np.asarray(side_val, dtype=np.float32)
    side_test = np.asarray(side_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    y_val_standard = np.asarray(y_val_standard, dtype=float)

    selected_side = _screen_features(side_train, y_train, side_val, y_val, min(128, side_train.shape[1]))
    side_train = side_train[:, selected_side]
    side_val = side_val[:, selected_side]
    side_test = side_test[:, selected_side]
    selected_side_names = [side_names[int(idx)] for idx in selected_side]

    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_scale = np.where(x_train.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, x_train.std(axis=(0, 1), keepdims=True))
    side_mean = side_train.mean(axis=0)
    side_scale = np.where(side_train.std(axis=0) < 1e-6, 1.0, side_train.std(axis=0))
    xs_train = (x_train - x_mean) / x_scale
    xs_val = (x_val - x_mean) / x_scale
    xs_test = (x_test - x_mean) / x_scale
    ss_train = (side_train - side_mean) / side_scale
    ss_val = (side_val - side_mean) / side_scale
    ss_test = (side_test - side_mean) / side_scale
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    z_train = (y_train - y_mean) / y_scale
    z_val = (y_val - y_mean) / y_scale
    train_weight = _late_life_weight(y_train).astype(np.float32)
    train_weight = train_weight / max(float(train_weight.mean()), 1e-12)

    model = _TemporalConvFairHybridRegressor(x_train.shape[2], side_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    xtr = torch.as_tensor(xs_train, dtype=torch.float32)
    str_ = torch.as_tensor(ss_train, dtype=torch.float32)
    ytr = torch.as_tensor(z_train, dtype=torch.float32)
    wtr = torch.as_tensor(train_weight, dtype=torch.float32)
    xv = torch.as_tensor(xs_val, dtype=torch.float32)
    sv = torch.as_tensor(ss_val, dtype=torch.float32)
    yv = torch.as_tensor(z_val, dtype=torch.float32)
    n = xtr.shape[0]
    batch_size = min(1024, n)
    best_score = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history = {"val_selection_score": [], "val_rmse_z": []}

    for _ in range(180):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(xtr[idx], str_[idx])
            mse = torch.sum(wtr[idx] * (pred - ytr[idx]).square()) / torch.clamp(wtr[idx].sum(), min=1e-12)
            asymmetric = _torch_asymmetric_rul_loss(pred, ytr[idx], wtr[idx], y_scale=y_scale)
            loss = mse + 1e-2 * asymmetric
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred_z = model(xv, sv)
            val_rmse = float(torch.sqrt(torch.mean((val_pred_z - yv).square())).cpu())
            val_pred = _postprocess_rul_predictions(val_pred_z.cpu().numpy() * y_scale + y_mean)
        score = _array_endpoint_score(y_val, y_val_standard, val_pred)
        history["val_selection_score"].append(float(score))
        history["val_rmse_z"].append(val_rmse)
        if score < best_score - 1e-6:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 40:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = _postprocess_rul_predictions(model(xv, sv).cpu().numpy() * y_scale + y_mean)
        calibrator = _fit_array_endpoint_rul_calibrator(y_val_standard, val_pred)
        test_pred = model(
            torch.as_tensor(xs_test, dtype=torch.float32),
            torch.as_tensor(ss_test, dtype=torch.float32),
        ).cpu().numpy()
    test_pred = _apply_rul_calibrator(_postprocess_rul_predictions(test_pred * y_scale + y_mean), calibrator)
    importance = np.asarray([abs(_corr(side_train[:, idx], y_train)) for idx in range(side_train.shape[1])])
    ranked = np.argsort(-importance)[:12]
    selected = [
        {"name": selected_side_names[int(idx)], "importance": float(importance[int(idx)])}
        for idx in ranked
    ]
    return test_pred, {
        "best_validation_endpoint_selection": float(best_score),
        "best_val_rmse_z": float(min(history["val_rmse_z"])),
        "epochs_ran": len(history["val_rmse_z"]),
        "calibrator": calibrator,
        "window": int(x_train.shape[1]),
        "architecture": "TCN raw-window backbone fused with selected fair endpoint operators",
        "selected_side_features": selected_side_names,
    }, selected


def _fit_sequence_attention_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_val_standard: np.ndarray,
    x_test: np.ndarray,
    feature_cols: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, float | str]]]:
    torch.manual_seed(seed)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    y_val_standard = np.asarray(y_val_standard, dtype=float)
    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_scale = np.where(x_train.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, x_train.std(axis=(0, 1), keepdims=True))
    xs_train = (x_train - x_mean) / x_scale
    xs_val = (x_val - x_mean) / x_scale
    xs_test = (x_test - x_mean) / x_scale
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    z_train = (y_train - y_mean) / y_scale
    z_val = (y_val - y_mean) / y_scale
    train_weight = _late_life_weight(y_train).astype(np.float32)
    val_weight = _late_life_weight(y_val).astype(np.float32)
    train_weight = train_weight / max(float(train_weight.mean()), 1e-12)
    val_weight = val_weight / max(float(val_weight.mean()), 1e-12)

    model = _TemporalOperatorAttentionRegressor(len(feature_cols))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    xtr = torch.as_tensor(xs_train, dtype=torch.float32)
    ytr = torch.as_tensor(z_train, dtype=torch.float32)
    wtr = torch.as_tensor(train_weight, dtype=torch.float32)
    xv = torch.as_tensor(xs_val, dtype=torch.float32)
    yv = torch.as_tensor(z_val, dtype=torch.float32)
    n = xtr.shape[0]
    batch_size = min(1024, n)
    best_score = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history = {"val_selection_score": [], "val_rmse_z": []}

    for _ in range(160):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(xtr[idx])
            mse = torch.sum(wtr[idx] * (pred - ytr[idx]).square()) / torch.clamp(wtr[idx].sum(), min=1e-12)
            asymmetric = _torch_asymmetric_rul_loss(pred, ytr[idx], wtr[idx], y_scale=y_scale)
            loss = mse + 1e-2 * asymmetric
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred_z = model(xv)
            val_rmse = float(torch.sqrt(torch.mean((val_pred_z - yv).square())).cpu())
            val_pred = _postprocess_rul_predictions(val_pred_z.cpu().numpy() * y_scale + y_mean)
        score = _array_endpoint_score(y_val, y_val_standard, val_pred)
        history["val_selection_score"].append(float(score))
        history["val_rmse_z"].append(val_rmse)
        if score < best_score - 1e-6:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 35:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = _postprocess_rul_predictions(model(xv).cpu().numpy() * y_scale + y_mean)
        calibrator = _fit_array_endpoint_rul_calibrator(y_val_standard, val_pred)
        test_pred = model(torch.as_tensor(xs_test, dtype=torch.float32)).cpu().numpy() * y_scale + y_mean
        test_pred = _apply_rul_calibrator(_postprocess_rul_predictions(test_pred), calibrator)
        attention = model.feature_attention(xv).mean(dim=0).cpu().numpy()
    ranked = np.argsort(-attention)[:12]
    selected = [{"name": feature_cols[int(idx)], "attention": float(attention[int(idx)])} for idx in ranked]
    return test_pred, {
        "best_validation_endpoint_selection": float(best_score),
        "best_val_rmse_z": float(min(history["val_rmse_z"])),
        "epochs_ran": len(history["val_rmse_z"]),
        "calibrator": calibrator,
        "window": int(x_train.shape[1]),
        "architecture": "depthwise temporal convolutions + self-attention over sensor/operator tokens",
    }, selected


def _torch_asymmetric_rul_loss(pred: Tensor, target: Tensor, weight: Tensor, *, y_scale: float) -> Tensor:
    diff_cycles = torch.clamp((pred - target) * float(y_scale), min=-80.0, max=80.0)
    penalties = torch.where(diff_cycles < 0.0, torch.expm1(-diff_cycles / 13.0), torch.expm1(diff_cycles / 10.0))
    return torch.sum(weight * penalties) / torch.clamp(weight.sum(), min=1e-12)


def _array_endpoint_score(y_val: np.ndarray, y_val_standard: np.ndarray, pred: np.ndarray) -> float:
    weight = _late_life_weight(np.minimum(np.asarray(y_val_standard, dtype=float), TARGET_CAP))
    calibrator = _fit_rul_calibrator(y_val_standard, pred, weight)
    calibrated = _apply_rul_calibrator(pred, calibrator)
    row_rmse = float(np.sqrt(np.mean((pred - np.asarray(y_val, dtype=float)) ** 2)))
    return (
        _weighted_nasa_score(y_val_standard, calibrated, weight)
        + 0.01 * _weighted_rmse(y_val_standard, calibrated, weight)
        + 0.02 * row_rmse
    )


def _screen_features(
    train: np.ndarray,
    y_train: np.ndarray,
    val: np.ndarray,
    y_val: np.ndarray,
    max_features: int,
) -> np.ndarray:
    values = np.concatenate([train, val], axis=0)
    target = np.concatenate([y_train, y_val], axis=0)
    scores = np.asarray([abs(_corr(values[:, idx], target)) for idx in range(values.shape[1])])
    stable = np.asarray([np.isfinite(values[:, idx]).all() and np.nanstd(values[:, idx]) > 1e-8 for idx in range(values.shape[1])])
    scores = np.where(stable, scores, -np.inf)
    return np.argsort(scores)[-max_features:]


def _fit_tuned_ridge_cycles(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    *,
    refit: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    pred_z, details = _fit_tuned_ridge_z(
        x_train,
        (y_train - y_mean) / y_scale,
        x_val,
        (y_val - y_mean) / y_scale,
        x_test,
        refit=refit,
    )
    return _postprocess_rul_predictions(pred_z * y_scale + y_mean), details


def _fit_tuned_ridge_z(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_val: np.ndarray,
    z_val: np.ndarray,
    x_test: np.ndarray,
    *,
    refit: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
    x_mean = x_train.mean(axis=0)
    x_scale = np.where(x_train.std(axis=0) < 1e-9, 1.0, x_train.std(axis=0))
    xtr = (x_train - x_mean) / x_scale
    xv = (x_val - x_mean) / x_scale
    best = None
    for alpha in alphas:
        pred = _ridge_predict(xtr, z_train, xv, alpha)
        rmse = float(np.sqrt(np.mean((pred - z_val) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, alpha)
    assert best is not None
    if refit:
        x_fit = np.concatenate([x_train, x_val], axis=0)
        z_fit = np.concatenate([z_train, z_val], axis=0)
        fit_mean = x_fit.mean(axis=0)
        fit_scale = np.where(x_fit.std(axis=0) < 1e-9, 1.0, x_fit.std(axis=0))
    else:
        x_fit = x_train
        z_fit = z_train
        fit_mean = x_mean
        fit_scale = x_scale
    pred = _ridge_predict((x_fit - fit_mean) / fit_scale, z_fit, (x_test - fit_mean) / fit_scale, best[1])
    return pred, {"alpha": float(best[1]), "validation_rmse_z": float(best[0])}


def _fit_endpoint_tuned_array_ridge_cycles(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_val_standard: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    z_train = (y_train - y_mean) / y_scale
    z_val = (y_val - y_mean) / y_scale
    train_s, val_s, _, _ = _standardize_train_apply(x_train, x_val, x_val)
    weight = _late_life_weight(np.minimum(np.asarray(y_val_standard, dtype=float), TARGET_CAP))
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 300.0]
    best: tuple[float, float, dict[str, float]] | None = None
    for alpha in alphas:
        pred_val = _postprocess_rul_predictions(_ridge_predict(train_s, z_train, val_s, alpha) * y_scale + y_mean)
        calibrator = _fit_rul_calibrator(y_val_standard, pred_val, weight)
        calibrated = _apply_rul_calibrator(pred_val, calibrator)
        endpoint_score = _weighted_nasa_score(y_val_standard, calibrated, weight)
        endpoint_rmse = _weighted_rmse(y_val_standard, calibrated, weight)
        row_rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        score = endpoint_score + 0.01 * endpoint_rmse + 0.005 * row_rmse
        if best is None or score < best[0]:
            best = (
                float(score),
                float(alpha),
                calibrator
                | {
                    "validation_endpoint_nasa": float(endpoint_score),
                    "validation_endpoint_rmse": float(endpoint_rmse),
                    "validation_row_rmse": row_rmse,
                },
            )
    assert best is not None
    _, alpha, calibrator = best
    x_fit = np.concatenate([x_train, x_val], axis=0)
    z_fit = np.concatenate([z_train, z_val], axis=0)
    fit_s, _, test_s, _ = _standardize_train_apply(x_fit, x_fit, x_test)
    pred = _ridge_predict(fit_s, z_fit, test_s, alpha) * y_scale + y_mean
    pred = _apply_rul_calibrator(_postprocess_rul_predictions(pred), calibrator)
    return pred, {
        "alpha": float(alpha),
        "calibrator": calibrator,
        "selection_objective": "validation sequence-window endpoint NASA score",
    }


def _fit_endpoint_selected_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_val_standard: np.ndarray,
    val_df: object,
    x_test: np.ndarray,
    ranked_indices: list[int],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    z_train = (y_train - y_mean) / y_scale
    endpoint_mask = _pseudo_endpoint_mask(val_df, y_val_standard, seed=seed)
    endpoint_weight = _late_life_weight(np.minimum(np.asarray(y_val_standard, dtype=float)[endpoint_mask], TARGET_CAP))
    alphas = [1e-4, 1e-2, 1.0, 10.0, 100.0]
    budgets = [8, 16, 32, 64, 128, 256, 512, min(1024, x_train.shape[1])]
    best: tuple[float, int, float, dict[str, float]] | None = None
    for budget in budgets:
        selected = np.asarray(ranked_indices[: min(budget, len(ranked_indices))], dtype=int)
        if selected.size == 0:
            continue
        train_s, val_s, _, _ = _standardize_train_apply(x_train[:, selected], x_val[:, selected], x_val[:, selected])
        for alpha in alphas:
            pred_val = _ridge_predict(train_s, z_train, val_s, alpha) * y_scale + y_mean
            pred_val = _postprocess_rul_predictions(pred_val)
            calibrator = _fit_rul_calibrator(
                np.asarray(y_val_standard, dtype=float)[endpoint_mask],
                pred_val[endpoint_mask],
                endpoint_weight,
            )
            calibrated = _apply_rul_calibrator(pred_val[endpoint_mask], calibrator)
            endpoint_score = _weighted_nasa_score(
                np.asarray(y_val_standard, dtype=float)[endpoint_mask],
                calibrated,
                endpoint_weight,
            )
            endpoint_rmse = _weighted_rmse(
                np.asarray(y_val_standard, dtype=float)[endpoint_mask],
                calibrated,
                endpoint_weight,
            )
            score = endpoint_score + 0.01 * endpoint_rmse + 0.0005 * selected.size
            if best is None or score < best[0]:
                best = (
                    float(score),
                    int(selected.size),
                    float(alpha),
                    calibrator
                    | {
                        "validation_endpoint_nasa": float(endpoint_score),
                        "validation_endpoint_rmse": float(endpoint_rmse),
                    },
                )
    assert best is not None
    _, selected_count, alpha, calibrator = best
    selected = np.asarray(ranked_indices[:selected_count], dtype=int)
    x_fit = np.concatenate([x_train[:, selected], x_val[:, selected]], axis=0)
    y_fit = np.concatenate([z_train, (y_val - y_mean) / y_scale], axis=0)
    fit_s, _, test_s, _ = _standardize_train_apply(x_fit, x_fit, x_test[:, selected])
    pred = _ridge_predict(fit_s, y_fit, test_s, alpha) * y_scale + y_mean
    pred = _apply_rul_calibrator(_postprocess_rul_predictions(pred), calibrator)
    details: dict[str, object] = {
        "selected_count": int(selected_count),
        "alpha": float(alpha),
        "calibrator": calibrator,
        "selection_objective": "validation pseudo-endpoint NASA score over joint-gate-ranked fair dictionary",
    }
    return pred, details


def _fit_endpoint_tuned_ridge_cycles(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_val_standard: np.ndarray,
    val_df: object,
    x_test: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    z_train = (y_train - y_mean) / y_scale
    z_val = (y_val - y_mean) / y_scale
    train_s, val_s, _, _ = _standardize_train_apply(x_train, x_val, x_val)
    endpoint_mask = _pseudo_endpoint_mask(val_df, y_val_standard, seed=seed)
    endpoint_weight = _late_life_weight(np.minimum(np.asarray(y_val_standard, dtype=float)[endpoint_mask], TARGET_CAP))
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 300.0]
    best: tuple[float, float, dict[str, float]] | None = None
    for alpha in alphas:
        pred_val = _ridge_predict(train_s, z_train, val_s, alpha) * y_scale + y_mean
        pred_val = _postprocess_rul_predictions(pred_val)
        calibrator = _fit_rul_calibrator(
            np.asarray(y_val_standard, dtype=float)[endpoint_mask],
            pred_val[endpoint_mask],
            endpoint_weight,
        )
        calibrated = _apply_rul_calibrator(pred_val[endpoint_mask], calibrator)
        endpoint_score = _weighted_nasa_score(
            np.asarray(y_val_standard, dtype=float)[endpoint_mask],
            calibrated,
            endpoint_weight,
        )
        endpoint_rmse = _weighted_rmse(
            np.asarray(y_val_standard, dtype=float)[endpoint_mask],
            calibrated,
            endpoint_weight,
        )
        row_rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        score = endpoint_score + 0.01 * endpoint_rmse + 0.005 * row_rmse
        if best is None or score < best[0]:
            best = (
                float(score),
                float(alpha),
                calibrator
                | {
                    "validation_endpoint_nasa": float(endpoint_score),
                    "validation_endpoint_rmse": float(endpoint_rmse),
                    "validation_row_rmse": row_rmse,
                },
            )
    assert best is not None
    _, alpha, calibrator = best
    x_fit = np.concatenate([x_train, x_val], axis=0)
    z_fit = np.concatenate([z_train, z_val], axis=0)
    fit_s, _, test_s, _ = _standardize_train_apply(x_fit, x_fit, x_test)
    pred = _ridge_predict(fit_s, z_fit, test_s, alpha) * y_scale + y_mean
    pred = _apply_rul_calibrator(_postprocess_rul_predictions(pred), calibrator)
    return pred, {
        "alpha": float(alpha),
        "calibrator": calibrator,
        "selection_objective": "validation pseudo-endpoint NASA score over full fair dictionary",
    }


def _standardize_train_apply(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean = train.mean(axis=0)
    scale = np.where(train.std(axis=0) < 1e-9, 1.0, train.std(axis=0))
    return (
        (train - mean) / scale,
        (val - mean) / scale,
        (test - mean) / scale,
        {"mean": mean, "scale": scale},
    )


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    reg = alpha * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y_train)
    return np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1) @ coef


def _row_and_standard_metrics(
    y_test: np.ndarray,
    y_test_standard: np.ndarray,
    pred: np.ndarray,
    test_df: object,
) -> dict[str, object]:
    unit = test_df["unit_number"].to_numpy()  # type: ignore[index]
    time = test_df["time_cycles"].to_numpy()  # type: ignore[index]
    err = pred - y_test
    return {
        "row_metrics": {
            "rmse_cycles": float(np.sqrt(np.mean(err**2))),
            "mae_cycles": float(np.mean(np.abs(err))),
        },
        "standard_last_cycle": _standard_last_cycle_metrics(y_test_standard, pred, unit, time),
    }


def _training_weight(df: object, y: np.ndarray) -> np.ndarray:
    weight = _late_life_weight(y)
    return weight * (1.0 + 4.0 * _last_cycle_mask(df))


def _calibration_weight(df: object, y: np.ndarray) -> np.ndarray:
    weight = _late_life_weight(y)
    return weight * (1.0 + 2.0 * _last_cycle_mask(df))


def _last_cycle_mask(df: object) -> np.ndarray:
    unit = df["unit_number"].to_numpy()  # type: ignore[index]
    time = df["time_cycles"].to_numpy()  # type: ignore[index]
    mask = np.zeros(unit.shape[0], dtype=float)
    for engine in np.unique(unit):
        idx = np.flatnonzero(unit == engine)
        mask[int(idx[np.argmax(time[idx])])] = 1.0
    return mask


def _fit_endpoint_rul_calibrator(
    df: object,
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    mask = _pseudo_endpoint_mask(df, y_true, seed=seed)
    weight = _late_life_weight(np.minimum(np.asarray(y_true, dtype=float)[mask], TARGET_CAP))
    calibrator = _fit_rul_calibrator(np.asarray(y_true, dtype=float)[mask], np.asarray(pred, dtype=float)[mask], weight)
    calibrator["n_calibration_endpoints"] = float(mask.sum())
    return calibrator


def _fit_multi_endpoint_rul_calibrator(
    df: object,
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    seed: int,
    n_per_engine: int = 5,
) -> dict[str, float]:
    idx = _pseudo_endpoint_indices(df, y_true, seed=seed, n_per_engine=n_per_engine)
    y_endpoint = np.asarray(y_true, dtype=float)[idx]
    weight = _late_life_weight(np.minimum(y_endpoint, TARGET_CAP))
    calibrator = _fit_rul_calibrator(y_endpoint, np.asarray(pred, dtype=float)[idx], weight)
    calibrator["n_calibration_endpoints"] = float(idx.size)
    return calibrator


def _fit_array_endpoint_rul_calibrator(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    weight = _late_life_weight(np.minimum(y_true, TARGET_CAP))
    calibrator = _fit_rul_calibrator(y_true, pred, weight)
    calibrator["n_calibration_endpoints"] = float(y_true.size)
    return calibrator


def _array_endpoint_nasa_selection_metric(y_true_standard: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
    y_true_standard = np.asarray(y_true_standard, dtype=float)
    weight = _late_life_weight(np.minimum(y_true_standard, TARGET_CAP))

    def metric(y_true: np.ndarray, pred: np.ndarray) -> float:
        pred = _postprocess_rul_predictions(np.asarray(pred, dtype=float))
        calibrator = _fit_rul_calibrator(y_true_standard, pred, weight)
        calibrated = _apply_rul_calibrator(pred, calibrator)
        row_rmse = float(np.sqrt(np.mean((pred - np.asarray(y_true, dtype=float)) ** 2)))
        return (
            _weighted_nasa_score(y_true_standard, calibrated, weight)
            + 0.01 * _weighted_rmse(y_true_standard, calibrated, weight)
            + 0.02 * row_rmse
        )

    return metric


def _endpoint_nasa_selection_metric(
    df: object,
    y_true_standard: np.ndarray,
    *,
    seed: int,
) -> Callable[[np.ndarray, np.ndarray], float]:
    idx = _pseudo_endpoint_indices(df, y_true_standard, seed=seed, n_per_engine=5)
    endpoint_true = np.asarray(y_true_standard, dtype=float)[idx]
    endpoint_weight = _late_life_weight(np.minimum(endpoint_true, TARGET_CAP))

    def metric(y_true: np.ndarray, pred: np.ndarray) -> float:
        pred = _postprocess_rul_predictions(np.asarray(pred, dtype=float))
        calibrator = _fit_rul_calibrator(endpoint_true, pred[idx], endpoint_weight)
        calibrated = _apply_rul_calibrator(pred[idx], calibrator)
        row_rmse = float(np.sqrt(np.mean((pred - np.asarray(y_true, dtype=float)) ** 2)))
        return (
            _weighted_nasa_score(endpoint_true, calibrated, endpoint_weight)
            + 0.01 * _weighted_rmse(endpoint_true, calibrated, endpoint_weight)
            + 0.02 * row_rmse
        )

    return metric


def _pseudo_endpoint_indices(
    df: object,
    y_true: np.ndarray,
    *,
    seed: int,
    n_per_engine: int,
) -> np.ndarray:
    unit = df["unit_number"].to_numpy()  # type: ignore[index]
    y_true = np.asarray(y_true, dtype=float)
    chosen: list[int] = []
    for engine in np.unique(unit):
        engine_idx = np.flatnonzero(unit == engine)
        used: set[int] = set()
        for slot in range(n_per_engine):
            target_rul = 10.0 + float((int(engine) * 37 + seed * 11 + slot * 29) % 125)
            local = int(engine_idx[np.argmin(np.abs(y_true[engine_idx] - target_rul))])
            if local not in used:
                chosen.append(local)
                used.add(local)
    return np.asarray(chosen, dtype=int)


def _pseudo_endpoint_mask(df: object, y_true: np.ndarray, *, seed: int) -> np.ndarray:
    unit = df["unit_number"].to_numpy()  # type: ignore[index]
    y_true = np.asarray(y_true, dtype=float)
    mask = np.zeros(unit.shape[0], dtype=bool)
    for engine in np.unique(unit):
        idx = np.flatnonzero(unit == engine)
        # Deterministic pseudo-test truncation: choose one endpoint per
        # validation engine with RUL in the same broad range as FD001 test
        # targets, instead of using the true failure row (RUL=0).
        target_rul = 10.0 + float((int(engine) * 37 + seed * 11) % 125)
        local = idx[np.argmin(np.abs(y_true[idx] - target_rul))]
        mask[int(local)] = True
    return mask


def _fit_rul_calibrator(y_true: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    weight = np.asarray(weight, dtype=float)
    best: tuple[float, float, float] | None = None
    for scale in np.linspace(0.72, 1.10, 20):
        for offset in np.linspace(-28.0, 8.0, 37):
            calibrated = _postprocess_rul_predictions(scale * pred + offset)
            score = _weighted_nasa_score(y_true, calibrated, weight) + 0.01 * _weighted_rmse(y_true, calibrated, weight)
            if best is None or score < best[0]:
                best = (score, float(scale), float(offset))
    assert best is not None
    calibrated = _postprocess_rul_predictions(best[1] * pred + best[2])
    return {
        "scale": best[1],
        "offset": best[2],
        "validation_weighted_nasa": _weighted_nasa_score(y_true, calibrated, weight),
        "validation_weighted_rmse": _weighted_rmse(y_true, calibrated, weight),
    }


def _apply_rul_calibrator(pred: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    return _postprocess_rul_predictions(calibrator["scale"] * np.asarray(pred, dtype=float) + calibrator["offset"])


def _weighted_nasa_score(y_true: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(diff < 0.0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    weight = np.asarray(weight, dtype=float)
    return float(np.sum(weight * penalties) / max(float(weight.sum()), 1e-12))


def _weighted_rmse(y_true: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    err = np.asarray(pred, dtype=float) - np.asarray(y_true, dtype=float)
    weight = np.asarray(weight, dtype=float)
    return float(np.sqrt(np.sum(weight * err**2) / max(float(weight.sum()), 1e-12)))


def _rename_selected_operators(selected: list[dict[str, object]], input_names: list[str]) -> list[dict[str, object]]:
    renamed = []
    for item in selected:
        row = dict(item)
        name = str(row["name"])
        for idx in range(len(input_names), 0, -1):
            name = name.replace(f"x{idx}", input_names[idx - 1])
        row["name"] = name
        renamed.append(row)
    return renamed


def _late_life_weight(y: np.ndarray) -> np.ndarray:
    return 1.0 + 4.0 * (1.0 - np.asarray(y, dtype=float) / TARGET_CAP)


def _zscore(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    scale = float(reference.std()) or 1.0
    return (y - float(reference.mean())) / scale


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
