# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Real-world C-MAPSS feature-discovery benchmark.

This benchmark asks whether omnibias can act as a feature/model-selection engine
on NASA turbofan degradation data. It uses no test leakage:

* train engines: fit omnibias field and discover transformations,
* validation engines: tune model hyperparameters,
* held-out validation parquet: final test metrics only.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from examples.symbolic_discovery.synthetic_feature_discovery.benchmark import (
    DiscoveredFeature,
    build_design_matrix,
    discover_features_from_derivatives,
    field_value_grad_hessian,
    fit_omnibias_field,
    full_generic_design_matrix,
    raw_design_matrix,
)

TARGET_CAP = 125.0
HF_REPO = "Samvik/nasa_turbofan_degradation_FD001"


@dataclass(frozen=True)
class CandidateFeature:
    name: str
    kind: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    complexity: float = 1.0
    order: int = 0


@dataclass(frozen=True)
class SelectedFeature:
    name: str
    kind: str
    validation_rmse_z: float
    selector: str = "greedy"
    probability: float = 1.0
    complexity: float = 1.0


def ensure_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    if list((data_dir / "data").glob("*.parquet")):
        return data_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install `huggingface_hub` or place C-MAPSS parquet files under data/.") from exc
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(data_dir))
    return data_dir


def load_splits(data_dir: Path, seed: int = 0) -> dict[str, object]:
    ensure_dataset(data_dir)
    parquet_dir = data_dir / "data"
    train_path = next(parquet_dir.glob("train-*.parquet"))
    test_path = next(parquet_dir.glob("valid-*.parquet"))
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    feature_cols = [
        c
        for c in train.columns
        if c not in {"unit_number", "RUL"} and float(train[c].std()) > 1e-9
    ]
    units = np.asarray(sorted(train["unit_number"].unique()))
    model_units, tune_units = train_test_split(units, test_size=0.2, random_state=seed)
    field_units, selector_units = train_test_split(model_units, test_size=0.25, random_state=seed + 17)
    field_train_df = train[train["unit_number"].isin(set(field_units))].copy()
    selector_val_df = train[train["unit_number"].isin(set(selector_units))].copy()
    model_train_df = train[train["unit_number"].isin(set(model_units))].copy()
    tune_val_df = train[train["unit_number"].isin(set(tune_units))].copy()
    return {
        "field_train_df": field_train_df,
        "selector_val_df": selector_val_df,
        "model_train_df": model_train_df,
        "tune_val_df": tune_val_df,
        # Backward-compatible aliases for callers/tests that still expect train/val.
        "train_df": model_train_df,
        "val_df": tune_val_df,
        "test_df": test,
        "feature_cols": feature_cols,
    }


def evaluate_benchmark(
    data_dir: Path,
    out_dir: Path,
    *,
    hidden: int = 512,
    seed: int = 0,
    max_field_rows: int = 6000,
    max_discovered_features: int = 20,
    max_selected_features: int = 25,
    selector: str = "gumbel",
    budget_grid: tuple[int, ...] = (5, 10, 20, 40, 80),
) -> dict[str, object]:
    split = load_splits(data_dir, seed=seed)
    field_train_df: pd.DataFrame = split["field_train_df"]  # type: ignore[assignment]
    selector_val_df: pd.DataFrame = split["selector_val_df"]  # type: ignore[assignment]
    train_df: pd.DataFrame = split["model_train_df"]  # type: ignore[assignment]
    val_df: pd.DataFrame = split["tune_val_df"]  # type: ignore[assignment]
    test_df: pd.DataFrame = split["test_df"]  # type: ignore[assignment]
    feature_cols: list[str] = split["feature_cols"]  # type: ignore[assignment]

    x_field = field_train_df[feature_cols].to_numpy(float)
    y_field = np.minimum(field_train_df["RUL"].to_numpy(float), TARGET_CAP)
    field_mean = float(y_field.mean())
    field_scale = float(y_field.std())
    z_field = (y_field - field_mean) / field_scale
    x_train = train_df[feature_cols].to_numpy(float)
    x_val = val_df[feature_cols].to_numpy(float)
    x_test = test_df[feature_cols].to_numpy(float)
    y_train = np.minimum(train_df["RUL"].to_numpy(float), TARGET_CAP)
    y_val = np.minimum(val_df["RUL"].to_numpy(float), TARGET_CAP)
    y_test = np.minimum(test_df["RUL"].to_numpy(float), TARGET_CAP)
    y_test_standard = test_df["RUL"].to_numpy(float)

    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    z_train = (y_train - y_mean) / y_scale
    z_val = (y_val - y_mean) / y_scale

    field_idx = _even_sample_indices(x_field.shape[0], max_field_rows)
    field = fit_omnibias_field(x_field[field_idx], z_field[field_idx], hidden=hidden, ridge=1e-5, seed=seed)
    _, grad, hess = field_value_grad_hessian(field, x_field[field_idx])
    discovered_raw = discover_features_from_derivatives(
        x_field[field_idx],
        grad,
        hess,
        curvature_threshold=0.5,
        interaction_threshold=0.5,
        periodic_threshold=0.85,
    )
    discovered = discovered_raw[:max_discovered_features]

    functional_train, functional_val, functional_test, functional_names, selected, selection_details = _select_functional_features(
        field_train_df,
        selector_val_df,
        train_df,
        val_df,
        test_df,
        feature_cols,
        z_field,
        (np.minimum(selector_val_df["RUL"].to_numpy(float), TARGET_CAP) - field_mean) / field_scale,
        z_train,
        z_val,
        discovered,
        field,
        max_selected_features=max_selected_features,
        selector=selector,
    )
    fair_train, fair_val, fair_test, fair_names = _fair_operator_dictionary_matrices(
        train_df,
        val_df,
        test_df,
        feature_cols,
        discovered,
        field,
    )
    budget_results = _operator_budget_curve(
        train_df,
        val_df,
        test_df,
        feature_cols,
        y_test,
        test_df["unit_number"].to_numpy(),
        z_train,
        z_val,
        y_mean,
        y_scale,
        discovered,
        field,
        selected,
        budgets=budget_grid,
    )

    matrix_sets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list[str]]] = []
    builders: list[tuple[str, Callable[[np.ndarray], tuple[np.ndarray, list[str]]]]] = [
        ("raw", raw_design_matrix),
        ("omnibias_derivative_magnitude", lambda x: build_design_matrix(x, discovered)),
        ("generic_dictionary", full_generic_design_matrix),
    ]
    for feature_set, builder in builders:
        xtr, names = builder(x_train)
        xv, _ = builder(x_val)
        xte, _ = builder(x_test)
        matrix_sets.append((feature_set, xtr, xv, xte, _pretty_feature_names(names, feature_cols)))
    matrix_sets.append(
        (
            "omnibias_functional_selected",
            functional_train,
            functional_val,
            functional_test,
            functional_names,
        )
    )
    matrix_sets.append(("fair_operator_dictionary", fair_train, fair_val, fair_test, fair_names))

    models: dict[str, object] = {}
    for feature_set, xtr, xv, xte, names in matrix_sets:
        pred_ridge, ridge_details = _fit_tuned_ridge_cycles(
            xtr, z_train, xv, z_val, xte, y_mean=y_mean, y_scale=y_scale
        )
        models[f"{feature_set}_ridge"] = _metrics(y_test, pred_ridge, test_df["unit_number"].to_numpy()) | {
            "standard_last_cycle": _standard_last_cycle_metrics(
                y_test_standard,
                pred_ridge,
                test_df["unit_number"].to_numpy(),
                test_df["time_cycles"].to_numpy(),
            ),
            "n_features": len(names),
            "features": names,
            "details": ridge_details,
        }
        tree = _fit_optional_hgb_cycles(xtr, z_train, xv, z_val, xte, y_mean=y_mean, y_scale=y_scale)
        if tree is not None:
            pred_tree, tree_details = tree
            models[f"{feature_set}_hist_gradient_boosting"] = _metrics(
                y_test, pred_tree, test_df["unit_number"].to_numpy()
            ) | {
                "standard_last_cycle": _standard_last_cycle_metrics(
                    y_test_standard,
                    pred_tree,
                    test_df["unit_number"].to_numpy(),
                    test_df["time_cycles"].to_numpy(),
                ),
                "n_features": len(names),
                "features": names,
                "details": tree_details,
            }

    discovered_payload = [
        {
            "name": _pretty_feature_name(feature.name, feature_cols),
            "raw_name": feature.name,
            "kind": feature.kind,
            "indices": feature.indices,
            "score": feature.score,
        }
        for feature in discovered
    ]
    results = {
        "dataset": "NASA C-MAPSS FD001 turbofan degradation",
        "target": f"RUL clipped at {TARGET_CAP:g} cycles",
        "fairness_protocol": {
            "field_fit_and_candidate_generation_split": "field-train engines only",
            "operator_selection_split": "selection-validation engines only",
            "model_tuning_split": "tuning-validation engines only",
            "final_scoring_split": "held-out validation parquet only",
        },
        "n_rows": {
            "field_train": int(field_train_df.shape[0]),
            "operator_selection_validation": int(selector_val_df.shape[0]),
            "model_train": int(train_df.shape[0]),
            "model_tuning_validation": int(val_df.shape[0]),
            "test": int(test_df.shape[0]),
        },
        "n_engines": {
            "field_train": int(field_train_df["unit_number"].nunique()),
            "operator_selection_validation": int(selector_val_df["unit_number"].nunique()),
            "model_train": int(train_df["unit_number"].nunique()),
            "model_tuning_validation": int(val_df["unit_number"].nunique()),
            "test": int(test_df["unit_number"].nunique()),
        },
        "feature_columns": feature_cols,
        "discovered_features": discovered_payload,
        "selected_functional_features": [
            {
                "name": feature.name,
                "kind": feature.kind,
                "validation_rmse_z_after_add": feature.validation_rmse_z,
                "selector": feature.selector,
                "probability": feature.probability,
                "complexity": feature.complexity,
            }
            for feature in selected
        ],
        "operator_selection_details": selection_details,
        "operator_budget": budget_results,
        "models": models,
        "feature_gain": _feature_gain(models),
        "standard_cmapss_protocol": _standard_protocol_payload(models),
        "literature_context": _literature_context(),
        "significance": _significance_dict(models, y_test, test_df["unit_number"].to_numpy()),
        "seed": seed,
    }
    write_artifacts(results, out_dir)
    return results


def evaluate_repeated_benchmark(
    data_dir: Path,
    out_dir: Path,
    *,
    n_repeats: int = 3,
    seed: int = 0,
    datasets: tuple[str, ...] = ("fd001",),
    **kwargs: object,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for dataset in datasets:
        dataset_dir = data_dir if dataset.lower() == "fd001" else data_dir.parent / f"cmapss_{dataset.lower()}"
        if not (dataset_dir / "data").exists() and dataset.lower() != "fd001":
            all_results.append({"dataset": dataset, "status": "skipped_missing_data", "data_dir": str(dataset_dir)})
            continue
        for offset in range(n_repeats):
            run_seed = seed + offset
            run_out = out_dir / dataset.lower() / f"seed_{run_seed}"
            result = evaluate_benchmark(dataset_dir, run_out, seed=run_seed, **kwargs)
            all_results.append({"dataset": dataset, "seed": run_seed, "status": "completed", "result": result})
    summary = _repeated_summary(all_results)
    (out_dir / "repeated_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "repeated_significance.json").write_text(
        json.dumps(summary.get("significance", {}), indent=2, sort_keys=True)
    )
    (out_dir / "standard_repeated_significance.json").write_text(
        json.dumps(summary.get("standard_significance", {}), indent=2, sort_keys=True)
    )
    _write_repeated_report(summary, out_dir)
    return summary


def _repeated_summary(all_results: list[dict[str, object]]) -> dict[str, object]:
    completed = [item for item in all_results if item.get("status") == "completed"]
    rows = []
    for item in completed:
        result = item["result"]
        assert isinstance(result, dict)
        models = result["models"]
        assert isinstance(models, dict)
        for model_name, metrics in models.items():
            if isinstance(metrics, dict):
                rows.append(
                    {
                        "dataset": item["dataset"],
                        "seed": item["seed"],
                        "model": model_name,
                        "rmse_cycles": float(metrics["rmse_cycles"]),
                        "mae_cycles": float(metrics["mae_cycles"]),
                        "n_features": int(metrics["n_features"]),
                    }
                    | _standard_run_metrics(metrics)
                )
    aggregates: dict[str, dict[str, float]] = {}
    for model in sorted({row["model"] for row in rows}):
        vals = [row for row in rows if row["model"] == model]
        aggregates[str(model)] = {
            "mean_rmse_cycles": float(np.mean([row["rmse_cycles"] for row in vals])),
            "std_rmse_cycles": float(np.std([row["rmse_cycles"] for row in vals])),
            "mean_mae_cycles": float(np.mean([row["mae_cycles"] for row in vals])),
            "mean_n_features": float(np.mean([row["n_features"] for row in vals])),
        }
        standard_vals = [row for row in vals if "standard_rmse_cycles" in row]
        if standard_vals:
            aggregates[str(model)] |= {
                "standard_mean_rmse_cycles": float(np.mean([row["standard_rmse_cycles"] for row in standard_vals])),
                "standard_std_rmse_cycles": float(np.std([row["standard_rmse_cycles"] for row in standard_vals])),
                "standard_mean_mae_cycles": float(np.mean([row["standard_mae_cycles"] for row in standard_vals])),
                "standard_mean_nasa_score": float(np.mean([row["standard_nasa_score"] for row in standard_vals])),
            }
    significance = _paired_repeated_significance(rows, "omnibias_functional_selected_hist_gradient_boosting")
    standard_significance = _paired_repeated_significance(
        rows,
        "omnibias_functional_selected_hist_gradient_boosting",
        metric="standard_rmse_cycles",
    )
    return {
        "runs": rows,
        "aggregates": aggregates,
        "significance": significance,
        "standard_significance": standard_significance,
        "raw_status": all_results,
    }


def _standard_run_metrics(metrics: dict[str, object]) -> dict[str, float]:
    standard = metrics.get("standard_last_cycle")
    if not isinstance(standard, dict):
        return {}
    return {
        "standard_rmse_cycles": float(standard["rmse_cycles"]),
        "standard_mae_cycles": float(standard["mae_cycles"]),
        "standard_nasa_score": float(standard["nasa_score"]),
    }


def _paired_repeated_significance(
    rows: list[dict[str, object]],
    reference: str,
    *,
    metric: str = "rmse_cycles",
) -> dict[str, object]:
    if not rows:
        return {}
    keys = sorted({(row["dataset"], row["seed"]) for row in rows})
    by_key_model = {(row["dataset"], row["seed"], row["model"]): row for row in rows}
    out = {"reference": reference, "comparisons": {}}
    models = sorted({str(row["model"]) for row in rows if row["model"] != reference})
    for model in models:
        deltas = []
        for dataset, seed in keys:
            ref = by_key_model.get((dataset, seed, reference))
            other = by_key_model.get((dataset, seed, model))
            if ref is None or other is None or metric not in ref or metric not in other:
                continue
            deltas.append(float(ref[metric]) - float(other[metric]))
        if not deltas:
            continue
        arr = np.asarray(deltas, dtype=float)
        out["comparisons"][model] = {
            f"mean_delta_{metric}": float(arr.mean()),
            "bootstrap_ci95": _bootstrap_ci(arr),
            "sign_flip_p_value": _sign_flip_p_value(arr),
            "n_pairs": int(arr.size),
        }
    return out


def _bootstrap_ci(values: np.ndarray, *, seed: int = 0, n_boot: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=values.size, replace=True))) for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _sign_flip_p_value(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    n = values.size
    if n <= 12:
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * n)).T.reshape(-1, n)
        means = np.abs((signs * values[None, :]).mean(axis=1))
        return float(np.mean(means >= observed))
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(5000, n))
    means = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(means >= observed))


def _write_repeated_report(summary: dict[str, object], out_dir: Path) -> None:
    lines = ["# Repeated C-MAPSS Operator Proof", "", "## Aggregate Metrics", ""]
    aggregates = summary.get("aggregates", {})
    if isinstance(aggregates, dict):
        for model, metrics in sorted(aggregates.items(), key=lambda item: item[1]["mean_rmse_cycles"]):
            lines.append(
                f"- `{model}`: mean RMSE `{metrics['mean_rmse_cycles']:.4f}`, "
                f"std `{metrics['std_rmse_cycles']:.4f}`, mean features `{metrics['mean_n_features']:.1f}`"
            )
    lines.extend(["", "## Standard C-MAPSS Last-Cycle Metrics", ""])
    if isinstance(aggregates, dict):
        standard_items = [
            (model, metrics)
            for model, metrics in aggregates.items()
            if isinstance(metrics, dict) and "standard_mean_rmse_cycles" in metrics
        ]
        for model, metrics in sorted(standard_items, key=lambda item: item[1]["standard_mean_rmse_cycles"]):
            lines.append(
                f"- `{model}`: last-cycle mean RMSE `{metrics['standard_mean_rmse_cycles']:.4f}`, "
                f"std `{metrics['standard_std_rmse_cycles']:.4f}`, "
                f"mean MAE `{metrics['standard_mean_mae_cycles']:.4f}`, "
                f"mean NASA Score `{metrics['standard_mean_nasa_score']:.2f}`"
            )
    lines.extend(["", "## Significance", ""])
    significance = summary.get("significance", {})
    if isinstance(significance, dict):
        comparisons = significance.get("comparisons", {})
        if isinstance(comparisons, dict):
            for model, stats in comparisons.items():
                lines.append(
                    f"- vs `{model}`: mean delta `{stats['mean_delta_rmse_cycles']:.4f}`, "
                    f"CI95 `{stats['bootstrap_ci95']}`, p `{stats['sign_flip_p_value']:.4f}`"
                )
    lines.extend(["", "## Standard Last-Cycle Significance", ""])
    standard_significance = summary.get("standard_significance", {})
    if isinstance(standard_significance, dict):
        comparisons = standard_significance.get("comparisons", {})
        if isinstance(comparisons, dict):
            for model, stats in comparisons.items():
                lines.append(
                    f"- vs `{model}`: mean delta `{stats['mean_delta_standard_rmse_cycles']:.4f}`, "
                    f"CI95 `{stats['bootstrap_ci95']}`, p `{stats['sign_flip_p_value']:.4f}`"
                )
    (out_dir / "repeated_report.md").write_text("\n".join(lines) + "\n")


def _select_functional_features(
    field_train_df: pd.DataFrame,
    selector_val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    z_field: np.ndarray,
    z_selector: np.ndarray,
    z_train: np.ndarray,
    z_val: np.ndarray,
    derivative_features: list[DiscoveredFeature],
    field: object,
    *,
    max_selected_features: int,
    selector: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[SelectedFeature], dict[str, object]]:
    raw_field = field_train_df[feature_cols].to_numpy(float)
    raw_selector = selector_val_df[feature_cols].to_numpy(float)
    raw_train = train_df[feature_cols].to_numpy(float)
    raw_val = val_df[feature_cols].to_numpy(float)
    raw_test = test_df[feature_cols].to_numpy(float)

    selection_candidates = _candidate_triplet(
        field_train_df,
        selector_val_df,
        test_df,
        feature_cols,
        derivative_features,
        field,
    )
    final_candidates = _candidate_triplet(
        train_df,
        val_df,
        test_df,
        feature_cols,
        derivative_features,
        field,
    )
    if selector == "greedy":
        selected = _greedy_validation_selection(
            raw_field,
            z_field,
            raw_selector,
            z_selector,
            selection_candidates,
            max_selected_features=max_selected_features,
            prefilter=120,
        )
        details: dict[str, object] = {"selector": "greedy", "candidate_count": len(selection_candidates)}
    elif selector == "gumbel":
        selected, details = _gumbel_topk_selection(
            raw_field,
            z_field,
            raw_selector,
            z_selector,
            selection_candidates,
            max_selected_features=max_selected_features,
            prefilter=120,
            seed=0,
        )
    else:
        raise ValueError(f"unknown selector {selector!r}")

    candidate_by_name = {candidate.name: candidate for candidate in final_candidates}
    selected_candidates = [candidate_by_name[feature.name] for feature in selected]
    if selected_candidates:
        selected_train = np.column_stack([candidate.train for candidate in selected_candidates])
        selected_val = np.column_stack([candidate.val for candidate in selected_candidates])
        selected_test = np.column_stack([candidate.test for candidate in selected_candidates])
        train_matrix = np.column_stack([raw_train, selected_train])
        val_matrix = np.column_stack([raw_val, selected_val])
        test_matrix = np.column_stack([raw_test, selected_test])
    else:
        train_matrix = raw_train
        val_matrix = raw_val
        test_matrix = raw_test
    names = feature_cols + [feature.name for feature in selected]
    return train_matrix, val_matrix, test_matrix, names, selected, details


def _past_trajectory_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train, names = _past_trajectory_matrix(train_df, feature_cols)
    val, _ = _past_trajectory_matrix(val_df, feature_cols)
    test, _ = _past_trajectory_matrix(test_df, feature_cols)
    return train, val, test, names


def _past_trajectory_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    work = df[["unit_number", *feature_cols]].copy()
    work["__row_order"] = np.arange(work.shape[0])
    work = work.sort_values(["unit_number", "time_cycles"])

    cols = [work[col].to_numpy(float) for col in feature_cols]
    names = list(feature_cols)
    sensor_cols = [col for col in feature_cols if col.startswith("s_")]
    for col in sensor_cols:
        group = work.groupby("unit_number", sort=False)[col]
        values = work[col].to_numpy(float)
        first = group.transform("first").to_numpy(float)
        shifted_5 = group.shift(5).fillna(group.transform("first")).to_numpy(float)
        roll_5 = group.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True).to_numpy(float)
        roll_15 = group.rolling(15, min_periods=1).mean().reset_index(level=0, drop=True).to_numpy(float)
        deriv = group.diff().fillna(0.0)
        generated = [
            (values - first, f"{col}_delta0"),
            (deriv.to_numpy(float), f"{col}_diff1"),
            (values - shifted_5, f"{col}_diff5"),
            ((values - shifted_5) / 5.0, f"{col}_slope5"),
            (roll_5, f"{col}_roll5"),
            (roll_15, f"{col}_roll15"),
            (group.transform(lambda s: np.cumsum(np.maximum(s - float(s.iloc[0]), 0.0))).to_numpy(float), f"{col}_cum_exposure"),
        ]
        current_deriv = deriv
        for order in range(2, 7):
            current_deriv = current_deriv.groupby(work["unit_number"], sort=False).diff().fillna(0.0)
            generated.append((current_deriv.to_numpy(float), f"{col}_diff{order}"))
        for column, name in generated:
            cols.append(column)
            names.append(name)

    matrix = pd.DataFrame(np.stack(cols, axis=1), columns=names)
    matrix["__row_order"] = work["__row_order"].to_numpy()
    matrix = matrix.sort_values("__row_order").drop(columns="__row_order")
    return matrix.to_numpy(float), names


def _candidate_triplet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    derivative_features: list[DiscoveredFeature],
    field: object,
) -> list[CandidateFeature]:
    traj_train, traj_val, traj_test, traj_names = _past_trajectory_matrices(train_df, val_df, test_df, feature_cols)
    raw_train = train_df[feature_cols].to_numpy(float)
    raw_val = val_df[feature_cols].to_numpy(float)
    raw_test = test_df[feature_cols].to_numpy(float)
    return _candidate_function_families(
        traj_train,
        traj_val,
        traj_test,
        traj_names,
        raw_train,
        raw_val,
        raw_test,
        feature_cols,
        derivative_features,
        field,
    )


def _candidate_function_families(
    traj_train: np.ndarray,
    traj_val: np.ndarray,
    traj_test: np.ndarray,
    traj_names: list[str],
    raw_train: np.ndarray,
    raw_val: np.ndarray,
    raw_test: np.ndarray,
    feature_cols: list[str],
    derivative_features: list[DiscoveredFeature],
    field: object,
) -> list[CandidateFeature]:
    candidates: list[CandidateFeature] = []
    raw_count = len(feature_cols)

    for idx, name in enumerate(traj_names[raw_count:], start=raw_count):
        candidates.append(
            CandidateFeature(
                name=name,
                kind="trajectory",
                train=traj_train[:, idx],
                val=traj_val[:, idx],
                test=traj_test[:, idx],
                complexity=_operator_complexity(name, "trajectory"),
                order=_operator_order(name),
            )
        )

    mean = traj_train.mean(axis=0)
    scale = np.where(traj_train.std(axis=0) < 1e-12, 1.0, traj_train.std(axis=0))
    z_train = (traj_train - mean) / scale
    z_val = (traj_val - mean) / scale
    z_test = (traj_test - mean) / scale
    for idx, name in enumerate(traj_names):
        for kind, train_col, val_col, test_col in [
            ("square", z_train[:, idx] ** 2, z_val[:, idx] ** 2, z_test[:, idx] ** 2),
            ("tanh", np.tanh(z_train[:, idx]), np.tanh(z_val[:, idx]), np.tanh(z_test[:, idx])),
            ("exp_clipped", np.exp(np.clip(z_train[:, idx], -3.0, 3.0)), np.exp(np.clip(z_val[:, idx], -3.0, 3.0)), np.exp(np.clip(z_test[:, idx], -3.0, 3.0))),
            (
                "log_abs",
                np.log1p(np.abs(z_train[:, idx])),
                np.log1p(np.abs(z_val[:, idx])),
                np.log1p(np.abs(z_test[:, idx])),
            ),
            (
                "inv_one_plus_abs",
                1.0 / (1.0 + np.abs(z_train[:, idx])),
                1.0 / (1.0 + np.abs(z_val[:, idx])),
                1.0 / (1.0 + np.abs(z_test[:, idx])),
            ),
        ]:
            candidates.append(
                CandidateFeature(
                    f"{kind}({name})",
                    kind,
                    train_col,
                    val_col,
                    test_col,
                    complexity=_operator_complexity(name, kind),
                    order=_operator_order(name),
                )
            )

    if "time_cycles" in traj_names:
        time_idx = traj_names.index("time_cycles")
        for idx, name in enumerate(traj_names):
            if idx == time_idx:
                continue
            if any(token in name for token in ["delta0", "diff5", "roll5"]):
                candidates.append(
                    CandidateFeature(
                        f"time_cycles*{name}",
                        "time_interaction",
                        z_train[:, time_idx] * z_train[:, idx],
                        z_val[:, time_idx] * z_val[:, idx],
                        z_test[:, time_idx] * z_test[:, idx],
                        complexity=_operator_complexity(name, "time_interaction"),
                        order=_operator_order(name),
                    )
                )

    derivative_matrix_train, derivative_names = build_design_matrix(raw_train, derivative_features, include_raw=False)
    derivative_matrix_val, _ = build_design_matrix(raw_val, derivative_features, include_raw=False)
    derivative_matrix_test, _ = build_design_matrix(raw_test, derivative_features, include_raw=False)
    for idx, raw_name in enumerate(derivative_names):
        name = _pretty_feature_name(raw_name, feature_cols)
        candidates.append(
            CandidateFeature(
                name=f"derivative:{name}",
                kind="derivative_routed",
                train=derivative_matrix_train[:, idx],
                val=derivative_matrix_val[:, idx],
                test=derivative_matrix_test[:, idx],
                complexity=2.0,
                order=2,
            )
        )

    if raw_train.shape[0] + raw_val.shape[0] + raw_test.shape[0] <= 5000:
        _, grad_train, hess_train = field_value_grad_hessian(field, raw_train)
        _, grad_val, hess_val = field_value_grad_hessian(field, raw_val)
        _, grad_test, hess_test = field_value_grad_hessian(field, raw_test)
        for idx, col in enumerate(feature_cols):
            candidates.append(
                CandidateFeature(
                    name=f"field_d1:{col}",
                    kind="field_derivative",
                    train=grad_train[:, idx],
                    val=grad_val[:, idx],
                    test=grad_test[:, idx],
                    complexity=2.0,
                    order=1,
                )
            )
            candidates.append(
                CandidateFeature(
                    name=f"field_d2:{col}",
                    kind="field_derivative",
                    train=hess_train[:, idx, idx],
                    val=hess_val[:, idx, idx],
                    test=hess_test[:, idx, idx],
                    complexity=3.0,
                    order=2,
                )
            )

    return _dedupe_candidates(candidates)


def _greedy_validation_selection(
    base_train: np.ndarray,
    z_train: np.ndarray,
    base_val: np.ndarray,
    z_val: np.ndarray,
    candidates: list[CandidateFeature],
    *,
    max_selected_features: int,
    prefilter: int,
) -> list[SelectedFeature]:
    if not candidates:
        return []
    baseline_rmse = _ridge_validation_rmse(base_train, z_train, base_val, z_val)
    scored = []
    for idx, candidate in enumerate(candidates):
        rmse = _ridge_validation_rmse(
            np.column_stack([base_train, candidate.train]),
            z_train,
            np.column_stack([base_val, candidate.val]),
            z_val,
        )
        scored.append((rmse, idx))
    remaining = [idx for _, idx in sorted(scored)[:prefilter]]

    selected: list[SelectedFeature] = []
    current_train = base_train
    current_val = base_val
    current_rmse = baseline_rmse
    while remaining and len(selected) < max_selected_features:
        best: tuple[float, int] | None = None
        for idx in remaining:
            candidate = candidates[idx]
            rmse = _ridge_validation_rmse(
                np.column_stack([current_train, candidate.train]),
                z_train,
                np.column_stack([current_val, candidate.val]),
                z_val,
            )
            if best is None or rmse < best[0]:
                best = (rmse, idx)
        assert best is not None
        if best[0] >= current_rmse - 1e-4:
            break
        current_rmse = best[0]
        chosen = candidates[best[1]]
        selected.append(
            SelectedFeature(
                chosen.name,
                chosen.kind,
                float(current_rmse),
                selector="greedy",
                probability=1.0,
                complexity=chosen.complexity,
            )
        )
        current_train = np.column_stack([current_train, chosen.train])
        current_val = np.column_stack([current_val, chosen.val])
        remaining.remove(best[1])
    return selected


def _gumbel_topk_selection(
    base_train: np.ndarray,
    z_train: np.ndarray,
    base_val: np.ndarray,
    z_val: np.ndarray,
    candidates: list[CandidateFeature],
    *,
    max_selected_features: int,
    prefilter: int,
    seed: int,
    n_draws: int = 18,
) -> tuple[list[SelectedFeature], dict[str, object]]:
    if not candidates:
        return [], {"selector": "gumbel", "candidate_count": 0}
    base_rmse = _ridge_validation_rmse(base_train, z_train, base_val, z_val)
    base_pred, _ = _fit_tuned_ridge_z(base_train, z_train, base_val, z_val, base_val, [1e-4, 1e-2, 1.0, 10.0])
    residual = z_val - base_pred
    single_scores = []
    for idx, candidate in enumerate(candidates):
        val_score = abs(_corr(candidate.val, residual))
        train_score = abs(_corr(candidate.train, z_train))
        score = (0.7 * val_score + 0.3 * train_score) / max(candidate.complexity, 1e-6)
        single_scores.append((score, idx, base_rmse))
    filtered = sorted(single_scores, reverse=True)[:prefilter]
    pool_indices = np.asarray([idx for _, idx, _ in filtered], dtype=int)
    logits = np.asarray([score for score, _, _ in filtered], dtype=float)
    if np.all(logits <= 0):
        logits = -np.asarray([candidates[idx].complexity for idx in pool_indices], dtype=float)
    logits = (logits - logits.mean()) / (logits.std() + 1e-8)
    logits -= 0.05 * np.asarray([candidates[idx].complexity for idx in pool_indices], dtype=float)

    rng = np.random.default_rng(seed)
    best_rmse = np.inf
    best_subset: list[int] = []
    best_temperature = 1.0
    for temperature in [1.2, 0.8, 0.5]:
        for _ in range(n_draws):
            gumbel = -np.log(-np.log(rng.uniform(1e-6, 1.0 - 1e-6, size=logits.size)))
            scores = (logits + gumbel) / temperature
            local = np.argsort(scores)[-max_selected_features:]
            subset = pool_indices[local].tolist()
            rmse = _subset_validation_rmse(base_train, z_train, base_val, z_val, candidates, subset)
            if rmse < best_rmse:
                best_rmse = rmse
                best_subset = subset
                best_temperature = temperature

    probs = _softmax(logits / max(best_temperature, 1e-6))
    probability_by_idx = {int(idx): float(prob) for idx, prob in zip(pool_indices, probs, strict=False)}
    selected = [
        SelectedFeature(
            candidates[idx].name,
            candidates[idx].kind,
            float(best_rmse),
            selector="gumbel",
            probability=probability_by_idx.get(idx, 0.0),
            complexity=candidates[idx].complexity,
        )
        for idx in best_subset
    ]
    selected.sort(key=lambda item: (-item.probability, item.complexity, item.name))
    return selected, {
        "selector": "gumbel",
        "candidate_count": len(candidates),
        "prefiltered_candidates": len(pool_indices),
        "selected_count": len(selected),
        "validation_rmse_z": float(best_rmse),
        "temperature": float(best_temperature),
        "draws_per_temperature": n_draws,
    }


def _subset_validation_rmse(
    base_train: np.ndarray,
    z_train: np.ndarray,
    base_val: np.ndarray,
    z_val: np.ndarray,
    candidates: list[CandidateFeature],
    subset: list[int],
) -> float:
    if not subset:
        return _ridge_validation_rmse(base_train, z_train, base_val, z_val)
    return _ridge_validation_rmse(
        np.column_stack([base_train] + [candidates[idx].train for idx in subset]),
        z_train,
        np.column_stack([base_val] + [candidates[idx].val for idx in subset]),
        z_val,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(), 1e-12)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


def _operator_order(name: str) -> int:
    for order in range(6, 0, -1):
        if f"diff{order}" in name:
            return order
    if "slope" in name:
        return 1
    if "field_d2" in name:
        return 2
    if "field_d1" in name:
        return 1
    return 0


def _operator_complexity(name: str, kind: str) -> float:
    complexity = 1.0
    order = _operator_order(name)
    if kind in {"square", "tanh", "log_abs", "inv_one_plus_abs"}:
        complexity += 0.5
    if kind == "exp_clipped":
        complexity += 1.0
    if kind in {"time_interaction", "derivative_routed", "field_derivative"}:
        complexity += 1.0
    if "roll15" in name or "cum_exposure" in name:
        complexity += 0.5
    complexity += 0.35 * max(order - 1, 0)
    return float(complexity)


def _ridge_validation_rmse(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    pred, _ = _fit_tuned_ridge_z(x_train, y_train, x_val, y_val, x_val, [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0])
    return float(np.sqrt(np.mean((pred - y_val) ** 2)))


def _dedupe_candidates(candidates: list[CandidateFeature]) -> list[CandidateFeature]:
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        unique.append(candidate)
    return unique


def _fair_operator_dictionary_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    derivative_features: list[DiscoveredFeature],
    field: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    raw_train = train_df[feature_cols].to_numpy(float)
    raw_val = val_df[feature_cols].to_numpy(float)
    raw_test = test_df[feature_cols].to_numpy(float)
    candidates = _candidate_triplet(train_df, val_df, test_df, feature_cols, derivative_features, field)
    stable = [candidate for candidate in candidates if _is_stable_candidate(candidate)]
    train = np.column_stack([raw_train] + [candidate.train for candidate in stable])
    val = np.column_stack([raw_val] + [candidate.val for candidate in stable])
    test = np.column_stack([raw_test] + [candidate.test for candidate in stable])
    return train, val, test, feature_cols + [candidate.name for candidate in stable]


def _is_stable_candidate(candidate: CandidateFeature) -> bool:
    values = np.concatenate([candidate.train, candidate.val])
    return bool(np.all(np.isfinite(values)) and np.nanstd(values) > 1e-10 and np.nanmax(np.abs(values)) < 1e8)


def _operator_budget_curve(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    y_test: np.ndarray,
    test_unit: np.ndarray,
    z_train: np.ndarray,
    z_val: np.ndarray,
    y_mean: float,
    y_scale: float,
    derivative_features: list[DiscoveredFeature],
    field: object,
    selected: list[SelectedFeature],
    *,
    budgets: tuple[int, ...],
) -> list[dict[str, object]]:
    raw_train = train_df[feature_cols].to_numpy(float)
    raw_val = val_df[feature_cols].to_numpy(float)
    raw_test = test_df[feature_cols].to_numpy(float)
    candidates = {candidate.name: candidate for candidate in _candidate_triplet(train_df, val_df, test_df, feature_cols, derivative_features, field)}
    out = []
    for budget in budgets:
        names = [feature.name for feature in selected[:budget] if feature.name in candidates]
        if names:
            train = np.column_stack([raw_train] + [candidates[name].train for name in names])
            val = np.column_stack([raw_val] + [candidates[name].val for name in names])
            test = np.column_stack([raw_test] + [candidates[name].test for name in names])
        else:
            train, val, test = raw_train, raw_val, raw_test
        pred, details = _fit_tuned_ridge_cycles(train, z_train, val, z_val, test, y_mean=y_mean, y_scale=y_scale)
        out.append({"budget": int(budget), "selected_count": len(names), "model": "ridge", **_metrics(y_test, pred, test_unit), "details": details})
    return out


def _significance_dict(models: dict[str, object], y_true: np.ndarray, unit: np.ndarray) -> dict[str, object]:
    if "omnibias_functional_selected_hist_gradient_boosting" not in models:
        return {}
    target = models["omnibias_functional_selected_hist_gradient_boosting"]
    if not isinstance(target, dict):
        return {}
    # Predictions are intentionally not retained in JSON artifacts; use aggregate tests when available.
    comparisons = {}
    for name, metrics in models.items():
        if name == "omnibias_functional_selected_hist_gradient_boosting" or not isinstance(metrics, dict):
            continue
        comparisons[name] = {
            "rmse_delta_cycles": float(target["rmse_cycles"]) - float(metrics["rmse_cycles"]),
            "mae_delta_cycles": float(target["mae_cycles"]) - float(metrics["mae_cycles"]),
            "note": "aggregate delta only; repeated-run engine-level tests are in repeated_significance.json",
        }
    return {"reference": "omnibias_functional_selected_hist_gradient_boosting", "comparisons": comparisons, "n_engines": int(np.unique(unit).size), "n_rows": int(y_true.size)}


def write_artifacts(results: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (out_dir / "selected_operators.json").write_text(
        json.dumps(results.get("selected_functional_features", []), indent=2, sort_keys=True)
    )
    (out_dir / "operator_budget.json").write_text(
        json.dumps(results.get("operator_budget", []), indent=2, sort_keys=True)
    )
    (out_dir / "significance.json").write_text(json.dumps(results.get("significance", {}), indent=2, sort_keys=True))
    (out_dir / "standard_cmapss_metrics.json").write_text(
        json.dumps(results.get("standard_cmapss_protocol", {}), indent=2, sort_keys=True)
    )
    (out_dir / "ablation_results.json").write_text(
        json.dumps(_ablation_payload(results.get("models", {})), indent=2, sort_keys=True)
    )
    report = [
        "# C-MAPSS Feature Discovery Benchmark",
        "",
        f"Dataset: `{results['dataset']}`",
        f"Target: `{results['target']}`",
        "",
        "## Discovered Features",
        "",
    ]
    for feature in results["discovered_features"]:  # type: ignore[index]
        report.append(f"- `{feature['name']}` ({feature['kind']}), score `{feature['score']:.4f}`")
    report.extend(["", "## Validation-Selected Functional Features", ""])
    for feature in results["selected_functional_features"]:  # type: ignore[index]
        report.append(
            f"- `{feature['name']}` ({feature['kind']}), "
            f"probability `{feature.get('probability', 1.0):.4f}`, "
            f"validation RMSE `{feature['validation_rmse_z_after_add']:.4f}`"
        )
    report.extend(["", "## Test Metrics", ""])
    models: dict[str, dict[str, object]] = results["models"]  # type: ignore[assignment]
    for name, metrics in sorted(models.items(), key=lambda item: item[1]["rmse_cycles"]):
        report.append(
            f"- `{name}`: RMSE `{metrics['rmse_cycles']:.4f}` cycles, "
            f"MAE `{metrics['mae_cycles']:.4f}` cycles, features `{metrics['n_features']}`"
        )
    report.extend(["", "## Standard C-MAPSS Last-Cycle Metrics", ""])
    standard = results.get("standard_cmapss_protocol", {})
    if isinstance(standard, dict):
        for name, metrics in sorted(standard.items(), key=lambda item: item[1]["rmse_cycles"]):
            report.append(
                f"- `{name}`: last-cycle RMSE `{metrics['rmse_cycles']:.4f}`, "
                f"MAE `{metrics['mae_cycles']:.4f}`, NASA Score `{metrics['nasa_score']:.2f}`"
            )
    report.extend(["", "## Literature Context", ""])
    literature = results.get("literature_context", {})
    if isinstance(literature, dict):
        note = literature.get("protocol_note")
        if isinstance(note, str):
            report.append(note)
        report.append("")
        for item in literature.get("sota_reference_points", []):  # type: ignore[union-attr]
            if isinstance(item, dict):
                score = item.get("fd001_nasa_score", "n/a")
                report.append(
                    f"- `{item['method']}`: FD001 RMSE `{item['fd001_rmse_cycles']}`, "
                    f"NASA Score `{score}` ({item['source']}; {item['url']})"
                )
    report.extend(["", "## Feature Gain", ""])
    for family, gain in results["feature_gain"].items():  # type: ignore[union-attr]
        line = (
            f"- `{family}`: raw RMSE `{gain['raw_rmse_cycles']:.4f}` -> "
            f"omnibias RMSE `{gain['omnibias_rmse_cycles']:.4f}`, "
            f"delta `{gain['delta_rmse_cycles']:.4f}`"
        )
        if "generic_dictionary_rmse_cycles" in gain:
            line += (
                f"; generic dictionary RMSE `{gain['generic_dictionary_rmse_cycles']:.4f}`, "
                f"delta vs generic `{gain['delta_vs_generic_dictionary_cycles']:.4f}`"
            )
        if "fair_operator_dictionary_rmse_cycles" in gain:
            line += (
                f"; fair operator RMSE `{gain['fair_operator_dictionary_rmse_cycles']:.4f}`, "
                f"delta vs fair `{gain['delta_vs_fair_operator_dictionary_cycles']:.4f}`"
            )
        report.append(line)
    report.extend(["", "## Operator Budget Curve", ""])
    for item in results.get("operator_budget", []):  # type: ignore[union-attr]
        report.append(
            f"- budget `{item['budget']}`: RMSE `{item['rmse_cycles']:.4f}`, "
            f"MAE `{item['mae_cycles']:.4f}`, selected `{item['selected_count']}`"
        )
    report.extend(
        [
            "",
            "## Fairness",
            "",
            "- Omnibias field fitting and candidate-family generation use field-train engines only.",
            "- Operator selection uses selection-validation engines only.",
            "- Model hyperparameters use tuning-validation engines only.",
            "- The held-out parquet test set is used only for final metrics.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report) + "\n")


def _ablation_payload(models: object) -> dict[str, object]:
    if not isinstance(models, dict):
        return {}
    wanted = [
        "raw",
        "generic_dictionary",
        "fair_operator_dictionary",
        "omnibias_derivative_magnitude",
        "omnibias_functional_selected",
    ]
    return {
        name: metrics
        for name, metrics in models.items()
        if isinstance(name, str) and any(name.startswith(prefix) for prefix in wanted)
    }


def _fit_tuned_ridge_cycles(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_val: np.ndarray,
    z_val: np.ndarray,
    x_test: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    pred_z, details = _fit_tuned_ridge_z(x_train, z_train, x_val, z_val, x_test, alphas)
    return _postprocess_rul_predictions(pred_z * y_scale + y_mean), details


def _fit_tuned_ridge_z(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, dict[str, float]]:
    x_mean = x_train.mean(axis=0)
    x_scale = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
    train_s = (x_train - x_mean) / x_scale
    val_s = (x_val - x_mean) / x_scale
    best: tuple[float, float] | None = None
    for alpha in alphas:
        pred_val = _ridge_predict(train_s, y_train, val_s, alpha)
        rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, alpha)
    assert best is not None
    alpha = best[1]
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)
    fit_mean = x_fit.mean(axis=0)
    fit_scale = np.where(x_fit.std(axis=0) < 1e-12, 1.0, x_fit.std(axis=0))
    pred_test = _ridge_predict((x_fit - fit_mean) / fit_scale, y_fit, (x_test - fit_mean) / fit_scale, alpha)
    return pred_test, {"alpha": float(alpha), "validation_rmse_z": float(best[0])}


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    reg = alpha * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y_train)
    return np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1) @ coef


def _fit_optional_hgb_cycles(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_val: np.ndarray,
    z_val: np.ndarray,
    x_test: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> tuple[np.ndarray, dict[str, object]] | None:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return None
    if x_train.shape[1] > 500:
        return None
    selected_feature_count = x_train.shape[1]
    if x_train.shape[1] > 250:
        scores = np.asarray([abs(_corr(x_train[:, idx], z_train)) for idx in range(x_train.shape[1])])
        keep = np.argsort(scores)[-250:]
        x_train = x_train[:, keep]
        x_val = x_val[:, keep]
        x_test = x_test[:, keep]
        selected_feature_count = int(keep.size)
    candidates = [{"max_iter": 60, "learning_rate": 0.08, "max_leaf_nodes": 15}]
    train_idx = _even_sample_indices(x_train.shape[0], 6000)
    x_train_fit = x_train[train_idx]
    z_train_fit = z_train[train_idx]
    best = None
    for params in candidates:
        model = HistGradientBoostingRegressor(random_state=0, **params)
        model.fit(x_train_fit, z_train_fit)
        pred = model.predict(x_val)
        rmse = float(np.sqrt(np.mean((pred - z_val) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, params)
    assert best is not None
    x_fit = np.concatenate([x_train, x_val], axis=0)
    z_fit = np.concatenate([z_train, z_val], axis=0)
    fit_idx = _even_sample_indices(x_fit.shape[0], 8000)
    model = HistGradientBoostingRegressor(random_state=0, **best[1])
    model.fit(x_fit[fit_idx], z_fit[fit_idx])
    pred = model.predict(x_test) * y_scale + y_mean
    return _postprocess_rul_predictions(pred), {
        "params": best[1],
        "validation_rmse_z": best[0],
        "hgb_feature_count": selected_feature_count,
    }


def _postprocess_rul_predictions(pred: np.ndarray) -> np.ndarray:
    """Apply the usual physical range constraint for capped C-MAPSS RUL labels."""
    return np.clip(np.asarray(pred, dtype=float), 0.0, TARGET_CAP)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, unit: np.ndarray) -> dict[str, object]:
    err = y_pred - y_true
    unit_rmses = []
    for engine in np.unique(unit):
        mask = unit == engine
        unit_rmses.append(float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))))
    return {
        "rmse_cycles": float(np.sqrt(np.mean(err**2))),
        "mae_cycles": float(np.mean(np.abs(err))),
        "mean_engine_rmse_cycles": float(np.mean(unit_rmses)),
    }


def _standard_last_cycle_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit: np.ndarray,
    time_cycles: np.ndarray,
) -> dict[str, object]:
    last_indices = []
    for engine in np.unique(unit):
        engine_idx = np.flatnonzero(unit == engine)
        last_indices.append(int(engine_idx[np.argmax(time_cycles[engine_idx])]))
    idx = np.asarray(last_indices, dtype=int)
    true_last = np.asarray(y_true, dtype=float)[idx]
    pred_last = np.asarray(y_pred, dtype=float)[idx]
    err = pred_last - true_last
    return {
        "rmse_cycles": float(np.sqrt(np.mean(err**2))),
        "mae_cycles": float(np.mean(np.abs(err))),
        "n_engines": int(idx.size),
        "nasa_score": _nasa_rul_score(true_last, pred_last),
        "target_protocol": "last observed cycle per test engine; uncapped true RUL",
    }


def _nasa_rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(diff < 0.0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(penalties))


def _standard_protocol_payload(models: dict[str, object]) -> dict[str, object]:
    out = {}
    for name, metrics in models.items():
        if isinstance(metrics, dict) and isinstance(metrics.get("standard_last_cycle"), dict):
            out[name] = metrics["standard_last_cycle"]
    return out


def _literature_context() -> dict[str, object]:
    return {
        "protocol_note": "Published C-MAPSS FD001 results usually evaluate one prediction per held-out test engine at its last observed cycle and report RMSE plus the asymmetric NASA RUL score.",
        "sota_reference_points": [
            {
                "method": "Acyclic graph network",
                "fd001_rmse_cycles": 11.96,
                "fd001_nasa_score": 229.0,
                "source": "Cited as prior best in Scientific Reports 2024 PSA-GHLSTM comparison",
                "url": "https://www.nature.com/articles/s41598-024-59095-3",
            },
            {
                "method": "PSA-GHLSTM transformer/LSTM hybrid",
                "fd001_rmse_cycles": 13.14,
                "fd001_rmse_std_cycles": 0.21,
                "fd001_nasa_score": 220.0,
                "fd001_nasa_score_std": 23.0,
                "source": "Scientific Reports 2024",
                "url": "https://www.nature.com/articles/s41598-024-59095-3",
            },
            {
                "method": "CAELSTM",
                "fd001_rmse_cycles": 14.44,
                "fd001_mae_cycles": 10.49,
                "fd001_nasa_score": 282.38,
                "source": "Scientific Reports 2025",
                "url": "https://www.nature.com/articles/s41598-025-09155-z",
            },
        ],
    }


def _feature_gain(models: dict[str, object]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for suffix in ["ridge", "hist_gradient_boosting"]:
        raw = models.get(f"raw_{suffix}")
        omni = models.get(f"omnibias_functional_selected_{suffix}")
        generic = models.get(f"generic_dictionary_{suffix}")
        fair = models.get(f"fair_operator_dictionary_{suffix}")
        if not isinstance(raw, dict) or not isinstance(omni, dict):
            continue
        raw_rmse = float(raw["rmse_cycles"])
        omni_rmse = float(omni["rmse_cycles"])
        gain = {
            "raw_rmse_cycles": raw_rmse,
            "omnibias_rmse_cycles": omni_rmse,
            "delta_rmse_cycles": omni_rmse - raw_rmse,
            "relative_change": (omni_rmse - raw_rmse) / max(raw_rmse, 1e-12),
        }
        if isinstance(generic, dict):
            generic_rmse = float(generic["rmse_cycles"])
            gain["generic_dictionary_rmse_cycles"] = generic_rmse
            gain["delta_vs_generic_dictionary_cycles"] = omni_rmse - generic_rmse
        if isinstance(fair, dict):
            fair_rmse = float(fair["rmse_cycles"])
            gain["fair_operator_dictionary_rmse_cycles"] = fair_rmse
            gain["delta_vs_fair_operator_dictionary_cycles"] = omni_rmse - fair_rmse
        out[suffix] = gain
    return out


def _pretty_feature_names(names: list[str], feature_cols: list[str]) -> list[str]:
    return [_pretty_feature_name(name, feature_cols) for name in names]


def _pretty_feature_name(name: str, feature_cols: list[str]) -> str:
    out = name
    for idx in range(len(feature_cols), 0, -1):
        out = out.replace(f"x{idx}", feature_cols[idx - 1])
    return out


def _even_sample_indices(n_rows: int, max_rows: int) -> np.ndarray:
    if n_rows <= max_rows:
        return np.arange(n_rows)
    return np.linspace(0, n_rows - 1, max_rows, dtype=int)
