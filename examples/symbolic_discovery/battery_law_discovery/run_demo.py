# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run the battery law-discovery demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .baselines import (
        BaselineResult,
        empirical_per_cell_baseline,
        finite_difference_sindy_baseline,
        ridge_capacity_baseline,
        sklearn_tree_baseline,
        tuned_feature_model_baselines,
    )
    from .features import build_feature_bundle, fit_feature_stats
    from .omnibias_law_model import build_omnibias_feature_matrix, fit_omnibias_law, rollout_law
    from .plot_results import plot_baseline_comparison, plot_capacity_rollout
    from .severson_loader import (
        CycleTable,
        load_severson_cycle_table,
        make_synthetic_cycle_table,
        train_test_cell_split,
        train_test_protocol_split,
    )
except ImportError:  # pragma: no cover
    from baselines import (
        BaselineResult,
        empirical_per_cell_baseline,
        finite_difference_sindy_baseline,
        ridge_capacity_baseline,
        sklearn_tree_baseline,
        tuned_feature_model_baselines,
    )
    from features import build_feature_bundle, fit_feature_stats
    from omnibias_law_model import build_omnibias_feature_matrix, fit_omnibias_law, rollout_law
    from plot_results import plot_baseline_comparison, plot_capacity_rollout
    from severson_loader import (
        CycleTable,
        load_severson_cycle_table,
        make_synthetic_cycle_table,
        train_test_cell_split,
        train_test_protocol_split,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/battery_law_discovery"))
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic capacity-fade data.")
    parser.add_argument("--quick", action="store_true", help="Use small cell count / fewer steps.")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["cell", "protocol"],
        default="cell",
        help="Held-out cell split or high-C-rate protocol extrapolation split.",
    )
    parser.add_argument(
        "--law-mode",
        choices=["physics", "direct"],
        default="physics",
        help="Use monotone physics-constrained law or unconstrained direct dq/dn law.",
    )
    parser.add_argument(
        "--operator-set",
        choices=["auto", "minimal", "poly", "capacity", "stress", "stress_interactions"],
        default="auto",
        help="Physics operator library to use; auto selects on a training derivative validation split.",
    )
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--sparsity-threshold",
        type=float,
        default=1e-3,
        help="Sequential thresholded ridge cutoff for the sparse law head.",
    )
    parser.add_argument(
        "--skip-tuned-baselines",
        action="store_true",
        help="Skip validation-tuned raw-vs-omnibias feature sklearn benchmark.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _load_table(args)
    train_table, test_table = _split_table(table, args)
    stats = fit_feature_stats(train_table)
    train = build_feature_bundle(train_table, stats)
    test = build_feature_bundle(test_table, stats)

    steps = 250 if args.quick else args.steps
    hidden = min(args.hidden, 32) if args.quick else args.hidden
    law_result = fit_omnibias_law(
        train.x,
        train.y,
        test.x,
        test.y,
        hidden=hidden,
        steps=steps,
        lr=args.lr,
        sparsity_threshold=args.sparsity_threshold,
        law_mode=args.law_mode,
        operator_set=args.operator_set,
        seed=args.seed,
    )

    augmented_train_x, augmented_test_x, operator_feature_names = _operator_feature_matrices(train, test, law_result)
    baseline_results = _run_baselines(train, test, args.sparsity_threshold)
    model_selection: dict[str, object] = {
        "operator_feature_names": operator_feature_names,
        "operator_feature_count": len(operator_feature_names),
    }
    if not args.skip_tuned_baselines:
        tuned_results, tuned_details = tuned_feature_model_baselines(
            train,
            test,
            raw_train_x=train.x,
            raw_test_x=test.x,
            augmented_train_x=augmented_train_x,
            augmented_test_x=augmented_test_x,
            seed=args.seed,
        )
        baseline_results.extend(tuned_results)
        model_selection["tuned_models"] = tuned_details
    simulator_pred = _rollout_test_cells(test, law_result.law)
    metrics = _metrics_dict(law_result, baseline_results, test, simulator_pred)
    feature_gain = _feature_gain_summary(baseline_results)
    feature_significance = _feature_significance_dict(test, baseline_results, seed=args.seed)
    if feature_gain:
        metrics["omnibias_feature_gain"] = feature_gain
    significance = _significance_dict(test, simulator_pred, baseline_results, seed=args.seed)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (out_dir / "significance.json").write_text(json.dumps(significance, indent=2, sort_keys=True))
    (out_dir / "feature_significance.json").write_text(
        json.dumps(feature_significance, indent=2, sort_keys=True)
    )
    (out_dir / "model_selection.json").write_text(json.dumps(model_selection, indent=2, sort_keys=True))
    (out_dir / "discovered_law.txt").write_text(law_result.law.equation() + "\n")

    ridge_pred = baseline_results[0].predictions if baseline_results else None
    plot_capacity_rollout(out_dir / "capacity_rollout.png", test, simulator_pred, ridge_pred)
    plot_baseline_comparison(out_dir / "baseline_comparison.png", metrics)
    _write_report(
        out_dir,
        args,
        table,
        train,
        test,
        law_result,
        simulator_pred,
        baseline_results,
        significance,
        feature_gain,
        feature_significance,
    )

    print(f"Wrote demo artifacts to {out_dir}")
    print(law_result.law.equation())
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps({"significance_vs_omnibias": significance}, indent=2, sort_keys=True))


def _load_table(args: argparse.Namespace) -> CycleTable:
    if args.synthetic:
        n_cells = 12 if args.quick else 32
        n_cycles = 80 if args.quick else 180
        return make_synthetic_cycle_table(n_cells=n_cells, n_cycles=n_cycles, seed=args.seed)
    if args.data_dir is None:
        raise SystemExit("Pass --data-dir <severson_dir> or use --synthetic for a smoke run.")
    max_cells = args.max_cells
    if args.quick and max_cells is None:
        max_cells = 24
    return load_severson_cycle_table(args.data_dir, max_cells=max_cells)


def _split_table(table: CycleTable, args: argparse.Namespace) -> tuple[CycleTable, CycleTable]:
    if args.split == "protocol":
        return train_test_protocol_split(table)
    return train_test_cell_split(table, seed=args.seed)


def _run_baselines(train, test, sparsity_threshold: float) -> list[BaselineResult]:
    results = [
        ridge_capacity_baseline(train, test),
        empirical_per_cell_baseline(test, mode="linear"),
        empirical_per_cell_baseline(test, mode="sqrt"),
        empirical_per_cell_baseline(test, mode="exp"),
        finite_difference_sindy_baseline(train, test, threshold=sparsity_threshold),
    ]
    optional = sklearn_tree_baseline(train, test)
    if optional is not None:
        results.append(optional)
    return results


def _operator_feature_matrices(train, test, law_result) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_aug, names = build_omnibias_feature_matrix(
        train.x,
        q_field=law_result.train_predictions,
        dqdn=law_result.derivative_train,
        d2qdn2=law_result.second_derivative_train,
        law=law_result.law,
    )
    test_aug, _ = build_omnibias_feature_matrix(
        test.x,
        q_field=law_result.predictions,
        dqdn=law_result.derivative_test,
        d2qdn2=law_result.second_derivative_test,
        law=law_result.law,
    )
    return train_aug, test_aug, names


def _rollout_test_cells(test, law) -> np.ndarray:
    pred = np.empty_like(test.y)
    for cell in np.unique(test.cell_id):
        idx = np.flatnonzero(test.cell_id == cell)
        order = np.argsort(test.cycle_norm[idx])
        ordered_idx = idx[order]
        n = test.cycle_norm[ordered_idx]
        rolled = rollout_law(law, test.x[ordered_idx[0]], float(test.y[ordered_idx[0]]), n)
        pred[ordered_idx] = rolled
    return pred


def _metrics_dict(
    law_result,
    baselines: list[BaselineResult],
    test,
    simulator_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    eol_mae = _cycle_life_mae(test.cycle_norm, test.y, simulator_pred, test.cell_id)
    out = {
        "omnibias_law": {
            "rmse_capacity": float(np.sqrt(np.mean((simulator_pred - test.y) ** 2))),
            "mae_capacity": float(np.mean(np.abs(simulator_pred - test.y))),
            "operator_derivative_rmse": law_result.train_derivative_rmse,
            "field_rmse_capacity": law_result.test_rmse,
            "eol_mae_cycles": eol_mae,
            "active_terms": int(np.count_nonzero(law_result.law.coef)),
            "operator_set": law_result.law.operator_set,
        }
    }
    for baseline in baselines:
        out[baseline.name] = {
            "rmse_capacity": baseline.rmse_capacity,
            "mae_capacity": baseline.mae_capacity,
            "eol_mae_cycles": baseline.eol_mae_cycles,
        }
    return out


def _feature_gain_summary(baselines: list[BaselineResult]) -> dict[str, dict[str, float]]:
    by_name = {baseline.name: baseline for baseline in baselines}
    out: dict[str, dict[str, float]] = {}
    for name, raw in by_name.items():
        if not name.endswith("_raw"):
            continue
        augmented_name = name[: -len("_raw")] + "_omnibias_augmented"
        augmented = by_name.get(augmented_name)
        if augmented is None:
            continue
        out[name[: -len("_raw")]] = {
            "raw_rmse": raw.rmse_capacity,
            "omnibias_augmented_rmse": augmented.rmse_capacity,
            "delta_rmse_augmented_minus_raw": augmented.rmse_capacity - raw.rmse_capacity,
            "relative_rmse_change": (augmented.rmse_capacity - raw.rmse_capacity) / max(raw.rmse_capacity, 1e-12),
        }
    return out


def _feature_significance_dict(
    test,
    baselines: list[BaselineResult],
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    by_name = {baseline.name: baseline for baseline in baselines}
    out: dict[str, dict[str, float]] = {}
    for name, raw in by_name.items():
        if not name.endswith("_raw"):
            continue
        family = name[: -len("_raw")]
        augmented = by_name.get(f"{family}_omnibias_augmented")
        if augmented is None:
            continue
        stats = _paired_cell_significance(test, augmented.predictions, raw.predictions, seed=seed)
        out[family] = {
            "n_cells": stats["n_cells"],
            "mean_delta_rmse_augmented_minus_raw": stats["mean_delta_rmse_omnibias_minus_baseline"],
            "bootstrap_ci_low": stats["bootstrap_ci_low"],
            "bootstrap_ci_high": stats["bootstrap_ci_high"],
            "sign_flip_p_value": stats["sign_flip_p_value"],
            "augmented_better_fraction": stats["omnibias_better_fraction"],
            "augmented_mean_cell_rmse": stats["omnibias_mean_cell_rmse"],
            "raw_mean_cell_rmse": stats["baseline_mean_cell_rmse"],
        }
    return out


def _significance_dict(
    test,
    omnibias_pred: np.ndarray,
    baselines: list[BaselineResult],
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    return {
        baseline.name: _paired_cell_significance(test, omnibias_pred, baseline.predictions, seed=seed)
        for baseline in baselines
    }


def _paired_cell_significance(
    test,
    omnibias_pred: np.ndarray,
    baseline_pred: np.ndarray,
    *,
    seed: int = 0,
    n_resamples: int = 5000,
) -> dict[str, float]:
    diffs = []
    omni_rmses = []
    base_rmses = []
    for cell in np.unique(test.cell_id):
        idx = np.flatnonzero(test.cell_id == cell)
        omni_rmse = float(np.sqrt(np.mean((omnibias_pred[idx] - test.y[idx]) ** 2)))
        base_rmse = float(np.sqrt(np.mean((baseline_pred[idx] - test.y[idx]) ** 2)))
        if np.isfinite(omni_rmse) and np.isfinite(base_rmse):
            diffs.append(omni_rmse - base_rmse)
            omni_rmses.append(omni_rmse)
            base_rmses.append(base_rmse)
    diff = np.asarray(diffs, dtype=float)
    if diff.size == 0:
        return {
            "n_cells": 0,
            "mean_delta_rmse_omnibias_minus_baseline": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "sign_flip_p_value": float("nan"),
            "omnibias_better_fraction": float("nan"),
            "omnibias_mean_cell_rmse": float("nan"),
            "baseline_mean_cell_rmse": float("nan"),
        }

    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.integers(0, diff.size, size=diff.size)
        boot[i] = float(np.mean(diff[sample]))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_resamples, diff.size))
    null = np.mean(signs * diff[None, :], axis=1)
    observed = float(np.mean(diff))
    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1.0) / (n_resamples + 1.0))
    return {
        "n_cells": int(diff.size),
        "mean_delta_rmse_omnibias_minus_baseline": observed,
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "sign_flip_p_value": p_value,
        "omnibias_better_fraction": float(np.mean(diff < 0.0)),
        "omnibias_mean_cell_rmse": float(np.mean(omni_rmses)),
        "baseline_mean_cell_rmse": float(np.mean(base_rmses)),
    }


def _cycle_life_mae(
    cycle_norm: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    cell_id: np.ndarray,
    threshold: float = 0.8,
) -> float:
    errors = []
    for cell in np.unique(cell_id):
        idx = np.flatnonzero(cell_id == cell)
        true_n = _first_crossing(cycle_norm[idx], true[idx], threshold)
        pred_n = _first_crossing(cycle_norm[idx], pred[idx], threshold)
        errors.append(abs(pred_n - true_n) * max(idx.size, 1))
    return float(np.mean(errors)) if errors else float("nan")


def _first_crossing(n: np.ndarray, q: np.ndarray, threshold: float) -> float:
    below = np.flatnonzero(q <= threshold)
    if below.size == 0:
        return float(n[-1])
    return float(n[below[0]])


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    table: CycleTable,
    train,
    test,
    law_result,
    simulator_pred: np.ndarray,
    baseline_results: list[BaselineResult],
    significance: dict[str, dict[str, float]],
    feature_gain: dict[str, dict[str, float]],
    feature_significance: dict[str, dict[str, float]],
) -> None:
    best_baseline = min(baseline_results, key=lambda r: r.rmse_capacity)
    simulator_rmse = float(np.sqrt(np.mean((simulator_pred - test.y) ** 2)))
    lines = [
        "# Battery Law Discovery Demo Report",
        "",
        "## Dataset",
        "",
        f"- rows: {len(table)}",
        f"- train cells: {len(np.unique(train.cell_id))}",
        f"- test cells: {len(np.unique(test.cell_id))}",
        f"- synthetic: {args.synthetic}",
        f"- split: {args.split}",
        f"- law mode: {args.law_mode}",
        f"- operator set: {law_result.law.operator_set or args.operator_set}",
        f"- sparsity threshold: {args.sparsity_threshold}",
        "",
        "## Discovered Law",
        "",
        "```text",
        law_result.law.equation(),
        "```",
        "",
        "## Metrics",
        "",
        f"- omnibias simulator capacity RMSE: `{simulator_rmse:.6f}`",
        f"- omnibias field capacity RMSE: `{law_result.test_rmse:.6f}`",
        f"- best baseline capacity RMSE: `{best_baseline.rmse_capacity:.6f}` ({best_baseline.name})",
        f"- law-vs-field derivative RMSE: `{law_result.train_derivative_rmse:.6f}`",
        "",
        "## Omnibias Feature Benchmark",
        "",
        *_format_feature_gain_lines(feature_gain),
        *_format_feature_significance_lines(feature_significance),
        "",
        "## Statistical Comparison",
        "",
        *_format_significance_lines(significance),
        "",
        "## Claim",
        "",
        "The demo learns a smooth capacity field, extracts closed-form derivative "
        "channels with omnibias/JAX, and compresses those channels into a sparse "
        "capacity-fade equation that can be rolled out as a cheap simulator.",
        "",
        "## Artifacts",
        "",
        "- `metrics.json`",
        "- `significance.json`",
        "- `feature_significance.json`",
        "- `model_selection.json`",
        "- `discovered_law.txt`",
        "- `capacity_rollout.png`",
        "- `baseline_comparison.png`",
    ]
    (out_dir / "demo_report.md").write_text("\n".join(lines) + "\n")


def _format_feature_gain_lines(feature_gain: dict[str, dict[str, float]]) -> list[str]:
    lines = []
    for family, stats in feature_gain.items():
        delta = stats["delta_rmse_augmented_minus_raw"]
        direction = "improved" if delta < 0 else "worsened"
        lines.append(
            f"- `{family}` {direction}: raw RMSE `{stats['raw_rmse']:.6f}` -> "
            f"omnibias-augmented RMSE `{stats['omnibias_augmented_rmse']:.6f}` "
            f"(delta `{delta:.6f}`)"
        )
    return lines or ["- Tuned feature benchmark was skipped or unavailable."]


def _format_feature_significance_lines(feature_significance: dict[str, dict[str, float]]) -> list[str]:
    lines = []
    for family, stats in feature_significance.items():
        delta = stats["mean_delta_rmse_augmented_minus_raw"]
        direction = "improved" if delta < 0 else "worsened"
        lines.append(
            f"- `{family}` paired cell test: omnibias features {direction} mean cell RMSE "
            f"by `{delta:.6f}`; 95% CI `[{stats['bootstrap_ci_low']:.6f}, "
            f"{stats['bootstrap_ci_high']:.6f}]`; p=`{stats['sign_flip_p_value']:.4f}`"
        )
    return lines


def _format_significance_lines(significance: dict[str, dict[str, float]]) -> list[str]:
    lines = []
    for name, stats in significance.items():
        delta = stats["mean_delta_rmse_omnibias_minus_baseline"]
        p_value = stats["sign_flip_p_value"]
        low = stats["bootstrap_ci_low"]
        high = stats["bootstrap_ci_high"]
        direction = "better" if delta < 0 else "worse"
        lines.append(
            f"- vs `{name}`: omnibias {direction} by mean cell RMSE delta "
            f"`{delta:.6f}`; 95% bootstrap CI `[{low:.6f}, {high:.6f}]`; "
            f"sign-flip p=`{p_value:.4f}`"
        )
    return lines or ["- No baseline comparisons available."]


if __name__ == "__main__":
    main()
