# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Benchmark the joint operator-predictor prototype on a controlled law."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omnibias.torch.architectures import fit_joint_operator_regressor


def true_law(x: np.ndarray) -> np.ndarray:
    return 3.0 * x[:, 0] ** 2 - 2.0 * x[:, 1] * x[:, 2] + np.sin(x[:, 3])


def make_dataset(n_samples: int = 3000, noise_std: float = 0.08, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_samples, 4)).astype(np.float32)
    x[:, 3] = rng.uniform(-np.pi, np.pi, size=n_samples)
    y = true_law(x) + rng.normal(0.0, noise_std, size=n_samples)
    y = y.astype(np.float32)
    order = rng.permutation(n_samples)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "x_val": x[val_idx],
        "y_val": y[val_idx],
        "x_test": x[test_idx],
        "y_test": y[test_idx],
    }


def evaluate_benchmark(out_dir: Path, *, seed: int = 0) -> dict[str, object]:
    split = make_dataset(seed=seed)
    raw_pred, raw_details = _fit_tuned_ridge(
        split["x_train"],
        split["y_train"],
        split["x_val"],
        split["y_val"],
        split["x_test"],
    )
    dict_train, dict_names = _generic_dictionary(split["x_train"])
    dict_val, _ = _generic_dictionary(split["x_val"])
    dict_test, _ = _generic_dictionary(split["x_test"])
    dict_pred, dict_details = _fit_tuned_ridge(
        dict_train,
        split["y_train"],
        dict_val,
        split["y_val"],
        dict_test,
    )
    fitted = fit_joint_operator_regressor(
        split["x_train"],
        split["y_train"],
        split["x_val"],
        split["y_val"],
        seed=seed,
        epochs=350,
        patience=80,
        sparsity_weight=1e-1,
        model_kwargs={
            "ombu_channels": 4,
            "stochastic_gates": False,
            "initial_gate_logit": -1.5,
        },
    )
    joint_pred = fitted.predict(split["x_test"])
    selected = fitted.selected_operators(top_k=15)

    results = {
        "dataset": "controlled symbolic regression law",
        "hidden_law": "y = 3*x1^2 - 2*x2*x3 + sin(x4) + noise",
        "models": {
            "raw_ridge": _metrics(split["y_test"], raw_pred) | {"n_features": 4, "details": raw_details},
            "exhaustive_dictionary_ridge": _metrics(split["y_test"], dict_pred)
            | {"n_features": len(dict_names), "details": dict_details},
            "joint_operator_regressor": _metrics(split["y_test"], joint_pred)
            | {
                "n_operators": fitted.model.n_operators,
                "active_operator_count_0.2": fitted.model.active_operator_count(0.2),
                "details": {
                    "best_val_rmse_z": min(fitted.history["val_rmse_z"]),
                    "epochs_ran": len(fitted.history["val_rmse_z"]),
                },
            },
        },
        "selected_operators": selected,
        "seed": seed,
    }
    write_artifacts(results, out_dir)
    return results


def write_artifacts(results: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    lines = [
        "# Joint Operator Regressor Benchmark",
        "",
        f"Hidden law: `{results['hidden_law']}`",
        "",
        "## Test Metrics",
        "",
    ]
    models = results["models"]
    assert isinstance(models, dict)
    for name, metrics in sorted(models.items(), key=lambda item: item[1]["rmse"]):
        lines.append(f"- `{name}`: RMSE `{metrics['rmse']:.4f}`, MAE `{metrics['mae']:.4f}`")
    lines.extend(["", "## Selected Operators", ""])
    for row in results["selected_operators"]:  # type: ignore[index]
        lines.append(
            f"- `{row['name']}` ({row['family']}): gate `{row['gate_probability']:.3f}`, "
            f"weight `{row['readout_weight']:.3f}`, importance `{row['importance']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The joint model learns operator gates and readout weights in one optimization loop. "
            "A successful run should rank `x1^2`, `x2*x3`, and `sin(x4)` near the top without a separate feature-selection pass.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))}


def _fit_tuned_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    x_mean = x_train.mean(axis=0)
    x_scale = np.where(x_train.std(axis=0) < 1e-9, 1.0, x_train.std(axis=0))
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    xtr = (x_train - x_mean) / x_scale
    xv = (x_val - x_mean) / x_scale
    ytr = (y_train - y_mean) / y_scale
    yv = (y_val - y_mean) / y_scale
    best = None
    for alpha in alphas:
        pred = _ridge_predict(xtr, ytr, xv, alpha)
        rmse = float(np.sqrt(np.mean((pred - yv) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, alpha)
    assert best is not None
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)
    fit_mean = x_fit.mean(axis=0)
    fit_scale = np.where(x_fit.std(axis=0) < 1e-9, 1.0, x_fit.std(axis=0))
    y_fit_z = (y_fit - y_mean) / y_scale
    pred_z = _ridge_predict((x_fit - fit_mean) / fit_scale, y_fit_z, (x_test - fit_mean) / fit_scale, best[1])
    return pred_z * y_scale + y_mean, {"alpha": float(best[1]), "validation_rmse_z": float(best[0])}


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    reg = alpha * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y_train)
    return np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1) @ coef


def _generic_dictionary(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    cols = [x[:, j] for j in range(x.shape[1])]
    names = [f"x{j + 1}" for j in range(x.shape[1])]
    for j in range(x.shape[1]):
        cols.extend([x[:, j] ** 2, np.sin(x[:, j]), np.cos(x[:, j])])
        names.extend([f"x{j + 1}^2", f"sin(x{j + 1})", f"cos(x{j + 1})"])
    for j in range(x.shape[1]):
        for k in range(j + 1, x.shape[1]):
            cols.append(x[:, j] * x[:, k])
            names.append(f"x{j + 1}*x{k + 1}")
    return np.stack(cols, axis=1), names
