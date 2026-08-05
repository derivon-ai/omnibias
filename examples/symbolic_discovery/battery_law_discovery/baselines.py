# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Low-dependency baselines for battery capacity and cycle-life prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .features import FeatureBundle, cell_indices
    from .omnibias_law_model import (
        build_physics_library,
        fit_physics_constrained_law,
        rollout_law,
    )
except ImportError:  # pragma: no cover
    from features import FeatureBundle, cell_indices
    from omnibias_law_model import build_physics_library, fit_physics_constrained_law, rollout_law


@dataclass(frozen=True)
class BaselineResult:
    name: str
    rmse_capacity: float
    mae_capacity: float
    eol_mae_cycles: float
    predictions: np.ndarray
    details: dict[str, object] | None = None


def ridge_capacity_baseline(
    train: FeatureBundle,
    test: FeatureBundle,
    alpha: float = 1e-3,
) -> BaselineResult:
    """Global ridge regression on simple polynomial features."""
    x_train = _design_matrix(train.x)
    x_test = _design_matrix(test.x)
    xtx = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
    coef = np.linalg.solve(xtx, x_train.T @ train.y)
    pred = x_test @ coef
    return _result("ridge_capacity", test, pred)


def empirical_per_cell_baseline(
    test: FeatureBundle,
    early_fraction: float = 0.2,
    mode: str = "linear",
) -> BaselineResult:
    """Fit a per-cell empirical fade curve on early observed cycles."""
    pred = np.empty_like(test.y)
    for _, idx in cell_indices(test.cell_id).items():
        n = test.cycle_norm[idx]
        q = test.y[idx]
        early = n <= max(float(n.min()) + early_fraction * (float(n.max()) - float(n.min())), 1e-12)
        if early.sum() < 3:
            early[: min(5, early.size)] = True
        pred[idx] = _fit_empirical(n[early], q[early], n, mode=mode)
    return _result(f"empirical_{mode}", test, pred)


def sklearn_tree_baseline(train: FeatureBundle, test: FeatureBundle) -> BaselineResult | None:
    """Optional tree baseline when scikit-learn is available."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return None
    model = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=100)
    model.fit(train.x, train.y)
    pred = model.predict(test.x)
    return _result("sklearn_gradient_boosting", test, pred)


def tuned_feature_model_baselines(
    train: FeatureBundle,
    test: FeatureBundle,
    *,
    raw_train_x: np.ndarray,
    raw_test_x: np.ndarray,
    augmented_train_x: np.ndarray,
    augmented_test_x: np.ndarray,
    seed: int = 0,
) -> tuple[list[BaselineResult], dict[str, dict[str, object]]]:
    """Tune raw-vs-omnibias feature models on a cell-level validation split."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return [], {
            "sklearn_tuned": {
                "skipped": True,
                "reason": "scikit-learn is not installed",
            }
        }

    fit_idx, val_idx = _cell_validation_indices(train, seed=seed)
    tune_fit_idx = fit_idx[_even_sample_indices(fit_idx.size, max_rows=20_000)]
    tune_val_idx = val_idx[_even_sample_indices(val_idx.size, max_rows=10_000)]
    final_fit_idx = np.arange(train.y.size)
    sampled_final_fit_idx = final_fit_idx[_even_sample_indices(final_fit_idx.size, max_rows=30_000)]
    benchmark_specs = [
        (
            "ridge",
            [
                ({"alpha": alpha}, make_pipeline(StandardScaler(), Ridge(alpha=alpha)))
                for alpha in [1e-4, 1e-2, 1.0, 10.0]
            ],
        ),
        (
            "hist_gradient_boosting",
            [
                (
                    {"max_iter": n, "max_leaf_nodes": leaves, "learning_rate": lr, "l2_regularization": l2},
                    HistGradientBoostingRegressor(
                        random_state=seed,
                        max_iter=n,
                        max_leaf_nodes=leaves,
                        learning_rate=lr,
                        l2_regularization=l2,
                    ),
                )
                for n in [80, 160]
                for leaves in [31]
                for lr in [0.1]
                for l2 in [0.0]
            ],
        ),
    ]

    outputs: list[BaselineResult] = []
    details: dict[str, dict[str, object]] = {}
    for feature_name, x_train_all, x_test in [
        ("raw", raw_train_x, raw_test_x),
        ("omnibias_augmented", augmented_train_x, augmented_test_x),
    ]:
        for model_name, candidates in benchmark_specs:
            best = _select_model(
                candidates,
                x_train_all[tune_fit_idx],
                train.y[tune_fit_idx],
                x_train_all[tune_val_idx],
                train.y[tune_val_idx],
            )
            params, model, val_rmse = best
            train_idx = final_fit_idx if model_name == "ridge" else sampled_final_fit_idx
            model.fit(x_train_all[train_idx], train.y[train_idx])
            pred = model.predict(x_test)
            name = f"tuned_{model_name}_{feature_name}"
            result = _result(name, test, pred)
            result = BaselineResult(
                name=result.name,
                rmse_capacity=result.rmse_capacity,
                mae_capacity=result.mae_capacity,
                eol_mae_cycles=result.eol_mae_cycles,
                predictions=result.predictions,
                details={
                    "feature_set": feature_name,
                    "model": model_name,
                    "validation_rmse": float(val_rmse),
                    "params": params,
                    "n_features": int(x_train_all.shape[1]),
                    "fit_rows": int(train_idx.size),
                },
            )
            outputs.append(result)
            details[name] = result.details or {}
    return outputs, details


def finite_difference_sindy_baseline(
    train: FeatureBundle,
    test: FeatureBundle,
    threshold: float = 1e-3,
) -> BaselineResult:
    """Symbolic baseline: finite-difference derivatives + constrained SINDy law."""
    dqdn = _finite_difference_derivative(train)
    library, names = build_physics_library(train.x, train.y)
    law = fit_physics_constrained_law(library, dqdn, train.y, names, threshold=threshold)
    pred = np.empty_like(test.y)
    for _, idx in cell_indices(test.cell_id).items():
        order = np.argsort(test.cycle_norm[idx])
        ordered_idx = idx[order]
        n = test.cycle_norm[ordered_idx]
        rolled = rollout_law(law, test.x[ordered_idx[0]], float(test.y[ordered_idx[0]]), n)
        pred[ordered_idx] = rolled
    return _result("finite_difference_physics_sindy", test, pred)


def _design_matrix(x: np.ndarray) -> np.ndarray:
    cycle = x[:, :1]
    return np.concatenate([np.ones((x.shape[0], 1)), x, cycle**2, x[:, 1:] * cycle], axis=1)


def _fit_empirical(n_train: np.ndarray, q_train: np.ndarray, n_all: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        coef = np.polyfit(n_train, q_train, 1)
        return np.polyval(coef, n_all)
    if mode == "sqrt":
        z_train = np.sqrt(np.maximum(n_train, 0.0))
        coef = np.polyfit(z_train, q_train, 1)
        return np.polyval(coef, np.sqrt(np.maximum(n_all, 0.0)))
    if mode == "exp":
        y = np.log(np.maximum(q_train, 1e-6))
        coef = np.polyfit(n_train, y, 1)
        return np.exp(np.polyval(coef, n_all))
    raise ValueError(f"unknown empirical baseline mode {mode!r}")


def _finite_difference_derivative(bundle: FeatureBundle) -> np.ndarray:
    dqdn = np.empty_like(bundle.y)
    for _, idx in cell_indices(bundle.cell_id).items():
        order = np.argsort(bundle.cycle_norm[idx])
        ordered_idx = idx[order]
        n = bundle.cycle_norm[ordered_idx]
        q = bundle.y[ordered_idx]
        if n.size < 3 or np.allclose(n, n[0]):
            deriv = np.zeros_like(q)
        else:
            deriv = np.gradient(q, n, edge_order=1)
        dqdn[ordered_idx] = deriv
    return np.nan_to_num(dqdn, nan=0.0, posinf=0.0, neginf=0.0)


def _cell_validation_indices(bundle: FeatureBundle, seed: int = 0, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    cells = np.unique(bundle.cell_id)
    rng = np.random.default_rng(seed)
    shuffled = np.array(cells)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(cells.size * val_fraction)))
    val_cells = set(shuffled[:n_val])
    val_mask = np.isin(bundle.cell_id, np.asarray(sorted(val_cells), dtype=str))
    fit_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if fit_idx.size == 0:
        return val_idx, val_idx
    return fit_idx, val_idx


def _even_sample_indices(n_rows: int, max_rows: int) -> np.ndarray:
    if n_rows <= max_rows:
        return np.arange(n_rows)
    return np.linspace(0, n_rows - 1, max_rows, dtype=int)


def _select_model(
    candidates: list[tuple[dict[str, object], object]],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, object], object, float]:
    best: tuple[dict[str, object], object, float] | None = None
    for params, model in candidates:
        model.fit(x_fit, y_fit)
        pred = model.predict(x_val)
        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        if best is None or rmse < best[2]:
            best = (params, model, rmse)
    assert best is not None
    return best


def _result(name: str, test: FeatureBundle, pred: np.ndarray) -> BaselineResult:
    pred = np.asarray(pred, dtype=float)
    err = pred - test.y
    return BaselineResult(
        name=name,
        rmse_capacity=float(np.sqrt(np.mean(err**2))),
        mae_capacity=float(np.mean(np.abs(err))),
        eol_mae_cycles=float(_cycle_life_mae(test, pred)),
        predictions=pred,
        details=None,
    )


def _cycle_life_mae(bundle: FeatureBundle, pred: np.ndarray, threshold: float = 0.8) -> float:
    errors = []
    for _, idx in cell_indices(bundle.cell_id).items():
        n = bundle.cycle_norm[idx]
        true = bundle.y[idx]
        p = pred[idx]
        true_n = _first_crossing(n, true, threshold)
        pred_n = _first_crossing(n, p, threshold)
        errors.append(abs(pred_n - true_n) * max(float(idx.size), 1.0))
    return float(np.mean(errors)) if errors else float("nan")


def _first_crossing(n: np.ndarray, q: np.ndarray, threshold: float) -> float:
    below = np.flatnonzero(q <= threshold)
    if below.size == 0:
        return float(n[-1])
    return float(n[below[0]])
