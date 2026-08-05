# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Controlled synthetic benchmark for omnibias feature discovery.

Hidden target:

    y = 3*x1^2 - 2*x2*x3 + sin(x4) + noise

The benchmark is intentionally split into train/validation/test. Omnibias
discovers transformations from the training split only; validation is used only
for model/hyperparameter selection; test is held out until final scoring.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class FieldFit:
    W: np.ndarray
    beta: np.ndarray
    c: np.ndarray
    b: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    train_rmse: float


@dataclass(frozen=True)
class DiscoveredFeature:
    name: str
    kind: str
    indices: tuple[int, ...]
    score: float


def make_dataset(
    n_samples: int = 6000,
    noise_std: float = 0.1,
    seed: int = 0,
) -> SplitData:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_samples, 4))
    x[:, 3] = rng.uniform(-np.pi, np.pi, size=n_samples)
    y_clean = true_law(x)
    y = y_clean + rng.normal(0.0, noise_std, size=n_samples)
    order = rng.permutation(n_samples)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return SplitData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
    )


def true_law(x: np.ndarray) -> np.ndarray:
    return 3.0 * x[:, 0] ** 2 - 2.0 * x[:, 1] * x[:, 2] + np.sin(x[:, 3])


def fit_omnibias_field(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: int = 1200,
    ridge: float = 1e-7,
    seed: int = 0,
) -> FieldFit:
    _ensure_workspace_imports()
    jnp = _jax_numpy()
    from omnibias.jax import get_activation

    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    xs = (x_train - x_mean) / x_scale
    rng = np.random.default_rng(seed)
    W_random = rng.normal(0.0, 0.8 / np.sqrt(xs.shape[1]), size=(hidden, xs.shape[1]))
    beta_random = rng.normal(0.0, 0.5, size=hidden)
    spec = get_activation("tanh")
    phi = np.asarray(spec.forward(jnp.asarray(xs @ W_random.T + beta_random)))
    phi_aug = np.concatenate([phi, np.ones((xs.shape[0], 1))], axis=1)
    reg = ridge * np.eye(phi_aug.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(phi_aug.T @ phi_aug + reg, phi_aug.T @ y_train)
    c = coef[:hidden]
    b = float(coef[-1])

    pred = phi_aug @ coef
    return FieldFit(
        W=W_random,
        beta=beta_random,
        c=c,
        b=b,
        x_mean=x_mean,
        x_scale=x_scale,
        train_rmse=float(np.sqrt(np.mean((pred - y_train) ** 2))),
    )


def field_value_grad_hessian(field: FieldFit, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_workspace_imports()
    jnp = _jax_numpy()
    from omnibias.jax import neural_field_value_grad_hessian

    xs = (x - field.x_mean) / field.x_scale
    value, grad_s, hess_s = neural_field_value_grad_hessian(
        jnp.asarray(xs),
        field.W,
        field.beta,
        field.c,
        field.b,
        "tanh",
    )
    scale = field.x_scale
    grad = np.asarray(grad_s) / scale
    hess = np.asarray(hess_s) / (scale[None, :, None] * scale[None, None, :])
    return np.asarray(value), grad, hess


def discover_features_from_derivatives(
    x_train: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    *,
    curvature_threshold: float = 1.0,
    interaction_threshold: float = 0.35,
    periodic_threshold: float = 0.55,
) -> list[DiscoveredFeature]:
    """Use only training-set derivatives to select transformations."""
    features: list[DiscoveredFeature] = []
    dim = x_train.shape[1]
    for j in range(dim):
        curvature = np.median(np.abs(hess[:, j, j]))
        if curvature >= curvature_threshold:
            features.append(DiscoveredFeature(f"x{j + 1}^2", "square", (j,), float(curvature)))
    for j in range(dim):
        for k in range(j + 1, dim):
            interaction = np.median(np.abs(hess[:, j, k]))
            if interaction >= interaction_threshold:
                features.append(DiscoveredFeature(f"x{j + 1}*x{k + 1}", "product", (j, k), float(interaction)))
    for j in range(dim):
        sin_corr = abs(_corr(hess[:, j, j], np.sin(x_train[:, j])))
        cos_corr = abs(_corr(grad[:, j], np.cos(x_train[:, j])))
        score = max(sin_corr, cos_corr)
        if score >= periodic_threshold:
            features.append(DiscoveredFeature(f"sin(x{j + 1})", "sin", (j,), float(score)))
    return _dedupe_features(features)


def build_design_matrix(x: np.ndarray, features: list[DiscoveredFeature], include_raw: bool = True) -> tuple[np.ndarray, list[str]]:
    cols = []
    names = []
    if include_raw:
        cols.extend([x[:, j] for j in range(x.shape[1])])
        names.extend([f"x{j + 1}" for j in range(x.shape[1])])
    for feature in features:
        if feature.kind == "square":
            col = x[:, feature.indices[0]] ** 2
        elif feature.kind == "product":
            col = x[:, feature.indices[0]] * x[:, feature.indices[1]]
        elif feature.kind == "sin":
            col = np.sin(x[:, feature.indices[0]])
        else:
            raise ValueError(f"unknown feature kind {feature.kind!r}")
        cols.append(col)
        names.append(feature.name)
    return np.stack(cols, axis=1), names


def raw_design_matrix(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return x, [f"x{j + 1}" for j in range(x.shape[1])]


def full_generic_design_matrix(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """A generic non-omnibias dictionary baseline over all raw variables."""
    cols = [x[:, j] for j in range(x.shape[1])]
    names = [f"x{j + 1}" for j in range(x.shape[1])]
    for j in range(x.shape[1]):
        cols.append(x[:, j] ** 2)
        names.append(f"x{j + 1}^2")
    for j in range(x.shape[1]):
        for k in range(j + 1, x.shape[1]):
            cols.append(x[:, j] * x[:, k])
            names.append(f"x{j + 1}*x{k + 1}")
    for j in range(x.shape[1]):
        cols.append(np.sin(x[:, j]))
        names.append(f"sin(x{j + 1})")
        cols.append(np.cos(x[:, j]))
        names.append(f"cos(x{j + 1})")
    return np.stack(cols, axis=1), names


def evaluate_benchmark(
    *,
    n_samples: int = 6000,
    noise_std: float = 0.1,
    hidden: int = 1200,
    seed: int = 0,
) -> dict[str, object]:
    data = make_dataset(n_samples=n_samples, noise_std=noise_std, seed=seed)
    field = fit_omnibias_field(data.x_train, data.y_train, hidden=hidden, seed=seed)
    train_value, train_grad, train_hess = field_value_grad_hessian(field, data.x_train)
    test_value, _, _ = field_value_grad_hessian(field, data.x_test)
    discovered = discover_features_from_derivatives(data.x_train, train_grad, train_hess)

    models = {}
    for name, builder in [
        ("raw_linear", raw_design_matrix),
        ("omnibias_discovered_linear", lambda x: build_design_matrix(x, discovered)),
        ("generic_dictionary_linear", full_generic_design_matrix),
    ]:
        x_train, feature_names = builder(data.x_train)
        x_val, _ = builder(data.x_val)
        x_test, _ = builder(data.x_test)
        pred, details = _fit_tuned_ridge(x_train, data.y_train, x_val, data.y_val, x_test)
        models[name] = _metrics(data.y_test, pred) | {
            "features": feature_names,
            "n_features": len(feature_names),
            "details": details,
        }

    tree_results = _fit_optional_tree_models(data, discovered)
    models.update(tree_results)
    models["omnibias_field"] = _metrics(data.y_test, test_value) | {
        "train_rmse": field.train_rmse,
        "n_features": int(field.W.shape[0]),
    }

    discovered_payload = [
        {
            "name": feature.name,
            "kind": feature.kind,
            "indices": feature.indices,
            "score": feature.score,
        }
        for feature in discovered
    ]
    return {
        "hidden_law": "y = 3*x1^2 - 2*x2*x3 + sin(x4) + noise",
        "noise_std": noise_std,
        "n_samples": n_samples,
        "seed": seed,
        "fairness_protocol": {
            "feature_discovery_split": "train only",
            "hyperparameter_selection_split": "validation only",
            "final_scoring_split": "test only",
            "omnibias_knows_formula": False,
        },
        "discovered_features": discovered_payload,
        "field_train_rmse": field.train_rmse,
        "models": models,
    }


def write_artifacts(results: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    report = [
        "# Synthetic Feature Discovery Benchmark",
        "",
        f"Hidden law: `{results['hidden_law']}`",
        "",
        "## Discovered Features",
        "",
    ]
    for feature in results["discovered_features"]:
        report.append(f"- `{feature['name']}` ({feature['kind']}), score `{feature['score']:.4f}`")
    report.extend(["", "## Test Metrics", ""])
    models = results["models"]
    for name, metrics in sorted(models.items(), key=lambda item: item[1]["rmse"]):
        report.append(f"- `{name}`: RMSE `{metrics['rmse']:.6f}`, MAE `{metrics['mae']:.6f}`")
    report.extend(
        [
            "",
            "## Fairness",
            "",
            "- Feature discovery used training data only.",
            "- Hyperparameters were selected on validation data only.",
            "- Test data was used only for final metrics.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report) + "\n")


def _fit_tuned_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    train_s = (x_train - x_mean) / x_scale
    val_s = (x_val - x_mean) / x_scale
    test_s = (x_test - x_mean) / x_scale
    y_mean = float(y_train.mean())
    y_center = y_train - y_mean
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in alphas:
        design = np.concatenate([train_s, np.ones((train_s.shape[0], 1))], axis=1)
        reg = alpha * np.eye(design.shape[1])
        reg[-1, -1] = 0.0
        coef = np.linalg.solve(design.T @ design + reg, design.T @ y_center)
        pred_val = np.concatenate([val_s, np.ones((val_s.shape[0], 1))], axis=1) @ coef + y_mean
        val_rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        if best is None or val_rmse < best[0]:
            best = (val_rmse, alpha, coef)
    assert best is not None
    _, alpha, coef = best
    # Refit on train+validation with selected alpha.
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)
    x_mean = x_fit.mean(axis=0)
    x_scale = x_fit.std(axis=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    fit_s = (x_fit - x_mean) / x_scale
    test_s = (x_test - x_mean) / x_scale
    y_mean = float(y_fit.mean())
    design = np.concatenate([fit_s, np.ones((fit_s.shape[0], 1))], axis=1)
    reg = alpha * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ (y_fit - y_mean))
    pred_test = np.concatenate([test_s, np.ones((test_s.shape[0], 1))], axis=1) @ coef + y_mean
    return pred_test, {"alpha": float(alpha), "validation_rmse": float(best[0])}


def _fit_optional_tree_models(data: SplitData, discovered: list[DiscoveredFeature]) -> dict[str, dict[str, object]]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return {}
    out = {}
    for name, builder in [
        ("raw_hist_gradient_boosting", raw_design_matrix),
        ("omnibias_discovered_hist_gradient_boosting", lambda x: build_design_matrix(x, discovered)),
    ]:
        x_train, feature_names = builder(data.x_train)
        x_val, _ = builder(data.x_val)
        x_test, _ = builder(data.x_test)
        candidates = [
            {"max_iter": n, "learning_rate": lr, "max_leaf_nodes": leaves}
            for n in [80, 160]
            for lr in [0.05, 0.1]
            for leaves in [15, 31]
        ]
        best: tuple[float, dict[str, float | int], HistGradientBoostingRegressor] | None = None
        for params in candidates:
            model = HistGradientBoostingRegressor(random_state=0, **params)
            model.fit(x_train, data.y_train)
            pred_val = model.predict(x_val)
            rmse = float(np.sqrt(np.mean((pred_val - data.y_val) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, params, model)
        assert best is not None
        _, params, _ = best
        x_fit = np.concatenate([x_train, x_val], axis=0)
        y_fit = np.concatenate([data.y_train, data.y_val], axis=0)
        model = HistGradientBoostingRegressor(random_state=0, **params)
        model.fit(x_fit, y_fit)
        pred_test = model.predict(x_test)
        out[name] = _metrics(data.y_test, pred_test) | {
            "features": feature_names,
            "n_features": len(feature_names),
            "details": {"params": params, "validation_rmse": best[0]},
        }
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }


def _dedupe_features(features: list[DiscoveredFeature]) -> list[DiscoveredFeature]:
    best: dict[str, DiscoveredFeature] = {}
    for feature in features:
        old = best.get(feature.name)
        if old is None or feature.score > old.score:
            best[feature.name] = feature
    return sorted(best.values(), key=lambda feature: (-feature.score, feature.name))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-12:
        return 0.0
    return float(aa @ bb / denom)


def _ensure_workspace_imports() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for rel in ["packages/omnibias-core/src", "packages/omnibias-jax/src"]:
        path = str(repo_root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def _jax_numpy():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return jnp
