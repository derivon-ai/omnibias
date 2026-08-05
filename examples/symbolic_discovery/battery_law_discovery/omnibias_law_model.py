# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Omnibias neural-field model and sparse battery degradation law."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _ensure_workspace_imports() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for rel in [
        "packages/omnibias-core/src",
        "packages/omnibias-jax/src",
    ]:
        path = str(repo_root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


@dataclass(frozen=True)
class FieldParams:
    W: np.ndarray
    beta: np.ndarray
    c: np.ndarray
    b: float


@dataclass(frozen=True)
class FieldFit:
    params: FieldParams
    losses: list[float]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    activation: str


@dataclass(frozen=True)
class SparseLaw:
    names: list[str]
    coef: np.ndarray
    intercept: float
    threshold: float
    kind: str = "direct"
    operator_set: str = ""

    def equation(self, lhs: str = "dq/dn") -> str:
        terms = [f"{self.intercept:.6g}"] if self.kind == "degradation" else [f"{self.intercept:+.6g}"]
        for name, value in zip(self.names, self.coef, strict=True):
            if abs(value) > 0:
                terms.append(f"{value:+.6g}*{name}")
        rhs = " ".join(terms)
        if self.kind == "degradation":
            return f"{lhs} = -q*({rhs})"
        return f"{lhs} = {rhs}"


@dataclass(frozen=True)
class OmnibiasLawResult:
    field: FieldFit
    law: SparseLaw
    train_rmse: float
    test_rmse: float
    train_derivative_rmse: float
    train_predictions: np.ndarray
    predictions: np.ndarray
    derivative_train: np.ndarray
    derivative_test: np.ndarray
    second_derivative_train: np.ndarray
    second_derivative_test: np.ndarray


def fit_omnibias_law(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden: int = 48,
    steps: int = 800,
    lr: float = 3e-3,
    activation: str = "tanh",
    sparsity_threshold: float = 1e-3,
    law_mode: str = "direct",
    operator_set: str = "stress",
    seed: int = 0,
) -> OmnibiasLawResult:
    """Fit a smooth capacity field, then identify a sparse degradation law."""
    _ensure_workspace_imports()
    jnp = _jax_numpy()
    from omnibias.jax import get_activation, neural_field_value_grad_hessian

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    xtr = (x_train - mean) / scale
    xte = (x_test - mean) / scale

    # Random-feature ridge fit: fixes W/beta, solves the output layer exactly.
    # This makes the demo deterministic and quick while still using the
    # omnibias closed-form derivative tower for operator extraction.
    del steps, lr
    rng = np.random.default_rng(seed)
    Wf = rng.normal(0.0, 0.8 / np.sqrt(xtr.shape[1]), size=(hidden, xtr.shape[1]))
    betaf = rng.normal(0.0, 0.4, size=(hidden,))
    spec = get_activation(activation)
    phi = np.asarray(spec.forward(jnp.asarray(xtr @ Wf.T + betaf)))
    phi_aug = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)
    reg = 1e-5 * np.eye(phi_aug.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(phi_aug.T @ phi_aug + reg, phi_aug.T @ y_train)
    cf = coef[:-1]
    bf = float(coef[-1])
    train_pred_np = phi_aug @ coef
    losses = [float(np.mean((train_pred_np - y_train) ** 2))]

    train_value, train_grad, train_hess = neural_field_value_grad_hessian(
        jnp.asarray(xtr), Wf, betaf, cf, bf, activation
    )
    test_value, test_grad, test_hess = neural_field_value_grad_hessian(
        jnp.asarray(xte), Wf, betaf, cf, bf, activation
    )
    # Chain rule back to raw cycle_norm. Feature 0 is normalized cycle index.
    dqdn_train = np.asarray(train_grad[:, 0]) / scale[0]
    dqdn_test = np.asarray(test_grad[:, 0]) / scale[0]
    d2qdn2_train = np.asarray(train_hess[:, 0, 0]) / (scale[0] ** 2)
    d2qdn2_test = np.asarray(test_hess[:, 0, 0]) / (scale[0] ** 2)

    train_q = np.asarray(train_value)
    test_q = np.asarray(test_value)
    if law_mode == "physics":
        if operator_set == "auto":
            law = select_physics_operator_law(
                x_train=x_train,
                q=train_q,
                dqdn=dqdn_train,
                threshold=sparsity_threshold,
                seed=seed,
            )
        else:
            library_train, names = build_physics_library(x_train=x_train, q=train_q, operator_set=operator_set)
            law = fit_physics_constrained_law(
                library_train,
                dqdn_train,
                train_q,
                names,
                threshold=sparsity_threshold,
                operator_set=operator_set,
            )
        library_test, _ = build_physics_library(x_train=x_test, q=test_q, operator_set=law.operator_set)
        law_pred_dqdn = predict_law(law, library_test, q=test_q)
    elif law_mode == "direct":
        library_train, names = build_operator_library(
            x_train=x_train,
            q=train_q,
            d2qdn2=d2qdn2_train,
        )
        law = fit_sparse_law(library_train, dqdn_train, names, threshold=sparsity_threshold)
        library_test, _ = build_operator_library(
            x_train=x_test,
            q=test_q,
            d2qdn2=d2qdn2_test,
        )
        law_pred_dqdn = predict_law(law, library_test)
    else:
        raise ValueError(f"unknown law_mode {law_mode!r}; expected 'direct' or 'physics'")
    derivative_rmse = float(np.sqrt(np.mean((law_pred_dqdn - dqdn_test) ** 2)))

    pred = np.asarray(test_value)
    field = FieldFit(
        params=FieldParams(
            W=np.asarray(Wf),
            beta=np.asarray(betaf),
            c=np.asarray(cf),
            b=float(bf),
        ),
        losses=losses,
        feature_mean=mean,
        feature_scale=scale,
        activation=activation,
    )
    return OmnibiasLawResult(
        field=field,
        law=law,
        train_rmse=float(np.sqrt(np.mean((np.asarray(train_value) - y_train) ** 2))),
        test_rmse=float(np.sqrt(np.mean((pred - y_test) ** 2))),
        train_derivative_rmse=derivative_rmse,
        train_predictions=train_q,
        predictions=pred,
        derivative_train=dqdn_train,
        derivative_test=dqdn_test,
        second_derivative_train=d2qdn2_train,
        second_derivative_test=d2qdn2_test,
    )


def build_operator_library(
    x_train: np.ndarray,
    q: np.ndarray,
    d2qdn2: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build candidate terms for sparse degradation-law fitting."""
    n = x_train[:, 0]
    # The remaining columns are normalized protocol/telemetry features.
    terms = [
        ("q", q),
        ("n", n),
        ("q*n", q * n),
        ("q^2", q**2),
        ("d2qdn2", d2qdn2),
    ]
    for j in range(1, x_train.shape[1]):
        terms.append((f"x{j}", x_train[:, j]))
        terms.append((f"q*x{j}", q * x_train[:, j]))
    names = [name for name, _ in terms]
    mat = np.stack([np.asarray(values, dtype=float) for _, values in terms], axis=1)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, names


def build_physics_library(
    x_train: np.ndarray,
    q: np.ndarray,
    operator_set: str = "stress",
) -> tuple[np.ndarray, list[str]]:
    """Build nonnegative degradation-rate terms for dq/dn = -q*k(...)."""
    n = np.clip(np.asarray(x_train[:, 0], dtype=float), 0.0, None)
    fade = np.clip(1.0 - np.asarray(q, dtype=float), 0.0, None)
    q_arr = np.asarray(q, dtype=float)
    terms = [
        ("n", n),
        ("fade", fade),
    ]
    if operator_set in {"poly", "stress", "stress_interactions", "capacity"}:
        terms.extend(
            [
                ("n^2", n**2),
                ("n*fade", n * fade),
            ]
        )
    if operator_set == "capacity":
        terms.extend(
            [
                ("q", q_arr),
                ("q*n", q_arr * n),
                ("q^2", q_arr**2),
            ]
        )
    if operator_set in {"stress", "stress_interactions"}:
        for j in range(1, x_train.shape[1]):
            stress = np.asarray(x_train[:, j], dtype=float) ** 2
            terms.append((f"x{j}^2", stress))
            terms.append((f"n*x{j}^2", n * stress))
            if operator_set == "stress_interactions":
                terms.append((f"fade*x{j}^2", fade * stress))
                terms.append((f"n*fade*x{j}^2", n * fade * stress))
    if operator_set not in {"minimal", "poly", "stress", "stress_interactions", "capacity"}:
        raise ValueError(f"unknown physics operator_set {operator_set!r}")
    names = [name for name, _ in terms]
    mat = np.stack([np.asarray(values, dtype=float) for _, values in terms], axis=1)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, names


def fit_sparse_law(
    library: np.ndarray,
    target: np.ndarray,
    names: list[str],
    *,
    ridge: float = 1e-5,
    threshold: float = 1e-3,
    n_refit: int = 5,
) -> SparseLaw:
    """Sequential thresholded ridge regression, SINDy-style."""
    x = np.asarray(library, dtype=float)
    y = np.asarray(target, dtype=float)
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    y_mean = float(y.mean())
    xs = (x - x_mean) / x_scale
    ys = y - y_mean
    active = np.ones(xs.shape[1], dtype=bool)
    coef_s = np.zeros(xs.shape[1])
    for _ in range(n_refit):
        xa = xs[:, active]
        lhs = xa.T @ xa + ridge * np.eye(xa.shape[1])
        coef_active = np.linalg.solve(lhs, xa.T @ ys)
        coef_s[:] = 0.0
        coef_s[active] = coef_active
        active = np.abs(coef_s) >= threshold
        if not active.any():
            active[np.argmax(np.abs(coef_s))] = True
            break
    coef = coef_s / x_scale
    coef[~active] = 0.0
    intercept = y_mean - float(x_mean @ coef)
    return SparseLaw(names=names, coef=coef, intercept=intercept, threshold=threshold)


def fit_physics_constrained_law(
    library: np.ndarray,
    dqdn: np.ndarray,
    q: np.ndarray,
    names: list[str],
    *,
    threshold: float = 1e-3,
    operator_set: str = "",
) -> SparseLaw:
    """Fit a monotone degradation law: dq/dn = -q * k, k >= 0."""
    x = np.asarray(library, dtype=float)
    q_safe = np.maximum(np.asarray(q, dtype=float), 1e-6)
    target = np.maximum(-np.asarray(dqdn, dtype=float) / q_safe, 0.0)
    design = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    scale = design.std(axis=0)
    scale[0] = 1.0
    scale = np.where(scale < 1e-12, 1.0, scale)
    coef_scaled = _nonnegative_lstsq(design / scale, target)
    coef_all = coef_scaled / scale
    coef_all[coef_all < threshold] = 0.0
    return SparseLaw(
        names=names,
        coef=coef_all[1:],
        intercept=float(coef_all[0]),
        threshold=threshold,
        kind="degradation",
        operator_set=operator_set,
    )


def select_physics_operator_law(
    x_train: np.ndarray,
    q: np.ndarray,
    dqdn: np.ndarray,
    *,
    threshold: float = 1e-3,
    seed: int = 0,
    candidate_sets: tuple[str, ...] = ("minimal", "poly", "capacity", "stress", "stress_interactions"),
) -> SparseLaw:
    """Select a physics operator library using a held-out derivative split."""
    rng = np.random.default_rng(seed)
    n_rows = x_train.shape[0]
    val_size = max(1, int(round(0.2 * n_rows)))
    perm = rng.permutation(n_rows)
    val_idx = perm[:val_size]
    fit_idx = perm[val_size:]
    if fit_idx.size == 0:
        fit_idx = val_idx

    best: tuple[float, SparseLaw] | None = None
    for name in candidate_sets:
        lib_fit, names = build_physics_library(x_train[fit_idx], q[fit_idx], operator_set=name)
        law = fit_physics_constrained_law(
            lib_fit,
            dqdn[fit_idx],
            q[fit_idx],
            names,
            threshold=threshold,
            operator_set=name,
        )
        lib_val, _ = build_physics_library(x_train[val_idx], q[val_idx], operator_set=name)
        pred = predict_law(law, lib_val, q=q[val_idx])
        rmse = float(np.sqrt(np.mean((pred - dqdn[val_idx]) ** 2)))
        active = int(np.count_nonzero(law.coef))
        score = rmse + 1e-4 * active
        if best is None or score < best[0]:
            best = (score, law)
    assert best is not None

    selected = best[1].operator_set
    full_lib, full_names = build_physics_library(x_train, q, operator_set=selected)
    return fit_physics_constrained_law(
        full_lib,
        dqdn,
        q,
        full_names,
        threshold=threshold,
        operator_set=selected,
    )


def predict_law(law: SparseLaw, library: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    value = law.intercept + np.asarray(library, dtype=float) @ law.coef
    if law.kind == "degradation":
        if q is None:
            raise ValueError("degradation laws require q for dq/dn prediction")
        return -np.asarray(q, dtype=float) * np.maximum(value, 0.0)
    return value


def build_omnibias_feature_matrix(
    x: np.ndarray,
    *,
    q_field: np.ndarray,
    dqdn: np.ndarray,
    d2qdn2: np.ndarray,
    law: SparseLaw,
) -> tuple[np.ndarray, list[str]]:
    """Augment raw features with omnibias operator and discovered-law features."""
    q_arr = np.asarray(q_field, dtype=float)
    dq_arr = np.asarray(dqdn, dtype=float)
    d2_arr = np.asarray(d2qdn2, dtype=float)
    if law.kind == "degradation":
        law_library, law_names = build_physics_library(np.asarray(x, dtype=float), q_arr, operator_set=law.operator_set)
        law_dqdn = predict_law(law, law_library, q=q_arr)
        law_rate = np.maximum(-law_dqdn / np.maximum(q_arr, 1e-6), 0.0)
    else:
        law_library, law_names = build_operator_library(np.asarray(x, dtype=float), q_arr, d2_arr)
        law_dqdn = predict_law(law, law_library)
        law_rate = -law_dqdn / np.maximum(q_arr, 1e-6)
    residual = dq_arr - law_dqdn
    feature_blocks = [
        np.asarray(x, dtype=float),
        q_arr[:, None],
        dq_arr[:, None],
        d2_arr[:, None],
        law_dqdn[:, None],
        law_rate[:, None],
        residual[:, None],
        law_library,
    ]
    names = (
        [f"raw_x{j}" for j in range(np.asarray(x).shape[1])]
        + ["q_field", "dqdn_field", "d2qdn2_field", "law_dqdn", "law_rate", "law_residual"]
        + [f"law_term:{name}" for name in law_names]
    )
    matrix = np.concatenate(feature_blocks, axis=1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, names


def rollout_law(
    law: SparseLaw,
    x0_features: np.ndarray,
    q0: float,
    n_grid: np.ndarray,
) -> np.ndarray:
    """Cheap Euler rollout using only law terms that can be evaluated locally."""
    q = np.empty_like(n_grid, dtype=float)
    q[0] = q0
    dx = np.asarray(x0_features, dtype=float)
    for i in range(1, n_grid.size):
        dt = float(n_grid[i] - n_grid[i - 1])
        if law.kind == "degradation":
            row = _rollout_physics_library_row(law.names, n_grid[i - 1], q[i - 1], dx)
            rate = max(float(law.intercept + row @ law.coef), 0.0)
            q_next = q[i - 1] - dt * q[i - 1] * rate
            q[i] = min(q[i - 1], max(q_next, 0.0))
        else:
            row = _rollout_library_row(law.names, n_grid[i - 1], q[i - 1], dx)
            q[i] = q[i - 1] + dt * float(law.intercept + row @ law.coef)
    return q


def _rollout_library_row(names: list[str], n: float, q: float, x: np.ndarray) -> np.ndarray:
    values = []
    for name in names:
        if name == "q":
            values.append(q)
        elif name == "n":
            values.append(n)
        elif name == "q*n":
            values.append(q * n)
        elif name == "q^2":
            values.append(q * q)
        elif name in {"dqdn", "d2qdn2"}:
            values.append(0.0)
        elif name.startswith("q*x"):
            j = int(name[3:])
            values.append(q * x[j])
        elif name.startswith("x"):
            j = int(name[1:])
            values.append(x[j])
        else:
            values.append(0.0)
    return np.asarray(values, dtype=float)


def _rollout_physics_library_row(names: list[str], n: float, q: float, x: np.ndarray) -> np.ndarray:
    fade = max(1.0 - q, 0.0)
    values = []
    for name in names:
        if name == "n":
            values.append(max(n, 0.0))
        elif name == "n^2":
            values.append(max(n, 0.0) ** 2)
        elif name == "fade":
            values.append(fade)
        elif name == "n*fade":
            values.append(max(n, 0.0) * fade)
        elif name == "q":
            values.append(q)
        elif name == "q*n":
            values.append(q * max(n, 0.0))
        elif name == "q^2":
            values.append(q * q)
        elif name.startswith("n*fade*x") and name.endswith("^2"):
            j = int(name[8:-2])
            values.append(max(n, 0.0) * fade * x[j] ** 2)
        elif name.startswith("fade*x") and name.endswith("^2"):
            j = int(name[6:-2])
            values.append(fade * x[j] ** 2)
        elif name.startswith("n*x") and name.endswith("^2"):
            j = int(name[3:-2])
            values.append(max(n, 0.0) * x[j] ** 2)
        elif name.startswith("x") and name.endswith("^2"):
            j = int(name[1:-2])
            values.append(x[j] ** 2)
        else:
            values.append(0.0)
    return np.asarray(values, dtype=float)


def _nonnegative_lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import nnls

        return nnls(x, y, maxiter=max(1000, 10 * x.shape[1]))[0]
    except Exception:
        coef = np.zeros(x.shape[1], dtype=float)
        lipschitz = float(np.linalg.norm(x, ord=2) ** 2)
        step = 1.0 / max(lipschitz, 1e-12)
        for _ in range(5000):
            grad = x.T @ (x @ coef - y)
            coef = np.maximum(coef - step * grad, 0.0)
        return coef


def _jax_numpy():
    try:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "This demo needs JAX for the omnibias closed-form neural field. "
            "Install the workspace with `uv sync --all-extras --dev` or install "
            "`omnibias-jax[jax]`."
        ) from exc

    jax.config.update("jax_enable_x64", True)
    return jnp
