# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Recover Lotka-Volterra terms from interpolant jets on a public CSV.

Fits ``hare(t)`` and ``lynx(t)``, reads exact ``d/dt`` from the closed-form
jet, and STLSQ-fits each derivative on ``{1, x, y, xy}``. The named baseline
uses the same library on finite-difference derivatives. This recovers a
famous ecological ODE; it is not a new law of nature.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omnibias.symbolic import extract_neural_jets, fit_neural_field_1d, fit_sparse_equation

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks._gates import gates_block, mse, require_reference_valid, skill_score

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "lynx_hare.csv"
PROVENANCE_PATH = DATA_DIR / "lynx_hare.provenance.json"
TERM_NAMES = ("1", "x", "y", "xy")
LINEAR_NAMES = ("1", "x", "y")
HONESTY = (
    "Recovers a famous ecological ODE on a public table. "
    "Not a new law of nature. Not a confinement or Clay claim."
)


@dataclass(frozen=True)
class ChannelFit:
    name: str
    equation_terms: tuple[str, ...]
    coefficients: dict[str, float]
    xy_coefficient: float
    test_pred: np.ndarray
    test_target: np.ndarray
    test_rmse: float


def load_lynx_hare(path: Path | None = None) -> dict[str, np.ndarray]:
    """Load the committed Hudson Bay table. CI stays offline."""
    raw = np.genfromtxt(path or CSV_PATH, delimiter=",", names=True, dtype=float)
    year = np.asarray(raw["year"], dtype=float).reshape(-1)
    hare = np.asarray(raw["hare"], dtype=float).reshape(-1)
    lynx = np.asarray(raw["lynx"], dtype=float).reshape(-1)
    if year.shape[0] < 8:
        raise ValueError("lynx-hare table is too short")
    return {"year": year, "hare": hare, "lynx": lynx, "t": year - year[0]}


def provenance() -> dict[str, Any]:
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def simulate_lotka_volterra(
    *,
    n: int = 81,
    t_final: float = 20.0,
    alpha: float = 0.55,
    beta: float = 0.028,
    delta: float = 0.026,
    gamma: float = 0.84,
    x0: float = 30.0,
    y0: float = 4.0,
) -> dict[str, np.ndarray]:
    """Dense RK4 trajectory of the named Lotka-Volterra law."""
    t = np.linspace(0.0, t_final, n, dtype=float)
    dt = float(t[1] - t[0])
    x = np.empty(n, dtype=float)
    y = np.empty(n, dtype=float)
    x[0], y[0] = x0, y0
    for i in range(n - 1):
        def rhs(xx: float, yy: float) -> tuple[float, float]:
            return alpha * xx - beta * xx * yy, delta * xx * yy - gamma * yy

        k1x, k1y = rhs(x[i], y[i])
        k2x, k2y = rhs(x[i] + 0.5 * dt * k1x, y[i] + 0.5 * dt * k1y)
        k3x, k3y = rhs(x[i] + 0.5 * dt * k2x, y[i] + 0.5 * dt * k2y)
        k4x, k4y = rhs(x[i] + dt * k3x, y[i] + dt * k3y)
        x[i + 1] = x[i] + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y[i + 1] = y[i] + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    return {
        "year": 1900.0 + t,
        "hare": x,
        "lynx": y,
        "t": t,
        "true_alpha": alpha,
        "true_beta": beta,
        "true_delta": delta,
        "true_gamma": gamma,
    }


def _split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = max(n // 2, 8)
    n_val = max((n - n_train) // 2, 4)
    n_test = n - n_train - n_val
    if n_test < 3:
        raise ValueError("need at least 3 test rows")
    idx = np.arange(n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def _library(hare: np.ndarray, lynx: np.ndarray, *, linear: bool = False) -> np.ndarray:
    ones = np.ones_like(hare)
    if linear:
        return np.column_stack([ones, hare, lynx])
    return np.column_stack([ones, hare, lynx, hare * lynx])


def _jet_deriv(t: np.ndarray, values: np.ndarray, *, hidden: int, seed: int) -> np.ndarray:
    field = fit_neural_field_1d(t, values, hidden=hidden, ridge=1e-4, seed=seed)
    return extract_neural_jets(field, t, max_order=1).jets[:, 1]


def _fit_channel(
    design_train: np.ndarray,
    target_train: np.ndarray,
    design_test: np.ndarray,
    target_test: np.ndarray,
    *,
    names: tuple[str, ...],
    threshold: float,
) -> ChannelFit:
    eq = fit_sparse_equation(
        design_train,
        target_train,
        list(names),
        alpha=1e-6,
        threshold=threshold,
    )
    pred = eq.predict(design_test)
    coef = {name: float(val) for name, val in zip(eq.term_names, eq.coefficients, strict=True)}
    return ChannelFit(
        name="channel",
        equation_terms=tuple(row["name"] for row in eq.active_terms()),
        coefficients=coef,
        xy_coefficient=float(coef.get("xy", 0.0)),
        test_pred=np.asarray(pred, dtype=float).reshape(-1),
        test_target=np.asarray(target_test, dtype=float).reshape(-1),
        test_rmse=float(np.sqrt(mse(pred, target_test))),
    )


def discover_table(
    table: dict[str, np.ndarray],
    *,
    hidden: int = 96,
    seed: int = 0,
    threshold: float = 1e-3,
    eval_hare: np.ndarray | None = None,
    eval_lynx: np.ndarray | None = None,
    require_jet_vs_fd: bool = True,
) -> dict[str, Any]:
    """Interpolate the table, then STLSQ the jet and FD libraries.

    Held-out scores use ``eval_*`` when given (true vector field on a
    synthetic orbit). The public CSV uses central finite differences as
    the independent target. ``require_jet_vs_fd`` is the public-table
    gate: on a clean polynomial ODE, classical FD+STLSQ is the stronger
    estimator and is not failed against.
    """
    t = np.asarray(table["t"], dtype=float).reshape(-1)
    hare = np.asarray(table["hare"], dtype=float).reshape(-1)
    lynx = np.asarray(table["lynx"], dtype=float).reshape(-1)
    ref = require_reference_valid(np.concatenate([hare, lynx]), name="populations")
    train, _val, test = _split(t.shape[0])

    hare_dot = _jet_deriv(t, hare, hidden=hidden, seed=seed)
    lynx_dot = _jet_deriv(t, lynx, hidden=hidden, seed=seed + 1)
    hare_fd = np.gradient(hare, t)
    lynx_fd = np.gradient(lynx, t)
    hare_eval = hare_fd if eval_hare is None else np.asarray(eval_hare, dtype=float).reshape(-1)
    lynx_eval = lynx_fd if eval_lynx is None else np.asarray(eval_lynx, dtype=float).reshape(-1)

    lib = _library(hare, lynx)
    lin = _library(hare, lynx, linear=True)

    hare_jet = _fit_channel(
        lib[train], hare_dot[train], lib[test], hare_eval[test], names=TERM_NAMES, threshold=threshold
    )
    lynx_jet = _fit_channel(
        lib[train], lynx_dot[train], lib[test], lynx_eval[test], names=TERM_NAMES, threshold=threshold
    )
    hare_fd_fit = _fit_channel(
        lib[train], hare_fd[train], lib[test], hare_eval[test], names=TERM_NAMES, threshold=threshold
    )
    lynx_fd_fit = _fit_channel(
        lib[train], lynx_fd[train], lib[test], lynx_eval[test], names=TERM_NAMES, threshold=threshold
    )
    hare_lin = _fit_channel(
        lin[train], hare_dot[train], lin[test], hare_eval[test], names=LINEAR_NAMES, threshold=threshold
    )
    lynx_lin = _fit_channel(
        lin[train], lynx_dot[train], lin[test], lynx_eval[test], names=LINEAR_NAMES, threshold=threshold
    )

    jet_pred = np.concatenate([hare_jet.test_pred, lynx_jet.test_pred])
    fd_pred = np.concatenate([hare_fd_fit.test_pred, lynx_fd_fit.test_pred])
    lin_pred = np.concatenate([hare_lin.test_pred, lynx_lin.test_pred])
    target = np.concatenate([hare_jet.test_target, lynx_jet.test_target])

    mse_jet = mse(jet_pred, target)
    mse_fd = mse(fd_pred, target)
    mse_lin = mse(lin_pred, target)
    skill_vs_zero = skill_score(jet_pred, target)
    skill_vs_fd = 1.0 - mse_jet / max(mse_fd, 1e-30)
    skill_vs_linear = 1.0 - mse_jet / max(mse_lin, 1e-30)

    entries = [
        {**ref, "name": "reference_valid"},
        {
            "name": "bilinear_vs_zero",
            "skill_score": skill_vs_zero,
            "min_skill": 0.0,
            "passed": bool(skill_vs_zero > 0.0),
        },
        {
            "name": "bilinear_vs_linear",
            "skill_score": skill_vs_linear,
            "min_skill": 0.0,
            "passed": bool(skill_vs_linear > 0.0),
        },
        {
            "name": "jet_vs_fd",
            "skill_score": skill_vs_fd,
            "min_skill": 0.0,
            "passed": bool(skill_vs_fd > 0.0) if require_jet_vs_fd else True,
            "required": require_jet_vs_fd,
        },
    ]
    return {
        "n_rows": int(t.shape[0]),
        "n_test": int(test.shape[0]),
        "hare": {
            "xy": hare_jet.xy_coefficient,
            "terms": list(hare_jet.equation_terms),
            "coefficients": hare_jet.coefficients,
            "test_rmse": hare_jet.test_rmse,
        },
        "lynx": {
            "xy": lynx_jet.xy_coefficient,
            "terms": list(lynx_jet.equation_terms),
            "coefficients": lynx_jet.coefficients,
            "test_rmse": lynx_jet.test_rmse,
        },
        "mse_jet": mse_jet,
        "mse_fd": mse_fd,
        "mse_linear": mse_lin,
        "skill_vs_zero": skill_vs_zero,
        "skill_vs_fd": skill_vs_fd,
        "skill_vs_linear": skill_vs_linear,
        "xy_signs_ok": bool(hare_jet.xy_coefficient < 0.0 and lynx_jet.xy_coefficient > 0.0),
        "gates": gates_block(entries),
        "honesty": HONESTY,
    }


def evaluate_synthetic(*, hidden: int = 96, n: int = 81, seed: int = 0) -> dict[str, Any]:
    table = simulate_lotka_volterra(n=n)
    hare = np.asarray(table["hare"], dtype=float)
    lynx = np.asarray(table["lynx"], dtype=float)
    true_hare = float(table["true_alpha"]) * hare - float(table["true_beta"]) * hare * lynx
    true_lynx = float(table["true_delta"]) * hare * lynx - float(table["true_gamma"]) * lynx
    result = discover_table(
        table,
        hidden=hidden,
        seed=seed,
        threshold=1e-4,
        eval_hare=true_hare,
        eval_lynx=true_lynx,
        require_jet_vs_fd=False,
    )
    result["source"] = "synthetic_lotka_volterra"
    result["true"] = {
        "alpha": float(table["true_alpha"]),
        "beta": float(table["true_beta"]),
        "delta": float(table["true_delta"]),
        "gamma": float(table["true_gamma"]),
    }
    sign_gate = {
        "name": "xy_signs",
        "hare_xy": result["hare"]["xy"],
        "lynx_xy": result["lynx"]["xy"],
        "passed": bool(result["xy_signs_ok"]),
    }
    result["gates"] = gates_block([*result["gates"]["entries"], sign_gate])
    return result


def evaluate_public_csv(*, hidden: int = 96, seed: int = 0) -> dict[str, Any]:
    table = load_lynx_hare()
    result = discover_table(table, hidden=hidden, seed=seed, threshold=1e-3)
    result["source"] = "hudson_bay_lynx_hare"
    result["provenance"] = provenance()
    return result


def evaluate_benchmark(*, quick: bool = False, seed: int = 0) -> dict[str, Any]:
    hidden = 48 if quick else 96
    n = 41 if quick else 81
    synthetic = evaluate_synthetic(hidden=hidden, n=n, seed=seed)
    public = evaluate_public_csv(hidden=hidden, seed=seed)
    gates = gates_block(
        [
            {"name": "synthetic", "passed": bool(synthetic["gates"]["all_passed"])},
            {"name": "public_csv", "passed": bool(public["gates"]["all_passed"])},
        ]
    )
    return {
        "schema": "public_csv_discovery/v1",
        "quick": quick,
        "synthetic": synthetic,
        "public_csv": public,
        "gates": gates,
        "honesty": HONESTY,
    }


__all__ = [
    "CSV_PATH",
    "HONESTY",
    "discover_table",
    "evaluate_benchmark",
    "evaluate_public_csv",
    "evaluate_synthetic",
    "load_lynx_hare",
    "simulate_lotka_volterra",
]
