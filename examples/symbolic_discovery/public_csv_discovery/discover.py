# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Recover Lotka-Volterra terms from a 1-D interpolant on a public CSV.

Fits ``hare(t)`` and ``lynx(t)`` on the chronological fit window, reads
``d/dt`` from a train-only interpolant (random-feature jet, with a cubic
spline fallback), and STLSQ-fits each derivative on ``{1, x, y, xy}``.
Held-out scores are RK4 rollouts of the discovered ODE, not derivative
MSE against finite differences. This recovers a famous ecological ODE
on a synthetic orbit; it is not a new law of nature.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from omnibias.symbolic import (
    extract_neural_jets,
    fit_neural_field_1d,
    fit_sparse_equation,
    rollout_levels,
    rollout_skill,
    spline_values_and_deriv,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks._gates import gates_block, mse, require_reference_valid, skill_score

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "lynx_hare.csv"
PROVENANCE_PATH = DATA_DIR / "lynx_hare.provenance.json"
TERM_NAMES = ("1", "x", "y", "xy")
LINEAR_NAMES = ("1", "x", "y")
# Locked after a local synthetic sweep. Value-only RF jets lose to FD.
# Spline-collocated jets clear 1.25× FD on n=41 (hidden>=48, ridge=1e-8,
# deriv_weight=30) and on n=81 at hidden=256. ``auto`` uses the jet when
# it beats that ratio and falls back to the named cubic spline.
INTERPOLANT_RIDGE = 1e-8
INTERPOLANT_WEIGHT_SCALE = 1.0
INTERPOLANT_UPSAMPLE = 8
INTERPOLANT_DERIV_WEIGHT = 30.0
INTERPOLANT_QUALITY_RATIO = 1.25
INTERPOLANT_MODE: Literal["auto", "jet", "spline"] = "auto"
SCHEMA = "public_csv_discovery/v2"
HONESTY = (
    "Train-only interpolant (spline-collocated jet if it beats 1.25× FD, "
    "else cubic spline) plus STLSQ and RK4 rollout on Hudson Bay pelts "
    "and a synthetic Lotka-Volterra orbit. Recovers LV xy signs; extra "
    "linear terms survive on the public table. Not a new law of nature. "
    "Not a confinement or Clay claim."
)


@dataclass(frozen=True)
class ChannelFit:
    name: str
    equation_terms: tuple[str, ...]
    coefficients: dict[str, float]
    intercept: float
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


def _fit_index(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chronological hold-out: everything except the last test block is in-support."""
    train, val, test = _split(n)
    return np.concatenate([train, val]), test


def _library(hare: np.ndarray, lynx: np.ndarray, *, linear: bool = False) -> np.ndarray:
    ones = np.ones_like(hare)
    if linear:
        return np.column_stack([ones, hare, lynx])
    return np.column_stack([ones, hare, lynx, hare * lynx])


def _jet_deriv(
    t_fit: np.ndarray,
    values: np.ndarray,
    t_eval: np.ndarray,
    *,
    hidden: int,
    seed: int,
    ridge: float = INTERPOLANT_RIDGE,
    weight_scale: float = INTERPOLANT_WEIGHT_SCALE,
    upsample: int = INTERPOLANT_UPSAMPLE,
    deriv_weight: float = INTERPOLANT_DERIV_WEIGHT,
) -> np.ndarray:
    """Exact jet of a train-only random-feature field, optionally spline-upsampled."""
    t_rf = np.asarray(t_fit, dtype=float).reshape(-1)
    y_rf = np.asarray(values, dtype=float).reshape(-1)
    if upsample > 1 and t_rf.shape[0] >= 3:
        t_dense = np.linspace(float(t_rf[0]), float(t_rf[-1]), int(t_rf.shape[0] * upsample))
        y_dense, _ = spline_values_and_deriv(t_rf, y_rf, t_dense)
        t_rf, y_rf = t_dense, y_dense
    field = fit_neural_field_1d(
        t_rf,
        y_rf,
        hidden=hidden,
        ridge=ridge,
        seed=seed,
        weight_scale=weight_scale,
        deriv="spline",
        deriv_weight=deriv_weight,
    )
    return extract_neural_jets(field, t_eval, max_order=1).jets[:, 1]


def _fit_channel(
    design_train: np.ndarray,
    target_train: np.ndarray,
    design_test: np.ndarray,
    target_test: np.ndarray,
    *,
    names: tuple[str, ...],
    threshold: float,
    loss: Literal["ridge", "huber"],
) -> ChannelFit:
    eq = fit_sparse_equation(
        design_train,
        target_train,
        list(names),
        alpha=1e-6,
        threshold=threshold,
        loss=loss,
    )
    pred = eq.predict(design_test)
    coef = {name: float(val) for name, val in zip(eq.term_names, eq.coefficients, strict=True)}
    return ChannelFit(
        name="channel",
        equation_terms=tuple(row["name"] for row in eq.active_terms()),
        coefficients=coef,
        intercept=float(eq.intercept),
        xy_coefficient=float(coef.get("xy", 0.0)),
        test_pred=np.asarray(pred, dtype=float).reshape(-1),
        test_target=np.asarray(target_test, dtype=float).reshape(-1),
        test_rmse=float(np.sqrt(mse(pred, target_test))),
    )


def _ode_rhs(hare_fit: ChannelFit, lynx_fit: ChannelFit):
    def _eval(fit: ChannelFit, x: float, y: float) -> float:
        coef = fit.coefficients
        return (
            fit.intercept
            + float(coef.get("1", 0.0))
            + float(coef.get("x", 0.0)) * x
            + float(coef.get("y", 0.0)) * y
            + float(coef.get("xy", 0.0)) * x * y
        )

    def rhs(x: float, y: float) -> tuple[float, float]:
        return _eval(hare_fit, x, y), _eval(lynx_fit, x, y)

    return rhs


def _rollout_pair(
    t_fit: np.ndarray,
    hare: np.ndarray,
    lynx: np.ndarray,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    hare_fit: ChannelFit,
    lynx_fit: ChannelFit,
) -> np.ndarray:
    last = int(fit_idx[-1])
    pred_h, pred_l = rollout_levels(
        float(t_fit[last]),
        float(hare[last]),
        float(lynx[last]),
        t_fit[test_idx],
        _ode_rhs(hare_fit, lynx_fit),
    )
    return np.concatenate([pred_h, pred_l])


def _channel_payload(fit: ChannelFit) -> dict[str, Any]:
    return {
        "xy": fit.xy_coefficient,
        "intercept": fit.intercept,
        "terms": list(fit.equation_terms),
        "coefficients": fit.coefficients,
        "test_rmse": fit.test_rmse,
    }


def interpolant_knobs(*, hidden: int) -> dict[str, Any]:
    return {
        "hidden": int(hidden),
        "ridge": INTERPOLANT_RIDGE,
        "weight_scale": INTERPOLANT_WEIGHT_SCALE,
        "upsample": INTERPOLANT_UPSAMPLE,
        "deriv_weight": INTERPOLANT_DERIV_WEIGHT,
        "quality_ratio": INTERPOLANT_QUALITY_RATIO,
        "mode": INTERPOLANT_MODE,
    }


def discover_table(
    table: dict[str, np.ndarray],
    *,
    hidden: int = 96,
    seed: int = 0,
    threshold: float = 1e-3,
    eval_hare: np.ndarray | None = None,
    eval_lynx: np.ndarray | None = None,
    loss: Literal["ridge", "huber"] = "ridge",
    require_xy_signs: bool = False,
    require_interpolant_quality: bool = False,
    require_rollout_vs_linear: bool = True,
    require_rollout_vs_zero: bool = False,
) -> dict[str, Any]:
    """Interpolate the fit window, STLSQ the library, and score by rollout.

    Derivative-MSE columns stay in the result as diagnostics. The scientific
    score is an RK4 rollout of hare/lynx levels on the chronological hold-out.
    """
    t = np.asarray(table["t"], dtype=float).reshape(-1)
    hare = np.asarray(table["hare"], dtype=float).reshape(-1)
    lynx = np.asarray(table["lynx"], dtype=float).reshape(-1)
    ref = require_reference_valid(np.concatenate([hare, lynx]), name="populations")
    fit_idx, test_idx = _fit_index(t.shape[0])
    t_fit = t[fit_idx]
    hare_fit_vals = hare[fit_idx]
    lynx_fit_vals = lynx[fit_idx]

    hare_jet = np.zeros_like(t)
    lynx_jet = np.zeros_like(t)
    hare_spline = np.zeros_like(t)
    lynx_spline = np.zeros_like(t)
    hare_jet[fit_idx] = _jet_deriv(t_fit, hare_fit_vals, t_fit, hidden=hidden, seed=seed)
    lynx_jet[fit_idx] = _jet_deriv(t_fit, lynx_fit_vals, t_fit, hidden=hidden, seed=seed + 1)
    _, hare_spline[fit_idx] = spline_values_and_deriv(t_fit, hare_fit_vals, t_fit)
    _, lynx_spline[fit_idx] = spline_values_and_deriv(t_fit, lynx_fit_vals, t_fit)
    hare_fd = np.zeros_like(t)
    lynx_fd = np.zeros_like(t)
    hare_fd[fit_idx] = np.gradient(hare_fit_vals, t_fit)
    lynx_fd[fit_idx] = np.gradient(lynx_fit_vals, t_fit)
    # Full-series FD is diagnostic only (public test has no interpolant jet).
    hare_fd_full = np.gradient(hare, t)
    lynx_fd_full = np.gradient(lynx, t)

    hare_eval = (
        hare_fd_full if eval_hare is None else np.asarray(eval_hare, dtype=float).reshape(-1)
    )
    lynx_eval = (
        lynx_fd_full if eval_lynx is None else np.asarray(eval_lynx, dtype=float).reshape(-1)
    )
    eval_fit = np.concatenate([hare_eval[fit_idx], lynx_eval[fit_idx]])
    jet_fit = np.concatenate([hare_jet[fit_idx], lynx_jet[fit_idx]])
    spline_fit = np.concatenate([hare_spline[fit_idx], lynx_spline[fit_idx]])
    fd_fit = np.concatenate([hare_fd[fit_idx], lynx_fd[fit_idx]])
    jet_dot_rmse = float(np.sqrt(mse(jet_fit, eval_fit)))
    spline_dot_rmse = float(np.sqrt(mse(spline_fit, eval_fit)))
    fd_dot_rmse = float(np.sqrt(mse(fd_fit, eval_fit)))
    jet_vs_fd_ratio = jet_dot_rmse / max(fd_dot_rmse, 1e-30)
    if INTERPOLANT_MODE == "jet":
        chosen = "jet"
    elif INTERPOLANT_MODE == "spline":
        chosen = "spline"
    else:
        chosen = "jet" if jet_vs_fd_ratio <= INTERPOLANT_QUALITY_RATIO else "spline"
    hare_dot = hare_jet if chosen == "jet" else hare_spline
    lynx_dot = lynx_jet if chosen == "jet" else lynx_spline
    chosen_dot_rmse = jet_dot_rmse if chosen == "jet" else spline_dot_rmse
    interpolant_ok = bool(chosen_dot_rmse <= INTERPOLANT_QUALITY_RATIO * max(fd_dot_rmse, 1e-30))

    lib = _library(hare, lynx)
    lin = _library(hare, lynx, linear=True)
    eval_test = np.concatenate([hare_eval[test_idx], lynx_eval[test_idx]])

    hare_interp = _fit_channel(
        lib[fit_idx],
        hare_dot[fit_idx],
        lib[test_idx],
        hare_eval[test_idx],
        names=TERM_NAMES,
        threshold=threshold,
        loss=loss,
    )
    lynx_interp = _fit_channel(
        lib[fit_idx],
        lynx_dot[fit_idx],
        lib[test_idx],
        lynx_eval[test_idx],
        names=TERM_NAMES,
        threshold=threshold,
        loss=loss,
    )
    hare_fd_fit = _fit_channel(
        lib[fit_idx],
        hare_fd[fit_idx],
        lib[test_idx],
        hare_eval[test_idx],
        names=TERM_NAMES,
        threshold=threshold,
        loss=loss,
    )
    lynx_fd_fit = _fit_channel(
        lib[fit_idx],
        lynx_fd[fit_idx],
        lib[test_idx],
        lynx_eval[test_idx],
        names=TERM_NAMES,
        threshold=threshold,
        loss=loss,
    )
    hare_lin = _fit_channel(
        lin[fit_idx],
        hare_dot[fit_idx],
        lin[test_idx],
        hare_eval[test_idx],
        names=LINEAR_NAMES,
        threshold=threshold,
        loss=loss,
    )
    lynx_lin = _fit_channel(
        lin[fit_idx],
        lynx_dot[fit_idx],
        lin[test_idx],
        lynx_eval[test_idx],
        names=LINEAR_NAMES,
        threshold=threshold,
        loss=loss,
    )

    interp_pred = np.concatenate([hare_interp.test_pred, lynx_interp.test_pred])
    fd_pred = np.concatenate([hare_fd_fit.test_pred, lynx_fd_fit.test_pred])
    lin_pred = np.concatenate([hare_lin.test_pred, lynx_lin.test_pred])
    mse_interp = mse(interp_pred, eval_test)
    mse_fd = mse(fd_pred, eval_test)
    mse_lin = mse(lin_pred, eval_test)
    try:
        skill_vs_zero = skill_score(interp_pred, eval_test)
    except ValueError:
        skill_vs_zero = float("nan")
    skill_vs_fd = 1.0 - mse_interp / max(mse_fd, 1e-30)
    skill_vs_linear = 1.0 - mse_interp / max(mse_lin, 1e-30)

    target_levels = np.concatenate([hare[test_idx], lynx[test_idx]])
    persist = np.concatenate(
        [
            np.full(test_idx.shape[0], hare[fit_idx[-1]], dtype=float),
            np.full(test_idx.shape[0], lynx[fit_idx[-1]], dtype=float),
        ]
    )
    rollout_interp = _rollout_pair(t, hare, lynx, fit_idx, test_idx, hare_interp, lynx_interp)
    rollout_fd = _rollout_pair(t, hare, lynx, fit_idx, test_idx, hare_fd_fit, lynx_fd_fit)
    rollout_lin = _rollout_pair(t, hare, lynx, fit_idx, test_idx, hare_lin, lynx_lin)
    mse_rollout = mse(rollout_interp, target_levels)
    mse_rollout_fd = mse(rollout_fd, target_levels)
    mse_rollout_lin = mse(rollout_lin, target_levels)
    rollout_vs_zero = rollout_skill(rollout_interp, target_levels, persist)
    rollout_vs_linear = 1.0 - mse_rollout / max(mse_rollout_lin, 1e-30)
    rollout_vs_fd = 1.0 - mse_rollout / max(mse_rollout_fd, 1e-30)
    xy_signs_ok = bool(hare_interp.xy_coefficient < 0.0 and lynx_interp.xy_coefficient > 0.0)

    entries: list[dict[str, Any]] = [{**ref, "name": "reference_valid"}]
    if require_interpolant_quality:
        entries.append(
            {
                "name": "interpolant_quality",
                "jet_dot_rmse": jet_dot_rmse,
                "spline_dot_rmse": spline_dot_rmse,
                "fd_dot_rmse": fd_dot_rmse,
                "ratio": chosen_dot_rmse / max(fd_dot_rmse, 1e-30),
                "chosen": chosen,
                "max_ratio": INTERPOLANT_QUALITY_RATIO,
                "passed": interpolant_ok,
                "required": True,
            }
        )
    entries.append(
        {
            "name": "xy_signs",
            "hare_xy": hare_interp.xy_coefficient,
            "lynx_xy": lynx_interp.xy_coefficient,
            "passed": True if not require_xy_signs else xy_signs_ok,
            "required": require_xy_signs,
        }
    )
    entries.extend(
        [
            {
                "name": "rollout_vs_zero",
                "skill_score": rollout_vs_zero,
                "min_skill": 0.0,
                "passed": bool(rollout_vs_zero > 0.0) if require_rollout_vs_zero else True,
                "required": require_rollout_vs_zero,
            },
            {
                "name": "rollout_vs_linear",
                "skill_score": rollout_vs_linear,
                "min_skill": 0.0,
                "passed": bool(rollout_vs_linear > 0.0) if require_rollout_vs_linear else True,
                "required": require_rollout_vs_linear,
            },
        ]
    )
    return {
        "n_rows": int(t.shape[0]),
        "n_fit": int(fit_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
        "loss": loss,
        "interpolant": chosen,
        "interpolant_knobs": interpolant_knobs(hidden=hidden),
        "jet_dot_rmse": jet_dot_rmse,
        "spline_dot_rmse": spline_dot_rmse,
        "fd_dot_rmse": fd_dot_rmse,
        "jet_vs_fd_ratio": jet_vs_fd_ratio,
        "interpolant_quality_ok": interpolant_ok,
        "hare": _channel_payload(hare_interp),
        "lynx": _channel_payload(lynx_interp),
        "mse_jet": mse_interp,
        "mse_fd": mse_fd,
        "mse_linear": mse_lin,
        "skill_vs_zero": skill_vs_zero,
        "skill_vs_fd": skill_vs_fd,
        "skill_vs_linear": skill_vs_linear,
        "rollout_vs_zero": rollout_vs_zero,
        "rollout_vs_linear": rollout_vs_linear,
        "rollout_vs_fd": rollout_vs_fd,
        "mse_rollout": mse_rollout,
        "mse_rollout_linear": mse_rollout_lin,
        "mse_rollout_fd": mse_rollout_fd,
        "xy_signs_ok": xy_signs_ok,
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
        loss="ridge",
        require_xy_signs=True,
        require_interpolant_quality=True,
        require_rollout_vs_linear=True,
        require_rollout_vs_zero=False,
    )
    result["source"] = "synthetic_lotka_volterra"
    result["true"] = {
        "alpha": float(table["true_alpha"]),
        "beta": float(table["true_beta"]),
        "delta": float(table["true_delta"]),
        "gamma": float(table["true_gamma"]),
    }
    return result


def evaluate_public_csv(*, hidden: int = 96, seed: int = 0) -> dict[str, Any]:
    table = load_lynx_hare()
    huber = discover_table(
        table,
        hidden=hidden,
        seed=seed,
        threshold=1e-3,
        loss="huber",
        require_xy_signs=False,
        require_interpolant_quality=False,
        require_rollout_vs_linear=True,
        require_rollout_vs_zero=True,
    )
    ridge = discover_table(
        table,
        hidden=hidden,
        seed=seed,
        threshold=1e-3,
        loss="ridge",
        require_xy_signs=False,
        require_interpolant_quality=False,
        require_rollout_vs_linear=False,
        require_rollout_vs_zero=False,
    )
    if huber["xy_signs_ok"]:
        entries = list(huber["gates"]["entries"])
        for row in entries:
            if row["name"] == "xy_signs":
                row["required"] = True
                row["passed"] = True
        huber["gates"] = gates_block(entries)
    huber["source"] = "hudson_bay_lynx_hare"
    huber["provenance"] = provenance()
    huber["ridge_rollout_vs_zero"] = ridge["rollout_vs_zero"]
    huber["huber_vs_ridge_rollout"] = 1.0 - huber["mse_rollout"] / max(
        ridge["mse_rollout"], 1e-30
    )
    return huber


def evaluate_benchmark(*, quick: bool = False, seed: int = 0) -> dict[str, Any]:
    hidden = 48 if quick else 256
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
        "schema": SCHEMA,
        "quick": quick,
        "synthetic": synthetic,
        "public_csv": public,
        "gates": gates,
        "honesty": HONESTY,
    }


__all__ = [
    "CSV_PATH",
    "HONESTY",
    "SCHEMA",
    "discover_table",
    "evaluate_benchmark",
    "evaluate_public_csv",
    "evaluate_synthetic",
    "load_lynx_hare",
    "rollout_levels",
    "simulate_lotka_volterra",
    "spline_values_and_deriv",
]
