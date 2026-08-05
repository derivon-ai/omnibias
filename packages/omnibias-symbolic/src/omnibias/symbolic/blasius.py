# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Blasius boundary-layer solve and jet-identity discovery.

The Blasius equation is a classic nonlinear boundary-value problem:

    f''' + 0.5*f*f'' = 0
    f(0) = 0, f'(0) = 0, f'(inf) = 1

It has no elementary closed-form solution. This module solves it numerically by
shooting for ``f''(0)`` and then asks the weak-prior jet compressor to recover
the governing identity from the numerical solution columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omnibias.symbolic.discovery import (
    JetBundle,
    extract_neural_jets,
    fit_neural_field_1d,
    fit_sparse_equation,
    rmse,
)
from omnibias.symbolic.expressions import (
    RationalExpression,
    fit_rational_expression,
    validate_rational_expression,
)


@dataclass(frozen=True)
class BlasiusSolution:
    eta: np.ndarray
    f: np.ndarray
    fp: np.ndarray
    fpp: np.ndarray
    fppp: np.ndarray
    fpp0: float
    eta_max: float

    def jet_bundle(self) -> JetBundle:
        return JetBundle(
            x=self.eta,
            jets=np.stack([self.f, self.fp, self.fpp, self.fppp], axis=1),
        )


def solve_blasius(
    *,
    eta_max: float = 8.0,
    n_steps: int = 2000,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> BlasiusSolution:
    """Solve the Blasius BVP by bisection shooting."""

    low = 0.2
    high = 0.5
    low_mismatch = _shoot_mismatch(low, eta_max=eta_max, n_steps=n_steps)
    high_mismatch = _shoot_mismatch(high, eta_max=eta_max, n_steps=n_steps)
    if low_mismatch * high_mismatch > 0.0:
        raise RuntimeError("failed to bracket Blasius shooting parameter")

    mid = 0.5 * (low + high)
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mismatch = _shoot_mismatch(mid, eta_max=eta_max, n_steps=n_steps)
        if abs(mismatch) < tol:
            break
        if low_mismatch * mismatch <= 0.0:
            high = mid
            high_mismatch = mismatch
        else:
            low = mid
            low_mismatch = mismatch

    eta, state = _integrate_blasius(mid, eta_max=eta_max, n_steps=n_steps)
    f = state[:, 0]
    fp = state[:, 1]
    fpp = state[:, 2]
    fppp = -0.5 * f * fpp
    return BlasiusSolution(
        eta=eta,
        f=f,
        fp=fp,
        fpp=fpp,
        fppp=fppp,
        fpp0=float(mid),
        eta_max=eta_max,
    )


def discover_blasius_identity(solution: BlasiusSolution | None = None) -> dict[str, object]:
    """Recover ``f''' = -0.5*f*f''`` from numerical Blasius jets."""

    if solution is None:
        solution = solve_blasius()
    bundle = solution.jet_bundle()
    design, names = _blasius_relation_library(bundle)
    target = bundle.jets[:, 3]

    indices = np.arange(bundle.x.shape[0])
    train = indices[indices % 3 == 0]
    val = indices[indices % 3 == 1]
    test = indices[indices % 3 == 2]

    best: tuple[float, float, float, object] | None = None
    for alpha in (1e-12, 1e-10, 1e-8, 1e-6):
        for threshold in (1e-8, 1e-6, 1e-4, 1e-3):
            equation = fit_sparse_equation(
                design[train],
                target[train],
                names,
                alpha=alpha,
                threshold=threshold,
            )
            pred_val = equation.predict(design[val])
            score = rmse(target[val], pred_val) + 2e-3 * len(equation.active_terms())
            if best is None or score < best[0]:
                best = (score, alpha, threshold, equation)
    if best is None:
        raise RuntimeError(
            "Blasius sparse discovery produced no candidates; check alphas / thresholds"
        )
    _, alpha, threshold, equation = best

    # Refit on train + validation using the chosen sparsity settings.
    fit = np.concatenate([train, val])
    equation = fit_sparse_equation(
        design[fit],
        target[fit],
        names,
        alpha=alpha,
        threshold=threshold,
    )
    pred = equation.predict(design[test])
    residual = target[test] - pred
    return {
        "hidden_law": "d3f = -0.5*f*d2f",
        "equation": equation.formula(lhs="d3f"),
        "selected_terms": equation.active_terms(),
        "metrics": {
            "test_rmse": rmse(target[test], pred),
            "max_abs_residual": float(np.max(np.abs(residual))),
            "boundary_fp_eta_max_error": float(abs(solution.fp[-1] - 1.0)),
        },
        "shooting": {
            "fpp0": solution.fpp0,
            "eta_max": solution.eta_max,
        },
    }


def discover_blasius_from_neural_surrogate(
    solution: BlasiusSolution | None = None,
    *,
    hidden: int = 512,
    ridge: float = 1e-8,
    seed: int = 1,
) -> dict[str, object]:
    """Fit only ``f(eta)``, extract neural jets, then recover the Blasius residual.

    This is the less-circular Blasius validation path. The neural field sees
    only solution values ``f``. Its derivatives are then produced by omnibias
    closed-form activation fastpaths, and the sparse compressor fits
    ``d3f = c*f*d2f`` from those neural jets.
    """

    if solution is None:
        solution = solve_blasius()
    field = fit_neural_field_1d(
        solution.eta,
        solution.f,
        hidden=hidden,
        ridge=ridge,
        activation="tanh",
        seed=seed,
    )
    bundle = extract_neural_jets(field, solution.eta, max_order=3)
    design = (bundle.jets[:, 0] * bundle.jets[:, 2])[:, None]
    names = ["f*d2f"]
    target = bundle.jets[:, 3]

    indices = np.arange(solution.eta.shape[0])
    train = indices[indices % 3 == 0]
    test = indices[indices % 3 == 2]
    equation = fit_sparse_equation(
        design[train],
        target[train],
        names,
        alpha=1e-10,
        threshold=0.0,
    )
    pred = equation.predict(design[test])
    return {
        "mode": "fit f(eta) only -> closed-form neural jets -> sparse residual",
        "equation": equation.formula(lhs="d3f"),
        "selected_terms": equation.active_terms(),
        "field_train_rmse": field.train_rmse,
        "metrics": {
            "neural_jet_test_rmse": rmse(target[test], pred),
            "true_d3f_rmse": rmse(solution.fppp[test], pred),
            "neural_d3f_vs_true_rmse": rmse(solution.fppp[test], target[test]),
        },
        "settings": {
            "hidden": hidden,
            "ridge": ridge,
            "seed": seed,
        },
    }


def discover_blasius_explicit_expression(
    solution: BlasiusSolution | None = None,
    *,
    numerator_degree: int = 10,
    denominator_degree: int = 6,
) -> dict[str, object]:
    """Fit an explicit rational surrogate ``f(eta) ~= P(t)/Q(t)``.

    This is not an elementary closed-form solution. It is a compact explicit
    expression with analytic derivatives, suitable for fast repeated evaluation
    and residual checks.
    """

    if solution is None:
        solution = solve_blasius()
    indices = np.arange(solution.eta.shape[0])
    train = indices[indices % 2 == 0]
    test = indices[indices % 2 == 1]
    expr = fit_rational_expression(
        solution.eta[train],
        solution.f[train],
        numerator_degree=numerator_degree,
        denominator_degree=denominator_degree,
        ridge=1e-12,
        variable="eta",
        name="BlasiusR",
    )
    value_metrics = validate_rational_expression(expr, solution.eta[test], solution.f[test])
    residual = expr.residual_blasius(solution.eta[test])
    f0 = float(expr.evaluate(np.asarray([0.0]))[0])
    fp0 = float(expr.evaluate(np.asarray([0.0]), derivative_order=1)[0])
    fp_end = float(expr.evaluate(np.asarray([solution.eta_max]), derivative_order=1)[0])
    fpp_end = float(expr.evaluate(np.asarray([solution.eta_max]), derivative_order=2)[0])
    return {
        "kind": "rational_pade_surrogate",
        "formula": expr.formula(digits=5),
        "numerator_degree": numerator_degree,
        "denominator_degree": denominator_degree,
        "metrics": {
            **value_metrics,
            "residual_rmse": rmse(np.zeros_like(residual), residual),
            "residual_max_abs": float(np.max(np.abs(residual))),
            "f0_abs": abs(f0),
            "fp0_abs": abs(fp0),
            "fp_eta_max_error": abs(fp_end - 1.0),
            "fpp_eta_max_abs": abs(fpp_end),
        },
        "coefficients": {
            "numerator": [float(value) for value in expr.numerator],
            "denominator": [float(value) for value in expr.denominator],
            "x_shift": expr.x_shift,
            "x_scale": expr.x_scale,
        },
    }


def discover_blasius_taylor_pade_expression(
    solution: BlasiusSolution | None = None,
    *,
    series_order: int = 44,
    numerator_degree: int = 24,
    denominator_degree: int = 16,
) -> dict[str, object]:
    """Construct a Taylor/Pade expression from the Blasius ODE recurrence.

    This path is more analytic than fitting a rational expression to sampled
    values. It uses the recurrence induced by ``f''' = -0.5*f*f''`` and chooses
    the free parameter ``a=f''(0)`` so the Pade surrogate satisfies
    ``f'(eta_max) ~= 1``.
    """

    if solution is None:
        solution = solve_blasius()
    a = _shoot_taylor_pade_parameter(
        eta_max=solution.eta_max,
        series_order=series_order,
        numerator_degree=numerator_degree,
        denominator_degree=denominator_degree,
    )
    expr = _taylor_pade_expression(
        a,
        eta_max=solution.eta_max,
        series_order=series_order,
        numerator_degree=numerator_degree,
        denominator_degree=denominator_degree,
    )
    residual = expr.residual_blasius(solution.eta)
    pred = expr.evaluate(solution.eta)
    fp_end = float(expr.evaluate(np.asarray([solution.eta_max]), derivative_order=1)[0])
    return {
        "kind": "ode_recurrence_taylor_pade",
        "formula": expr.formula(digits=5),
        "series_order": series_order,
        "numerator_degree": numerator_degree,
        "denominator_degree": denominator_degree,
        "shooting": {
            "fpp0": a,
            "fpp0_reference": solution.fpp0,
            "fpp0_abs_error": abs(a - solution.fpp0),
            "fp_eta_max": fp_end,
        },
        "metrics": {
            "value_rmse": rmse(solution.f, pred),
            "value_max_abs": float(np.max(np.abs(pred - solution.f))),
            "residual_rmse": rmse(np.zeros_like(residual), residual),
            "residual_max_abs": float(np.max(np.abs(residual))),
            "f0_abs": abs(float(expr.evaluate(np.asarray([0.0]))[0])),
            "fp0_abs": abs(float(expr.evaluate(np.asarray([0.0]), derivative_order=1)[0])),
            "fp_eta_max_error": abs(fp_end - 1.0),
        },
        "coefficients": {
            "numerator": [float(value) for value in expr.numerator],
            "denominator": [float(value) for value in expr.denominator],
            "x_shift": expr.x_shift,
            "x_scale": expr.x_scale,
        },
    }


def evaluate_blasius(out_dir: Path | None = None) -> dict[str, object]:
    solution = solve_blasius()
    result = discover_blasius_identity(solution)
    neural_result = discover_blasius_from_neural_surrogate(solution)
    explicit = discover_blasius_explicit_expression(solution)
    taylor_pade = discover_blasius_taylor_pade_expression(solution)
    payload = {
        "problem": "Blasius boundary-layer equation",
        "boundary_conditions": "f(0)=0, f'(0)=0, f'(inf)=1",
        "solution": {
            "fpp0": solution.fpp0,
            "fp_eta_max": float(solution.fp[-1]),
            "eta_max": solution.eta_max,
            "n_points": int(solution.eta.size),
        },
        "discovery": result,
        "neural_surrogate_discovery": neural_result,
        "explicit_expression": explicit,
        "taylor_pade_expression": taylor_pade,
    }
    if out_dir is not None:
        write_blasius_artifacts(payload, out_dir)
    return payload


def write_blasius_artifacts(payload: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blasius_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    discovery = payload["discovery"]
    neural = payload["neural_surrogate_discovery"]
    explicit = payload["explicit_expression"]
    taylor_pade = payload["taylor_pade_expression"]
    solution = payload["solution"]
    if not isinstance(discovery, dict):
        raise TypeError(f"payload['discovery'] must be a dict, got {type(discovery).__name__}")
    if not isinstance(neural, dict):
        raise TypeError(
            f"payload['neural_surrogate_discovery'] must be a dict, got {type(neural).__name__}"
        )
    if not isinstance(explicit, dict):
        raise TypeError(
            f"payload['explicit_expression'] must be a dict, got {type(explicit).__name__}"
        )
    if not isinstance(taylor_pade, dict):
        raise TypeError(
            f"payload['taylor_pade_expression'] must be a dict, got {type(taylor_pade).__name__}"
        )
    if not isinstance(solution, dict):
        raise TypeError(f"payload['solution'] must be a dict, got {type(solution).__name__}")
    lines = [
        "# Blasius Boundary-Layer Discovery",
        "",
        "Numerical BVP:",
        "",
        "```text",
        "f''' + 0.5*f*f'' = 0",
        "f(0)=0, f'(0)=0, f'(inf)=1",
        "```",
        "",
        f"Shooting result: `f''(0) = {solution['fpp0']:.12f}`",
        f"Boundary error at eta_max: `{abs(solution['fp_eta_max'] - 1.0):.3e}`",
        "",
        "Recovered identity:",
        "",
        f"```text\n{discovery['equation']}\n```",
        "",
        f"Test RMSE: `{discovery['metrics']['test_rmse']:.3e}`",
        "",
        "Blind neural-surrogate recovery:",
        "",
        f"```text\n{neural['equation']}\n```",
        "",
        f"Field train RMSE: `{neural['field_train_rmse']:.3e}`",
        f"Neural-jet test RMSE: `{neural['metrics']['neural_jet_test_rmse']:.3e}`",
        f"RMSE versus numerical `f'''`: `{neural['metrics']['true_d3f_rmse']:.3e}`",
        "",
        "Explicit rational expression:",
        "",
        f"```text\n{explicit['formula']}\n```",
        "",
        f"Value RMSE: `{explicit['metrics']['value_rmse']:.3e}`",
        f"Residual RMSE: `{explicit['metrics']['residual_rmse']:.3e}`",
        "",
        "ODE-derived Taylor/Pade expression:",
        "",
        f"```text\n{taylor_pade['formula']}\n```",
        "",
        f"Taylor/Pade `f''(0)`: `{taylor_pade['shooting']['fpp0']:.12f}`",
        f"Value RMSE: `{taylor_pade['metrics']['value_rmse']:.3e}`",
        f"Residual RMSE: `{taylor_pade['metrics']['residual_rmse']:.3e}`",
    ]
    (out_dir / "blasius_report.md").write_text("\n".join(lines) + "\n")


def _shoot_mismatch(fpp0: float, *, eta_max: float, n_steps: int) -> float:
    _, state = _integrate_blasius(fpp0, eta_max=eta_max, n_steps=n_steps)
    return float(state[-1, 1] - 1.0)


def _integrate_blasius(
    fpp0: float,
    *,
    eta_max: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    eta = np.linspace(0.0, eta_max, n_steps + 1)
    h = float(eta[1] - eta[0])
    state = np.zeros((eta.size, 3), dtype=float)
    state[0] = np.asarray([0.0, 0.0, fpp0], dtype=float)
    for idx in range(n_steps):
        y = state[idx]
        k1 = _blasius_rhs(y)
        k2 = _blasius_rhs(y + 0.5 * h * k1)
        k3 = _blasius_rhs(y + 0.5 * h * k2)
        k4 = _blasius_rhs(y + h * k3)
        state[idx + 1] = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return eta, state


def _blasius_rhs(state: np.ndarray) -> np.ndarray:
    f, fp, fpp = state
    return np.asarray([fp, fpp, -0.5 * f * fpp], dtype=float)


def _blasius_taylor_coefficients(fpp0: float, order: int) -> np.ndarray:
    coeffs = np.zeros(order + 1, dtype=float)
    coeffs[2] = 0.5 * fpp0
    for power in range(order - 2):
        product = 0.0
        for left in range(power + 1):
            right = power - left
            product += coeffs[left] * (right + 2) * (right + 1) * coeffs[right + 2]
        coeffs[power + 3] = -0.5 * product / ((power + 3) * (power + 2) * (power + 1))
    return coeffs


def _pade_from_taylor(series: np.ndarray, numerator_degree: int, denominator_degree: int) -> tuple[np.ndarray, np.ndarray]:
    if numerator_degree + denominator_degree >= series.size:
        raise ValueError("series must contain at least numerator_degree + denominator_degree + 1 terms")
    if denominator_degree == 0:
        return series[: numerator_degree + 1], np.empty(0)

    rows = []
    rhs = []
    for power in range(numerator_degree + 1, numerator_degree + denominator_degree + 1):
        rows.append([series[power - j] for j in range(1, denominator_degree + 1)])
        rhs.append(-series[power])
    matrix = np.asarray(rows, dtype=float)
    target = np.asarray(rhs, dtype=float)
    denominator_tail = np.linalg.lstsq(matrix, target, rcond=None)[0]
    denominator = np.concatenate([np.ones(1), denominator_tail])
    numerator = np.zeros(numerator_degree + 1, dtype=float)
    for power in range(numerator_degree + 1):
        numerator[power] = sum(
            denominator[j] * series[power - j]
            for j in range(min(power, denominator_degree) + 1)
        )
    return numerator, denominator_tail


def _taylor_pade_expression(
    fpp0: float,
    *,
    eta_max: float,
    series_order: int,
    numerator_degree: int,
    denominator_degree: int,
) -> RationalExpression:
    eta_series = _blasius_taylor_coefficients(fpp0, series_order)
    scale_powers = eta_max ** np.arange(series_order + 1)
    t_series = eta_series * scale_powers
    numerator, denominator_tail = _pade_from_taylor(t_series, numerator_degree, denominator_degree)
    return RationalExpression(
        numerator=numerator,
        denominator_tail=denominator_tail,
        x_shift=0.0,
        x_scale=eta_max,
        variable="eta",
        name="TaylorPadeBlasius",
    )


def _shoot_taylor_pade_parameter(
    *,
    eta_max: float,
    series_order: int,
    numerator_degree: int,
    denominator_degree: int,
    low: float = 0.1,
    high: float = 0.6,
) -> float:
    def mismatch(value: float) -> float:
        expr = _taylor_pade_expression(
            value,
            eta_max=eta_max,
            series_order=series_order,
            numerator_degree=numerator_degree,
            denominator_degree=denominator_degree,
        )
        return float(expr.evaluate(np.asarray([eta_max]), derivative_order=1)[0] - 1.0)

    low_mismatch = mismatch(low)
    high_mismatch = mismatch(high)
    if low_mismatch * high_mismatch > 0.0:
        raise RuntimeError("failed to bracket Taylor/Pade Blasius shooting parameter")

    for _ in range(80):
        mid = 0.5 * (low + high)
        mid_mismatch = mismatch(mid)
        if abs(mid_mismatch) < 1e-10:
            return mid
        if low_mismatch * mid_mismatch <= 0.0:
            high = mid
            high_mismatch = mid_mismatch
        else:
            low = mid
            low_mismatch = mid_mismatch
    return 0.5 * (low + high)


def _blasius_relation_library(bundle: JetBundle) -> tuple[np.ndarray, list[str]]:
    f = bundle.jets[:, 0]
    fp = bundle.jets[:, 1]
    fpp = bundle.jets[:, 2]
    eta = bundle.x
    cols = [
        f,
        fp,
        fpp,
        eta,
        f * f,
        f * fp,
        f * fpp,
        fp * fp,
        fp * fpp,
        fpp * fpp,
        eta * f,
        eta * fp,
        eta * fpp,
    ]
    names = [
        "f",
        "df",
        "d2f",
        "eta",
        "f^2",
        "f*df",
        "f*d2f",
        "df^2",
        "df*d2f",
        "d2f^2",
        "eta*f",
        "eta*df",
        "eta*d2f",
    ]
    return np.stack(cols, axis=1), names


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Blasius discovery demo.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional directory for evaluation artifacts (default: write nothing)",
    )
    args = parser.parse_args()
    payload = evaluate_blasius(args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
