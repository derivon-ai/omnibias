# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Implicit / DAE residual discovery from interpolant jets.

Not a DAE integrator, not index reduction, not an existence proof.
Shared-parameter stacked residuals only. ``yang_mills_claim`` stays false.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from omnibias.symbolic.discovery import SparseEquation, fit_sparse_equation
from omnibias.symbolic.field_discovery import extract_field_jet, fit_neural_field_nd


@dataclass(frozen=True)
class ImplicitSystemResult:
    differential: dict[str, SparseEquation]
    algebraic_rmse: dict[str, float]
    passed: bool
    yang_mills_claim: bool = False
    continuum_claim: bool = False
    notes: dict[str, str] = field(default_factory=dict)


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((target - pred) ** 2)))


class ImplicitSystemDiscoverer:
    """Interpolate channels, then fit ``y' = f(y)`` and algebraic squares."""

    def discover(
        self,
        independent: np.ndarray,
        channels: Mapping[str, np.ndarray],
        *,
        differential: Sequence[str] | None = None,
        algebraic_squares: Sequence[str] = (),
        hidden: int = 48,
    ) -> ImplicitSystemResult:
        t = np.asarray(independent, dtype=float).reshape(-1, 1)
        names = list(channels)
        values = {
            name: np.asarray(channels[name], dtype=float).reshape(-1) for name in names
        }
        n = t.shape[0]
        if any(arr.shape[0] != n for arr in values.values()):
            raise ValueError("channels must share the independent-variable length")
        jets = {}
        for name, series in values.items():
            fitted = fit_neural_field_nd(
                t, series, hidden=hidden, var_names=("t",), activation="tanh"
            )
            jets[name] = extract_field_jet(fitted, t, max_order=1)
        diff_names = list(differential) if differential is not None else names
        design_names = names
        design = np.column_stack([values[name] for name in design_names])
        differential_eqs: dict[str, SparseEquation] = {}
        passed_diff = True
        for name in diff_names:
            lhs = jets[name].partial((1,))
            eq = fit_sparse_equation(
                design, lhs, design_names, alpha=1e-10, threshold=1e-3
            )
            differential_eqs[name] = eq
            if _rmse(lhs, eq.predict(design)) > 0.15:
                passed_diff = False
        algebraic_rmse: dict[str, float] = {}
        passed_alg = True
        if algebraic_squares:
            squares = np.stack([values[name] ** 2 for name in algebraic_squares], axis=1)
            total = squares.sum(axis=1)
            target = np.full(n, float(np.mean(total)))
            rmse = _rmse(target, total)
            algebraic_rmse["+".join(f"{name}^2" for name in algebraic_squares)] = rmse
            if rmse > 0.05:
                passed_alg = False
        return ImplicitSystemResult(
            differential=differential_eqs,
            algebraic_rmse=algebraic_rmse,
            passed=bool(passed_diff and passed_alg),
            notes={"kind": "implicit_residual_discovery"},
        )


__all__ = ["ImplicitSystemDiscoverer", "ImplicitSystemResult"]
