# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Riccati flow net (theory 09-10).

The layer is the time-``t`` flow of the founding Riccati ODE
``ds/dt = s(1-s)`` (logistic) or ``d tanh/dt = 1-tanh^2``. Depth is
integration time, not a DEQ fixed point and not a CNF density ODE.

This forward is not founding bias collapse (no ``delta -> 0`` pack).
Bias collapse may still supply jets of ``s(t)`` in ``s0``. Temperature
collapse (``beta -> inf``, feasibility) does not appear. Do not
conflate the two.

Not ImageNet. Not CCF stretch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.core.ftc import sigmoid

DISCLAIMER = (
    "Riccati time flow; not a DEQ, not a CNF, and not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "claimed_deq": False,
        "claimed_cnf": False,
        "imagenet_claim": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class RiccatiFlowConfig:
    family: str = "logistic"
    t: float = 1.0


DEFAULT_CONFIG = RiccatiFlowConfig()


def logistic_flow(s0: float, t: float) -> float:
    """Closed-form ``s(t) = s0 / (s0 + (1-s0) e^{-t})``."""
    if not 0.0 < s0 < 1.0:
        raise ValueError("logistic flow requires s0 in (0, 1)")
    denom = s0 + (1.0 - s0) * math.exp(-t)
    return s0 / denom


def logistic_ds_ds0(s0: float, t: float) -> float:
    """``∂s/∂s0 = e^{-t} / (s0 + (1-s0) e^{-t})^2``."""
    if not 0.0 < s0 < 1.0:
        raise ValueError("logistic flow requires s0 in (0, 1)")
    denom = s0 + (1.0 - s0) * math.exp(-t)
    return math.exp(-t) / (denom * denom)


def tanh_flow(s0: float, t: float) -> float:
    """Closed-form ``tanh(artanh(s0) + t)``."""
    if not -1.0 < s0 < 1.0:
        raise ValueError("tanh flow requires s0 in (-1, 1)")
    return math.tanh(math.atanh(s0) + t)


def riccati_flow(s0: float, *, config: RiccatiFlowConfig | None = None) -> float:
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.family == "logistic":
        return logistic_flow(s0, cfg.t)
    if cfg.family == "tanh":
        return tanh_flow(s0, cfg.t)
    raise ValueError(f"unknown Riccati family {cfg.family!r}")


def worked_example() -> dict[str, float]:
    s0 = 0.25
    t = 1.0
    s = logistic_flow(s0, t)
    exact = s0 / (s0 + (1.0 - s0) * math.exp(-t))
    ds = logistic_ds_ds0(s0, t)
    return {
        "s": s,
        "exact": exact,
        "err": abs(s - exact),
        "ds_ds0": ds,
        "ds_ds0_err": abs(ds - math.exp(-t) / (s0 + (1.0 - s0) * math.exp(-t)) ** 2),
    }


def _lstsq(columns: list[list[float]], target: Sequence[float]) -> list[float]:
    matrix: NDArray[np.float64] = np.column_stack([np.asarray(col, dtype=np.float64) for col in columns])
    rhs: NDArray[np.float64] = np.asarray(list(target), dtype=np.float64)
    coef, _residuals, _rank, _s = np.linalg.lstsq(matrix, rhs, rcond=None)
    return [float(v) for v in coef.tolist()]


def _rmse(pred: Sequence[float], target: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, target, strict=True)) / float(len(pred)))


def _fit_t(s0s: Sequence[float], targets: Sequence[float]) -> float:
    """Recover constant ``T`` from ``s = logistic_flow(s0, T)``."""
    logs: list[float] = []
    for s0, y in zip(s0s, targets, strict=True):
        ratio = (s0 * (1.0 - y)) / (y * (1.0 - s0))
        if ratio <= 0.0:
            continue
        logs.append(-math.log(ratio))
    if not logs:
        raise ValueError("could not recover T")
    return sum(logs) / float(len(logs))


def flow_skill(*, seeds: int = 5, n: int = 32, width: int = 8) -> dict[str, object]:
    """G2: fit ``T`` on logistic samples vs a width-8 sigmoid MLP."""
    s0s = [0.1 + 0.3 * i / (n - 1) for i in range(n)]
    t_true = 1.0
    target = [logistic_flow(s0, t_true) for s0 in s0s]
    flow_err: list[float] = []
    mlp_err: list[float] = []
    for seed in range(seeds):
        t_hat = _fit_t(s0s, target)
        pred_f = [logistic_flow(s0, t_hat) for s0 in s0s]
        flow_err.append(_rmse(pred_f, target))
        biases = [0.1 + 0.3 * i / max(width - 1, 1) + 0.02 * float(seed) for i in range(width)]
        cols = [[sigmoid(s0 + b) for s0 in s0s] for b in biases]
        amps = _lstsq(cols, target)
        pred_m = [
            sum(a * sigmoid(s0 + b) for a, b in zip(amps, biases, strict=True)) for s0 in s0s
        ]
        mlp_err.append(_rmse(pred_m, target))
    f_med = sorted(flow_err)[len(flow_err) // 2]
    m_med = sorted(mlp_err)[len(mlp_err) // 2]
    return {
        "flow_rmse": flow_err,
        "mlp_rmse": mlp_err,
        "flow_median": f_med,
        "mlp_median": m_med,
        "below_1e6": f_med < 1e-6,
        "beats_mlp": f_med < m_med,
        "claimed_deq": False,
        "claimed_cnf": False,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "RiccatiFlowConfig",
    "flow_skill",
    "honesty_payload",
    "logistic_ds_ds0",
    "logistic_flow",
    "riccati_flow",
    "tanh_flow",
    "worked_example",
]
