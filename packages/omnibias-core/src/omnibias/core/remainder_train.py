# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Remainder training (theory 09-18).

The loss is ``R_N = f - T_N``, the Taylor remainder after order
``N``, not a rewrite of the 03-10 diagnostic or the 03-13
architecture search. Jets of ``T_N`` use the founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``, feasibility)
does not appear. Do not conflate the two.

Not CCF stretch. ``R_N`` is of the model (or a named analytic
target), not of an unknown PDE solution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.core.refine import residual_indicator

DISCLAIMER = (
    "remainder loss R_N; not spec 03-10, not spec 03-13, and not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "is_03_10": False,
        "is_03_13": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class RemainderTrainConfig:
    jet_order: int = 2
    birth_threshold: float = 1e-3
    use_pade: bool = False


DEFAULT_CONFIG = RemainderTrainConfig()


def taylor_eval(jet: Sequence[float], x: float, x0: float = 0.0) -> float:
    """``T_N(x; x0)`` from ``(f, f', ..., f^{(N)})`` at ``x0``."""
    h = x - x0
    acc = 0.0
    fact = 1.0
    pow_h = 1.0
    for k, deriv in enumerate(jet):
        if k > 0:
            fact *= float(k)
            pow_h *= h
        acc += float(deriv) * pow_h / fact
    return acc


def remainder(value: float, jet: Sequence[float], x: float, x0: float = 0.0) -> float:
    return value - taylor_eval(jet, x, x0)


def exp_jet(order: int, x0: float = 0.0) -> list[float]:
    e = math.exp(x0)
    return [e for _ in range(order + 1)]


def analytic_exp_remainder(x: float, *, order: int = 2, x0: float = 0.0) -> float:
    return remainder(math.exp(x), exp_jet(order, x0), x, x0)


def worked_example() -> dict[str, float]:
    x = 0.2
    r2 = analytic_exp_remainder(x, order=2, x0=0.0)
    t2 = taylor_eval(exp_jet(2), x)
    return {
        "R2": r2,
        "T2": t2,
        "exp": math.exp(x),
        "loss": r2 * r2,
        "R2_err": abs(r2 - (math.exp(x) - (1.0 + x + 0.5 * x * x))),
    }


def pade_11_exp(x: float) -> float:
    """``[1/1]`` Padé of ``exp`` about 0: ``(1 + x/2) / (1 - x/2)``."""
    if x >= 2.0:
        raise ValueError("Padé [1/1] of exp has a pole at x=2")
    return (1.0 + 0.5 * x) / (1.0 - 0.5 * x)


def _sigmoid(z: float) -> float:
    az = abs(z)
    s = 1.0 / (1.0 + math.exp(-az))
    return s if z >= 0.0 else 1.0 - s


def ombu_value(x: float, amps: Sequence[float], biases: Sequence[float], w: float) -> float:
    return sum(float(a) * _sigmoid(w * x + float(b)) for a, b in zip(amps, biases, strict=True))


def ombu_jet0(amps: Sequence[float], biases: Sequence[float], w: float, order: int) -> list[float]:
    """``(u, u', ..., u^{(order)})`` of ``sum a sigmoid(w x + b)`` at ``x=0``."""
    if order < 0 or order > 2:
        raise ValueError("ombu_jet0 implements orders 0..2")
    u = 0.0
    up = 0.0
    upp = 0.0
    for a, b in zip(amps, biases, strict=True):
        s = _sigmoid(float(b))
        u += float(a) * s
        if order >= 1:
            sp = s * (1.0 - s)
            up += float(a) * w * sp
        if order >= 2:
            spp = s * (1.0 - s) * (1.0 - 2.0 * s)
            upp += float(a) * w * w * spp
    jet = [u]
    if order >= 1:
        jet.append(up)
    if order >= 2:
        jet.append(upp)
    return jet


def model_remainder(
    x: float,
    amps: Sequence[float],
    biases: Sequence[float],
    *,
    w: float = 1.0,
    order: int = 2,
) -> float:
    jet = ombu_jet0(amps, biases, w, order)
    return remainder(ombu_value(x, amps, biases, w), jet, x, 0.0)


def _lstsq(columns: list[list[float]], target: Sequence[float]) -> list[float]:
    matrix: NDArray[np.float64] = np.column_stack([np.asarray(col, dtype=np.float64) for col in columns])
    rhs: NDArray[np.float64] = np.asarray(list(target), dtype=np.float64)
    coef, _residuals, _rank, _s = np.linalg.lstsq(matrix, rhs, rcond=None)
    return [float(v) for v in coef.tolist()]


def _grid(n: int = 33) -> list[float]:
    return [-0.3 + 0.6 * i / (n - 1) for i in range(n)]


def _fit_ombu(xs: Sequence[float], target: Sequence[float], biases: Sequence[float], w: float) -> list[float]:
    cols = [[_sigmoid(w * x + b) for x in xs] for b in biases]
    return _lstsq(cols, target)


def remainder_loss(
    values: Sequence[float],
    jet: Sequence[float],
    xs: Sequence[float],
    x0: float = 0.0,
    *,
    config: RemainderTrainConfig | None = None,
) -> dict[str, float | bool | int]:
    """Mean-square ``R_N``. Suggests a 03-13 look if the loss stays large."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.jet_order != len(jet) - 1:
        raise ValueError("jet length must be jet_order + 1")
    rs = [remainder(v, jet, x, x0) for v, x in zip(values, xs, strict=True)]
    if cfg.use_pade:
        rs = [math.exp(x) - pade_11_exp(x) for x in xs]
    loss = sum(r * r for r in rs) / float(len(rs))
    peak_i, peak = residual_indicator(rs)
    birth = loss > cfg.birth_threshold
    return {
        "loss": loss,
        "max_abs": max(abs(r) for r in rs),
        "birth_suggested": birth,
        "indicator_index": peak_i,
        "indicator_peak": peak,
    }


def exp_remainder_skill(*, seeds: int = 5, width: int = 12) -> dict[str, object]:
    """G2: remainder-train vs value-only MSE on the model's ``R_2``."""
    xs = _grid()
    t2 = [taylor_eval(exp_jet(2), x) for x in xs]
    exps = [math.exp(x) for x in xs]
    rem_max: list[float] = []
    mse_max: list[float] = []
    for seed in range(seeds):
        biases = [-0.3 + 0.6 * i / max(width - 1, 1) + 0.01 * float(seed) for i in range(width)]
        w = 1.0
        a_rem = _fit_ombu(xs, t2, biases, w)
        a_mse = _fit_ombu(xs, exps, biases, w)
        rem_max.append(max(abs(model_remainder(x, a_rem, biases, w=w)) for x in xs))
        mse_max.append(max(abs(model_remainder(x, a_mse, biases, w=w)) for x in xs))
    r_med = sorted(rem_max)[len(rem_max) // 2]
    m_med = sorted(mse_max)[len(mse_max) // 2]
    return {
        "remainder_max": rem_max,
        "value_max": mse_max,
        "remainder_median": r_med,
        "value_median": m_med,
        "beats_value": r_med < m_med,
        "below_1e4": r_med < 1e-4,
        "is_03_10": False,
        "is_03_13": False,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "RemainderTrainConfig",
    "analytic_exp_remainder",
    "exp_jet",
    "exp_remainder_skill",
    "honesty_payload",
    "model_remainder",
    "ombu_jet0",
    "ombu_value",
    "pade_11_exp",
    "remainder",
    "remainder_loss",
    "taylor_eval",
    "worked_example",
]
