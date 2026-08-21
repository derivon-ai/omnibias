# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Pack-MoE: slab-mass router over OMBU-pack experts (theory 09-07).

Expert ``e`` is a pack (order / window / position). The gate is
``g_e = I_e / sum I_j`` with ``I_e`` an ``integral`` (or ``band``)
cell — the window knob. Softmax is not the default router. The
founding bias collapse (``delta -> 0``) may appear only in an
expert collapse head. Temperature collapse (``beta -> inf``,
feasibility) is recorded when ``beta != 1``. Do not conflate the
two.

Not a generic ImageNet MoE. Not a 05-02 LightGBM reversal. Not CCF
stretch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.core.ftc import softplus


def _sigmoid(z: float) -> float:
    """Complement-consistent sigmoid so ``σ(-z) = 1 - σ(z)`` bit-exactly."""
    az = abs(z)
    s = 1.0 / (1.0 + math.exp(-az))
    return s if z >= 0.0 else 1.0 - s

DISCLAIMER = (
    "slab-mass Pack-MoE; not a softmax router, not ImageNet, not a 05-02 reversal, not CCF stretch"
)


def honesty_payload(*, beta: float = 1.0) -> dict[str, bool]:
    return {
        "router_is_softmax": False,
        "temperature_collapse_used": beta != 1.0,
        "imagenet_claim": False,
        "lightgbm_reversal": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class PackMoEConfig:
    n_experts: int = 4
    router: str = "integral"
    beta: float = 1.0
    allow_softmax: bool = False
    eps: float = 1e-12


DEFAULT_CONFIG = PackMoEConfig()


@dataclass(frozen=True)
class ExpertWindow:
    b_lo: float
    b_hi: float
    w: float = 1.0


def _require_router(config: PackMoEConfig) -> None:
    if config.router == "softmax" and not config.allow_softmax:
        raise ValueError("softmax router is forbidden as the default; pass allow_softmax=True")
    if config.router not in {"integral", "band", "softmax"}:
        raise ValueError(f"router must be 'integral', 'band', or 'softmax', got {config.router!r}")


def slab_mass(x: float, window: ExpertWindow, *, router: str) -> float:
    z_lo = window.w * x + window.b_lo
    z_hi = window.w * x + window.b_hi
    if router == "integral":
        return softplus(z_hi) - softplus(z_lo)
    if router == "band":
        return _sigmoid(z_hi) - _sigmoid(z_lo)
    raise ValueError(f"slab_mass does not implement {router!r}")


def pack_moe_gates(
    x: float,
    windows: Sequence[ExpertWindow],
    *,
    config: PackMoEConfig | None = None,
) -> list[float]:
    cfg = DEFAULT_CONFIG if config is None else config
    _require_router(cfg)
    if cfg.router == "softmax":
        logits = [window.w * x + 0.5 * (window.b_lo + window.b_hi) for window in windows]
        m = max(logits)
        exps = [math.exp(cfg.beta * (v - m)) for v in logits]
        denom = sum(exps) + cfg.eps
        return [e / denom for e in exps]
    masses = [slab_mass(x, window, router=cfg.router) for window in windows]
    if cfg.beta != 1.0:
        masses = [max(m, 0.0) ** cfg.beta for m in masses]
    total = sum(masses)
    denom = total if total > cfg.eps else total + cfg.eps
    return [m / denom for m in masses]


def pack_moe_forward(
    x: float,
    experts: Sequence[float],
    windows: Sequence[ExpertWindow],
    *,
    config: PackMoEConfig | None = None,
) -> float:
    gates = pack_moe_gates(x, windows, config=config)
    if len(gates) != len(experts):
        raise ValueError("experts and windows must have equal length")
    return sum(g * f for g, f in zip(gates, experts, strict=True))


def worked_example() -> dict[str, float]:
    # Band masses σ(0)-σ(-0.2) and σ(0.2)-σ(0) are equal by oddness of
    # sigmoid about 0. Equal-width *integral* windows are not (softplus
    # is not odd); the spec's 0.5 split is this band identity.
    windows = (ExpertWindow(-0.2, 0.0), ExpertWindow(0.0, 0.2))
    cfg = PackMoEConfig(router="band")
    gates = pack_moe_gates(0.0, windows, config=cfg)
    y = pack_moe_forward(0.0, (1.0, 3.0), windows, config=cfg)
    return {
        "g_a": gates[0],
        "g_b": gates[1],
        "y": y,
        "g_a_err": abs(gates[0] - 0.5),
        "g_b_err": abs(gates[1] - 0.5),
        "y_err": abs(y - 2.0),
    }


def _lstsq(columns: list[list[float]], target: Sequence[float]) -> list[float]:
    matrix: NDArray[np.float64] = np.column_stack([np.asarray(col, dtype=np.float64) for col in columns])
    rhs: NDArray[np.float64] = np.asarray(list(target), dtype=np.float64)
    coef, _residuals, _rank, _s = np.linalg.lstsq(matrix, rhs, rcond=None)
    return [float(v) for v in coef.tolist()]


def _grid(n: int = 97) -> list[float]:
    return [-math.pi + 2.0 * math.pi * i / (n - 1) for i in range(n)]


def _target(xs: Sequence[float]) -> list[float]:
    return [math.sin(x) + 0.3 * math.sin(4.0 * x) for x in xs]


def _biases(n: int, seed: int, lo: float, hi: float) -> list[float]:
    """Bias centers on ``[lo, hi]``. A single pack uses one interval."""
    span = hi - lo
    return [lo + span * i / max(n - 1, 1) + 0.03 * float(seed) for i in range(n)]


def _identity_col(xs: Sequence[float], w: float, bias: float) -> list[float]:
    return [_sigmoid(w * x + bias) for x in xs]


def _fit_rmse(columns: list[list[float]], target: Sequence[float]) -> float:
    coef = _lstsq(columns, target)
    pred = [sum(c * col[i] for c, col in zip(coef, columns, strict=True)) for i in range(len(target))]
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, target, strict=True)) / float(len(target)))


def two_tone_skill(*, seeds: int = 5, width: int = 40) -> dict[str, object]:
    """G2: two-scale Pack-MoE vs a single-scale pack of the same width."""
    xs = _grid()
    target = _target(xs)
    windows = (ExpertWindow(-8.0, 8.0, 1.0), ExpertWindow(-8.0, 8.0, 1.0))
    cfg = PackMoEConfig(n_experts=2, router="integral", beta=1.0)
    half = width // 2
    moe_errs: list[float] = []
    pack_errs: list[float] = []
    zero_errs: list[float] = []
    for seed in range(seeds):
        gates = [pack_moe_gates(x, windows, config=cfg) for x in xs]
        moe_cols: list[list[float]] = []
        for bias in _biases(half, seed, -math.pi, 0.0):
            raw = _identity_col(xs, 1.0, bias)
            moe_cols.append([g[0] * v for g, v in zip(gates, raw, strict=True)])
        for bias in _biases(half, seed + 17, 0.0, math.pi):
            raw = _identity_col(xs, 4.0, bias)
            moe_cols.append([g[1] * v for g, v in zip(gates, raw, strict=True)])
        pack_cols = [_identity_col(xs, 1.0, b) for b in _biases(width, seed, -0.4, 0.4)]
        moe_errs.append(_fit_rmse(moe_cols, target))
        pack_errs.append(_fit_rmse(pack_cols, target))
        energy = math.sqrt(sum(t * t for t in target) / float(len(target)))
        zero_errs.append(energy)
    moe_med = sorted(moe_errs)[len(moe_errs) // 2]
    pack_med = sorted(pack_errs)[len(pack_errs) // 2]
    zero_med = sorted(zero_errs)[len(zero_errs) // 2]
    return {
        "moe_errors": moe_errs,
        "pack_errors": pack_errs,
        "moe_median": moe_med,
        "pack_median": pack_med,
        "skill": 1.0 - moe_med / zero_med if zero_med else 0.0,
        "beats_pack": moe_med < pack_med,
        "below_1e2": moe_med < 1e-2,
        "temperature_collapse_used": False,
        "router_is_softmax": False,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "ExpertWindow",
    "PackMoEConfig",
    "honesty_payload",
    "pack_moe_forward",
    "pack_moe_gates",
    "slab_mass",
    "two_tone_skill",
    "worked_example",
]
