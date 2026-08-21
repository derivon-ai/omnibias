# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Integral-kernel operator (theory 09-14).

A volumetric 1-D kernel whose cell **is** the OMBU ``integral`` role
``S(z + b_hi) - S(z + b_lo)`` with ``S' = sigma``. Not a surface
BEM-Net (02-06) and not a Fourier multiplier.

The window is the knob. founding bias collapse (``delta -> 0``) of
that window recovers a ``sigma`` kernel and is not the default.
Temperature collapse (``beta -> inf``, feasibility) does not appear.
do not conflate the two.

Not FNO SOTA. Not CCF stretch. Not NS.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.ftc import sigmoid, softplus

DISCLAIMER = (
    "Integral-kernel operator: OMBU integral cell, not BEM-Net, not FNO SOTA, "
    "and not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "claimed_bem_net": False,
        "claimed_fno_sota": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class IntegralKernelConfig:
    n_dirs: int = 4
    window: float = 0.5
    w: float = 1.0


DEFAULT_CONFIG = IntegralKernelConfig()


def integral_cell(z: float, b_lo: float, b_hi: float) -> float:
    """``S(z + b_hi) - S(z + b_lo)``. Matches ``OperatorBlock(op='integral')``."""
    if b_hi == b_lo:
        raise ValueError("window is degenerate; collapse is a separate head")
    return softplus(z + b_hi) - softplus(z + b_lo)


def kernel(x: float, y: float, *, config: IntegralKernelConfig | None = None) -> float:
    """Volumetric kernel in ``z = w (x - y)``. Not BEM-Net."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.window <= 0.0:
        raise ValueError(f"window must be > 0, got {cfg.window}")
    half = 0.5 * cfg.window
    return integral_cell(cfg.w * (x - y), -half, half)


def _trap_weights(nodes: Sequence[float]) -> list[float]:
    if len(nodes) < 2:
        raise ValueError("need at least two nodes")
    weights = [0.0] * len(nodes)
    for i in range(len(nodes) - 1):
        half = 0.5 * (nodes[i + 1] - nodes[i])
        weights[i] += half
        weights[i + 1] += half
    return weights


def integral_kernel_apply(
    source: Sequence[float],
    coords: Sequence[float],
    params: Sequence[float] | None = None,
    *,
    config: IntegralKernelConfig | None = None,
    nodes: Sequence[float] | None = None,
) -> list[float]:
    """Apply ``(Kf)(x) = sum_j w_j K(x, y_j) f(y_j)``. Not BEM-Net."""
    cfg = DEFAULT_CONFIG if config is None else config
    ys = list(nodes) if nodes is not None else list(coords)
    if len(source) != len(ys):
        raise ValueError("source and nodes must have the same length")
    weights = _trap_weights(ys)
    scale = 1.0 if not params else float(params[0])
    out: list[float] = []
    for x in coords:
        acc = 0.0
        for y, f, weight in zip(ys, source, weights, strict=True):
            acc += kernel(x, y, config=cfg) * f * weight
        out.append(scale * acc)
    return out


def volterra_apply(
    source: Sequence[float],
    coords: Sequence[float],
    *,
    nodes: Sequence[float] | None = None,
) -> list[float]:
    """Named 1-D antiderivative ``int_{y0}^x f``. Heaviside limit of the kernel."""
    ys = list(nodes) if nodes is not None else list(coords)
    if len(source) != len(ys):
        raise ValueError("source and nodes must have the same length")
    if ys != list(coords):
        raise ValueError("volterra_apply requires coords == nodes")
    out = [0.0]
    for i in range(1, len(ys)):
        out.append(out[-1] + 0.5 * (source[i] + source[i - 1]) * (ys[i] - ys[i - 1]))
    return out


def worked_example() -> dict[str, float]:
    """Mass of a unit source on ``[0, 1]`` at ``x=0.5``, ``w=1``.

    ``softplus(0.5) - softplus(-0.5)`` is a mass, not ``||f||_1``.
    """
    mass = integral_cell(0.5, -1.0, 0.0)
    expected = softplus(0.5) - softplus(-0.5)
    g1_cell = integral_cell(0.5, -0.5, 0.5)
    g1_ref = softplus(1.0) - softplus(0.0)
    return {
        "mass": mass,
        "mass_expected": expected,
        "mass_err": abs(mass - expected),
        "g1_cell": g1_cell,
        "g1_ref": g1_ref,
        "g1_err": abs(g1_cell - g1_ref),
    }


def _rmse(pred: Sequence[float], target: Sequence[float]) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(pred, target, strict=True)) / float(len(pred))
    )


def _lstsq(columns: list[list[float]], target: Sequence[float]) -> list[float]:
    import numpy as np
    from numpy.typing import NDArray

    matrix: NDArray[np.float64] = np.column_stack(
        [np.asarray(col, dtype=np.float64) for col in columns]
    )
    rhs: NDArray[np.float64] = np.asarray(list(target), dtype=np.float64)
    coef, _residuals, _rank, _s = np.linalg.lstsq(matrix, rhs, rcond=None)
    return [float(v) for v in coef.tolist()]


def antiderivative_skill(
    *,
    seeds: int = 5,
    n: int = 33,
    n_dirs: int = 4,
) -> dict[str, object]:
    """G2: named Volterra ``int_0^x cos`` vs a same-width MLP kernel."""
    lo = 0.0
    hi = 0.5 * math.pi
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    fs = [math.cos(x) for x in xs]
    target = [math.sin(x) for x in xs]
    pred = volterra_apply(fs, xs, nodes=xs)
    max_err = max(abs(a - b) for a, b in zip(pred, target, strict=True))
    skill = 1.0 - _rmse(pred, target) / _rmse([0.0] * n, target)
    mlp_max: list[float] = []
    for seed in range(seeds):
        cols: list[list[float]] = []
        for k in range(n_dirs):
            alpha = 1.0 + 0.7 * float(k) + 0.15 * float(seed)
            bias = -0.4 + 0.2 * float(k) - 0.05 * float(seed)
            col = []
            weights = _trap_weights(xs)
            for x in xs:
                acc = 0.0
                for y, f, weight in zip(xs, fs, weights, strict=True):
                    acc += sigmoid(alpha * (x - y) + bias) * f * weight
                col.append(acc)
            cols.append(col)
        amps = _lstsq(cols, target)
        mlp = [
            sum(a * col[i] for a, col in zip(amps, cols, strict=True)) for i in range(n)
        ]
        mlp_max.append(max(abs(a - b) for a, b in zip(mlp, target, strict=True)))
    mlp_med = sorted(mlp_max)[len(mlp_max) // 2]
    return {
        "max_err": max_err,
        "skill": skill,
        "mlp_median": mlp_med,
        "below_1e3": max_err < 1e-3,
        "skill_positive": skill > 0.0,
        "beats_mlp": max_err < mlp_med,
        "g2_earned": max_err < 1e-3 and skill > 0.0 and max_err < mlp_med,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "IntegralKernelConfig",
    "antiderivative_skill",
    "honesty_payload",
    "integral_cell",
    "integral_kernel_apply",
    "kernel",
    "volterra_apply",
    "worked_example",
]
