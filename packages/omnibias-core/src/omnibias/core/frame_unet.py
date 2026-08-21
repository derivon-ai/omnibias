# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Frame-UNet (theory 09-04).

The encoder raises pack order (01-07 band selector). The decoder is
integral synthesis. Skips store a **band** FTC identity
``dI/dx = w (sigma(z+b_hi) - sigma(z+b_lo))`` and an optional
**collapse** head ``sigma^(n)`` separately. Mixing them is a bug.

Encoder collapse is founding bias collapse (``delta -> 0``). The
decoder gap is the window knob. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.

``sigma'`` is not an admissible wavelet (01-06). Frames are not
orthonormal and not compactly supported. Not Littlewood–Paley
completeness. Not ImageNet. Not CCF stretch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.core.ftc import ftc_block, sigmoid, softplus
from omnibias.core.polynomials import sigmoid_polynomial_coeffs

DISCLAIMER = (
    "Frame-UNet: order encoder + integral decoder; band skip is not "
    "a collapse head; sigma' is not admissible; not ImageNet; not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "sigma_prime_admissible": False,
        "littlewood_paley_complete": False,
        "imagenet_claim": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class FrameUNetConfig:
    max_order: int = 2
    width: int = 8
    window: float = 0.2


DEFAULT_CONFIG = FrameUNetConfig()


def encoder_head(x: float, w: float, center: float, order: int) -> float:
    """``w^n sigma^(n)(w x + center)`` via the shared Eulerian tower."""
    if order < 0:
        raise ValueError("order must be >= 0")
    z = w * x + center
    s = sigmoid(z)
    acc = 0.0
    for c in reversed(sigmoid_polynomial_coeffs(order)):
        acc = acc * s + float(c)
    return (w**order) * acc


def decoder_cell(x: float, w: float, center: float, window: float) -> tuple[float, float]:
    """Integral cell and its FTC band ``dI/dx``."""
    if window <= 0.0:
        raise ValueError("window must be positive")
    half = 0.5 * window
    integral, deriv, _collapse = ftc_block(x, w, center - half, center + half)
    return integral, deriv


def band_skip(x: float, w: float, center: float, window: float) -> float:
    """Window-role skip: FTC of the decoder slab, not a collapse head."""
    _integral, deriv = decoder_cell(x, w, center, window)
    return deriv


def collapse_skip(x: float, w: float, center: float, order: int) -> float:
    """Optional encoder collapse skip. Distinct from :func:`band_skip`."""
    return encoder_head(x, w, center, order)


def worked_example() -> dict[str, float]:
    """Spec 09-04 numbers: ``delta=0.2`` at ``x=0``, order-1 collapse."""
    x = 0.0
    w = 1.0
    center = 0.0
    window = 0.2
    band = band_skip(x, w, center, window)
    collapse = collapse_skip(x, w, center, 1)
    sig_hi = sigmoid(0.1)
    sig_lo = sigmoid(-0.1)
    band_closed = w * (sig_hi - sig_lo)
    collapse_closed = 0.25
    return {
        "band": band,
        "collapse": collapse,
        "band_closed": band_closed,
        "collapse_closed": collapse_closed,
        "band_err": abs(band - band_closed),
        "collapse_err": abs(collapse - collapse_closed),
        "skip_gap": abs(band - collapse),
        "I": softplus(0.1) - softplus(-0.1),
    }


def _centers(width: int, seed: int) -> list[float]:
    if width < 1:
        raise ValueError("width must be >= 1")
    return [
        -math.pi + 2.0 * math.pi * (i / max(width - 1, 1)) + 0.04 * float(seed)
        for i in range(width)
    ]


def frame_unet_forward(
    x: float,
    *,
    config: FrameUNetConfig | None = None,
    amps: Sequence[float] | None = None,
    seed: int = 0,
) -> tuple[float, dict[str, object]]:
    """Decoder synthesis plus labelled band / collapse skips."""
    cfg = DEFAULT_CONFIG if config is None else config
    centers = _centers(cfg.width, seed)
    n_scales = cfg.max_order + 1
    n_units = n_scales * cfg.width
    coef = [1.0] * n_units if amps is None else [float(a) for a in amps]
    if len(coef) != n_units:
        raise ValueError(f"amps length must be {n_units}, got {len(coef)}")
    y = 0.0
    bands: list[float] = []
    collapses: list[float] = []
    k = 0
    for order in range(n_scales):
        for center in centers:
            integral, band = decoder_cell(x, 1.0, center, cfg.window)
            y += coef[k] * integral
            bands.append(band)
            collapses.append(collapse_skip(x, 1.0, center, max(order, 1) if order > 0 else 0))
            k += 1
    skips: dict[str, object] = {
        "band": bands,
        "collapse": collapses,
        "kinds": ("band", "collapse"),
    }
    return y, skips


def _lstsq(columns: list[list[float]], target: Sequence[float]) -> list[float]:
    matrix: NDArray[np.float64] = np.column_stack([np.asarray(col, dtype=np.float64) for col in columns])
    rhs: NDArray[np.float64] = np.asarray(list(target), dtype=np.float64)
    coef, _residuals, _rank, _s = np.linalg.lstsq(matrix, rhs, rcond=None)
    return [float(v) for v in coef.tolist()]


def _unet_columns(xs: Sequence[float], config: FrameUNetConfig, seed: int) -> list[list[float]]:
    centers = _centers(config.width, seed)
    cols: list[list[float]] = []
    for order in range(config.max_order + 1):
        for center in centers:
            cols.append([decoder_cell(x, 1.0, center, config.window)[0] for x in xs])
            cols.append([encoder_head(x, 1.0, center, order) for x in xs])
    return cols


def _scan_columns(xs: Sequence[float], config: FrameUNetConfig, seed: int) -> list[list[float]]:
    """Same-width Scan-Net smoke: one identity-sigmoid bank of ``width``."""
    centers = _centers(config.width, seed)
    return [[sigmoid(x + center) for x in xs] for center in centers]


def _mse(pred: Sequence[float], target: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(pred, target, strict=True)) / float(len(pred))


def denoise_skill(*, seeds: int = 5, n_grid: int = 97) -> dict[str, object]:
    """G2: 1-D sine + noise vs zero predictor and a same-width Scan-Net."""
    cfg = FrameUNetConfig()
    xs = [-math.pi + 2.0 * math.pi * i / (n_grid - 1) for i in range(n_grid)]
    clean = [math.sin(x) + 0.25 * math.sin(4.0 * x) for x in xs]
    train_idx = list(range(0, n_grid, 2))
    hold_idx = list(range(1, n_grid, 2))
    xs_tr = [xs[i] for i in train_idx]
    xs_ho = [xs[i] for i in hold_idx]
    clean_ho = [clean[i] for i in hold_idx]
    zero = _mse([0.0] * len(clean_ho), clean_ho)
    unet: list[float] = []
    scan: list[float] = []
    for seed in range(seeds):
        noisy_tr = [
            clean[i] + 0.08 * math.sin(17.0 * xs[i] + 0.4 * float(seed)) for i in train_idx
        ]
        u_cols = _unet_columns(xs_tr, cfg, seed)
        s_cols = _scan_columns(xs_tr, cfg, seed)
        a_u = _lstsq(u_cols, noisy_tr)
        a_s = _lstsq(s_cols, noisy_tr)
        u_ho_cols = _unet_columns(xs_ho, cfg, seed)
        s_ho_cols = _scan_columns(xs_ho, cfg, seed)
        pred_u = [sum(c * a for c, a in zip(col, a_u, strict=True)) for col in zip(*u_ho_cols, strict=True)]
        pred_s = [sum(c * a for c, a in zip(col, a_s, strict=True)) for col in zip(*s_ho_cols, strict=True)]
        unet.append(_mse(pred_u, clean_ho))
        scan.append(_mse(pred_s, clean_ho))
    u_med = sorted(unet)[len(unet) // 2]
    s_med = sorted(scan)[len(scan) // 2]
    worse_all = all(u > s for u, s in zip(unet, scan, strict=True))
    return {
        "unet_mse": unet,
        "scan_mse": scan,
        "unet_median": u_med,
        "scan_median": s_med,
        "zero_mse": zero,
        "below_zero": u_med < zero,
        "not_worse_than_scan": u_med <= s_med and not worse_all,
        "sigma_prime_admissible": False,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "FrameUNetConfig",
    "band_skip",
    "collapse_skip",
    "decoder_cell",
    "denoise_skill",
    "encoder_head",
    "frame_unet_forward",
    "honesty_payload",
    "worked_example",
]
