# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""1-D fundamental-theorem cell (theory 09-03 + 09-17).

The cell is the unused ``integral`` role
``I = S(w x + b_hi) - S(w x + b_lo)`` with ``S' = sigma``. The
derivative head is FTC of that window, not a second pack. The
founding bias collapse (``delta -> 0``) of the same window is
``I / delta -> sigma(w x + b)``. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.

This is a 1-D FTC identity, not a VPINN / weak form, and not CCF
stretch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DISCLAIMER = (
    "1-D FTC integral cell; not a VPINN, not a weak form, and not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "claimed_weak_form": False,
        "claimed_vpinn": False,
        "ccf_stretch_claim": False,
        "theorem_prover_verified": False,
    }


def softplus(z: float) -> float:
    """``log(1 + exp(z))`` with the stable ``z + log1p(exp(-z))`` branch."""
    if z >= 0.0:
        return float(z + math.log1p(math.exp(-z)))
    return float(math.log1p(math.exp(z)))


def sigmoid(z: float) -> float:
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


def ftc_block(
    x: float,
    w: float,
    b_lo: float,
    b_hi: float,
    *,
    activation: str = "sigmoid",
) -> tuple[float, float, float]:
    """Closed-form integral cell, FTC derivative, and collapse head."""
    if activation != "sigmoid":
        raise ValueError(f"only sigmoid is implemented, got {activation!r}")
    if b_hi == b_lo:
        raise ValueError("window is degenerate; collapse is a separate head")
    z_lo = w * x + b_lo
    z_hi = w * x + b_hi
    integral = softplus(z_hi) - softplus(z_lo)
    deriv = w * (sigmoid(z_hi) - sigmoid(z_lo))
    collapse = integral / (b_hi - b_lo)
    return integral, deriv, collapse


def worked_example() -> dict[str, float]:
    """Spec 09-03 / 09-17 numbers at ``x=0``, ``delta=0.2``."""
    integral, deriv, collapse = ftc_block(0.0, 1.0, -0.1, 0.1)
    sig_hi = sigmoid(0.1)
    sig_lo = sigmoid(-0.1)
    return {
        "I": integral,
        "dI": deriv,
        "collapse": collapse,
        "sigmoid_0": sigmoid(0.0),
        "ftc_band": sig_hi - sig_lo,
        "collapse_err": abs(collapse - sigmoid(0.0)),
        "ftc_err": abs(deriv - (sig_hi - sig_lo)),
    }


@dataclass(frozen=True)
class FTCNetConfig:
    n_blocks: int = 2
    width: int = 12
    window: float = 0.6


@dataclass(frozen=True)
class DualFTCConfig:
    w_deriv: float = 1.0
    w_int: float = 1.0


DEFAULT_DUAL = DualFTCConfig()


@dataclass(frozen=True)
class Pack:
    w: float
    center: float
    half_window: float
    amp: float = 1.0


def _lo_hi(pack: Pack) -> tuple[float, float]:
    return pack.center - pack.half_window, pack.center + pack.half_window


def pack_integral(x: float, pack: Pack) -> float:
    b_lo, b_hi = _lo_hi(pack)
    integral, _d, _c = ftc_block(x, pack.w, b_lo, b_hi)
    return pack.amp * integral


def pack_deriv(x: float, pack: Pack) -> float:
    b_lo, b_hi = _lo_hi(pack)
    _i, deriv, _c = ftc_block(x, pack.w, b_lo, b_hi)
    return pack.amp * deriv


def identity_value(x: float, pack: Pack) -> float:
    return pack.amp * sigmoid(pack.w * x + pack.center)


def identity_deriv(x: float, pack: Pack) -> float:
    """``d/dx sigma(w x + c)`` of an identity cell, not an integral window."""
    s = sigmoid(pack.w * x + pack.center)
    return pack.amp * pack.w * s * (1.0 - s)


def evaluate_sum(
    xs: Sequence[float],
    packs: Sequence[Pack],
    kind: str,
) -> list[float]:
    fn = {
        "integral": pack_integral,
        "deriv": pack_deriv,
        "identity": identity_value,
        "identity_deriv": identity_deriv,
    }[kind]
    return [sum(fn(x, p) for p in packs) for x in xs]


def _lstsq(columns: list[list[float]], target: Sequence[float]) -> list[float]:
    matrix: NDArray[np.float64] = np.column_stack([np.asarray(col, dtype=np.float64) for col in columns])
    rhs: NDArray[np.float64] = np.asarray(list(target), dtype=np.float64)
    coef, _residuals, _rank, _s = np.linalg.lstsq(matrix, rhs, rcond=None)
    return [float(v) for v in coef.tolist()]


def _seed_packs(config: FTCNetConfig, seed: int) -> list[Pack]:
    half = 0.5 * config.window
    n = config.n_blocks * config.width
    packs: list[Pack] = []
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        center = -math.pi + 2.0 * math.pi * t + 0.05 * float(seed)
        w = 1.0 + 0.3 * float((i + seed) % 4)
        if (i + seed) % 2 == 1:
            w = -w
        packs.append(Pack(w=w, center=center, half_window=half, amp=1.0))
    return packs


def _with_amps(packs: Sequence[Pack], amps: Sequence[float]) -> list[Pack]:
    return [
        Pack(p.w, p.center, p.half_window, float(a))
        for p, a in zip(packs, amps, strict=True)
    ]


def fit_ftc_deriv(
    xs: Sequence[float],
    target: Sequence[float],
    packs: Sequence[Pack],
) -> list[Pack]:
    cols = [[pack_deriv(x, Pack(p.w, p.center, p.half_window, 1.0)) for x in xs] for p in packs]
    return _with_amps(packs, _lstsq(cols, target))


def fit_identity_deriv(
    xs: Sequence[float],
    target: Sequence[float],
    packs: Sequence[Pack],
) -> list[Pack]:
    cols = [[identity_deriv(x, Pack(p.w, p.center, p.half_window, 1.0)) for x in xs] for p in packs]
    return _with_amps(packs, _lstsq(cols, target))


def fit_dual(
    xs: Sequence[float],
    f: Sequence[float],
    F: Sequence[float],
    packs: Sequence[Pack],
    *,
    config: DualFTCConfig = DEFAULT_DUAL,
) -> list[Pack]:
    """Stacked ``[sqrt(w_d) dI; sqrt(w_I) I]`` against ``[f; F]``."""
    wd = math.sqrt(config.w_deriv)
    wi = math.sqrt(config.w_int)
    cols: list[list[float]] = []
    for p in packs:
        unit = Pack(p.w, p.center, p.half_window, 1.0)
        col = [wd * pack_deriv(x, unit) for x in xs]
        col.extend(wi * pack_integral(x, unit) for x in xs)
        cols.append(col)
    target = [wd * v for v in f]
    target.extend(wi * v for v in F)
    return _with_amps(packs, _lstsq(cols, target))


def max_abs(values: Sequence[float]) -> float:
    return max(abs(v) for v in values) if values else 0.0


@dataclass(frozen=True)
class DualFTCResult:
    loss: float
    r_D: list[float]
    r_I: list[float]
    max_r_D: float
    max_r_I: float


def dual_ftc_loss(
    I_vals: Sequence[float],
    dI_vals: Sequence[float],
    f_vals: Sequence[float],
    F_vals: Sequence[float],
    *,
    config: DualFTCConfig = DEFAULT_DUAL,
    I_a: float | None = None,
) -> DualFTCResult:
    """``r_D = dI - f``, ``r_I = I - I(a) - Integral_a^x f``. ``I_a`` defaults to ``I[0]``."""
    if not (len(I_vals) == len(dI_vals) == len(f_vals) == len(F_vals)):
        raise ValueError("dual_ftc_loss requires aligned sequences")
    ia = float(I_vals[0] if I_a is None else I_a)
    r_d = [di - fi for di, fi in zip(dI_vals, f_vals, strict=True)]
    r_i = [iv - ia - Fv for iv, Fv in zip(I_vals, F_vals, strict=True)]
    loss = config.w_deriv * sum(v * v for v in r_d) + config.w_int * sum(v * v for v in r_i)
    n = float(len(r_d))
    return DualFTCResult(loss / n, r_d, r_i, max_abs(r_d), max_abs(r_i))


def _grid(n: int = 65) -> list[float]:
    # [-pi, pi]
    lo, hi = -math.pi, math.pi
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def skill_report(*, seeds: int = 5, n_grid: int = 129) -> dict[str, object]:
    """G2 for 09-03: fit ``dI/dx = cos x`` vs identity-cell ``sigma'``."""
    xs = _grid(n_grid)
    target = [math.cos(x) for x in xs]
    zero_err = max_abs(target)
    ftc_errs: list[float] = []
    id_errs: list[float] = []
    cfg = FTCNetConfig()
    for seed in range(seeds):
        packs = _seed_packs(cfg, seed)
        ftc = fit_ftc_deriv(xs, target, packs)
        ident = fit_identity_deriv(xs, target, packs)
        ftc_errs.append(max_abs([a - b for a, b in zip(evaluate_sum(xs, ftc, "deriv"), target, strict=True)]))
        id_errs.append(
            max_abs([a - b for a, b in zip(evaluate_sum(xs, ident, "identity_deriv"), target, strict=True)])
        )
    ftc_med = sorted(ftc_errs)[len(ftc_errs) // 2]
    id_med = sorted(id_errs)[len(id_errs) // 2]
    return {
        "ftc_errors": ftc_errs,
        "identity_errors": id_errs,
        "ftc_median": ftc_med,
        "identity_median": id_med,
        "zero_error": zero_err,
        "skill": 1.0 - ftc_med / zero_err if zero_err else 0.0,
        "beats_identity": ftc_med < id_med,
        "below_1e4": ftc_med < 1e-4,
        "claimed_weak_form": False,
    }


def dual_skill_report(*, seeds: int = 5, n_grid: int = 129) -> dict[str, object]:
    """G2 for 09-17: dual residual vs derivative-only on ``f = cos``."""
    xs = _grid(n_grid)
    f = [math.cos(x) for x in xs]
    F = [math.sin(x) - math.sin(xs[0]) for x in xs]
    cfg = FTCNetConfig()
    dual_rd: list[float] = []
    dual_ri: list[float] = []
    only_ri: list[float] = []
    zero_i = max_abs(F)
    for seed in range(seeds):
        packs = _seed_packs(cfg, seed)
        deriv_only = fit_ftc_deriv(xs, f, packs)
        dual = fit_dual(xs, f, F, packs)
        d_only_I = evaluate_sum(xs, deriv_only, "integral")
        d_only_dI = evaluate_sum(xs, deriv_only, "deriv")
        dual_I = evaluate_sum(xs, dual, "integral")
        dual_dI = evaluate_sum(xs, dual, "deriv")
        only = dual_ftc_loss(d_only_I, d_only_dI, f, F, I_a=0.0)
        both = dual_ftc_loss(dual_I, dual_dI, f, F, I_a=0.0)
        dual_rd.append(both.max_r_D)
        dual_ri.append(both.max_r_I)
        only_ri.append(only.max_r_I)
    med_rd = sorted(dual_rd)[len(dual_rd) // 2]
    med_ri = sorted(dual_ri)[len(dual_ri) // 2]
    med_only = sorted(only_ri)[len(only_ri) // 2]
    return {
        "dual_max_r_D": dual_rd,
        "dual_max_r_I": dual_ri,
        "deriv_only_max_r_I": only_ri,
        "median_r_D": med_rd,
        "median_r_I": med_ri,
        "median_deriv_only_r_I": med_only,
        "skill": 1.0 - med_ri / zero_i if zero_i else 0.0,
        "ri_below_deriv_only": med_ri < med_only,
        "below_1e4": med_rd < 1e-4 and med_ri < 1e-4,
        "claimed_vpinn": False,
    }


__all__ = [
    "DEFAULT_DUAL",
    "DISCLAIMER",
    "DualFTCConfig",
    "DualFTCResult",
    "FTCNetConfig",
    "Pack",
    "dual_ftc_loss",
    "dual_skill_report",
    "evaluate_sum",
    "fit_dual",
    "fit_ftc_deriv",
    "fit_identity_deriv",
    "ftc_block",
    "honesty_payload",
    "identity_deriv",
    "identity_value",
    "max_abs",
    "pack_deriv",
    "pack_integral",
    "sigmoid",
    "skill_report",
    "softplus",
    "worked_example",
]
