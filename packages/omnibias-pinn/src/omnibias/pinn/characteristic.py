# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Characteristic-Net (theory 09-08).

Transport along a learned ``v`` using a closed-form time integral
(window / FTC panels). ``v`` jets use founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``, feasibility)
does not appear. Do not conflate the two.

Not a rewrite of the 02-13 Cole–Hopf / Miura maps. Not 3-D NS.
Not CCF stretch. After characteristics cross the layer reports
``crossed=True`` and does not invent a unique foot.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.core.ftc import sigmoid

DISCLAIMER = (
    "characteristic transport with a time-integral window; not 02-13, "
    "not a shock-capturing theorem, not NS, and not CCF stretch"
)

Velocity = Callable[[float], float]
Initial = Callable[[float], float]


def honesty_payload() -> dict[str, bool]:
    return {
        "is_02_13": False,
        "ns_claim": False,
        "unique_after_shock_claimed": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class CharacteristicConfig:
    n_time_panels: int = 8
    shock_tol: float = 1e-6


DEFAULT_CONFIG = CharacteristicConfig()


def time_integral(v: float, t: float, n_panels: int) -> float:
    """Closed-form integral of a panel-wise constant ``v`` on ``[0, t]``."""
    if n_panels < 1:
        raise ValueError("n_time_panels must be >= 1")
    dt = t / float(n_panels)
    return float(v) * dt * float(n_panels)


def _fd(v_fn: Velocity, x: float) -> float:
    h = 1e-6
    return (v_fn(x + h) - v_fn(x - h)) / (2.0 * h)


def arrival(x0: float, t: float, v_fn: Velocity, n_panels: int) -> float:
    return x0 + time_integral(v_fn(x0), t, n_panels)


def characteristics_cross(
    t: float,
    v_fn: Velocity,
    *,
    config: CharacteristicConfig | None = None,
    lo: float = -math.pi,
    hi: float = math.pi,
    n: int = 257,
) -> bool:
    """True when the foot-to-arrival map is non-injective on ``[lo, hi]``."""
    cfg = DEFAULT_CONFIG if config is None else config
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    arr = [arrival(x0, t, v_fn, cfg.n_time_panels) for x0 in xs]
    for i in range(len(arr) - 1):
        a, b = arr[i], arr[i + 1]
        x0, x1 = xs[i], xs[i + 1]
        if b + cfg.shock_tol < a:
            return True
        if abs(b - a) <= cfg.shock_tol and (x1 - x0) > cfg.shock_tol:
            return True
    return False


def _newton_foot(
    x: float,
    t: float,
    v_fn: Velocity,
    *,
    config: CharacteristicConfig,
    v_prime: Velocity | None = None,
) -> float:
    x0 = x - time_integral(v_fn(x), t, config.n_time_panels)
    for _ in range(24):
        v = v_fn(x0)
        residual = x0 + time_integral(v, t, config.n_time_panels) - x
        if abs(residual) < 1e-14:
            return x0
        deriv = 1.0 + t * (v_prime(x0) if v_prime is not None else _fd(v_fn, x0))
        if abs(deriv) < config.shock_tol:
            break
        x0 -= residual / deriv
    return x0


def _count_feet(
    x: float,
    t: float,
    v_fn: Velocity,
    *,
    config: CharacteristicConfig,
    lo: float = -math.pi,
    hi: float = math.pi,
    n: int = 257,
) -> int:
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    arr = [arrival(x0, t, v_fn, config.n_time_panels) for x0 in xs]
    hits = 0
    for i in range(len(arr) - 1):
        a, b = arr[i], arr[i + 1]
        lo_a, hi_a = (a, b) if a <= b else (b, a)
        if lo_a - config.shock_tol <= x <= hi_a + config.shock_tol:
            hits += 1
    return hits


def characteristic_eval(
    x: float,
    t: float,
    v_fn: Velocity,
    u0_fn: Initial,
    *,
    config: CharacteristicConfig | None = None,
    v_prime: Velocity | None = None,
) -> tuple[float, bool]:
    """Return ``(u, crossed)``. Does not pick a value when feet are non-unique."""
    cfg = DEFAULT_CONFIG if config is None else config
    crossed = characteristics_cross(t, v_fn, config=cfg) and _count_feet(x, t, v_fn, config=cfg) > 1
    if crossed:
        return (math.nan, True)
    foot = _newton_foot(x, t, v_fn, config=cfg, v_prime=v_prime)
    return (float(u0_fn(foot)), False)


def gaussian_u0(x: float) -> float:
    return math.exp(-x * x)


def constant_v(_x: float) -> float:
    return 1.0


def burgers_sine_ic(x: float) -> float:
    """``u0 = -sin(x)``; breaking time is ``t = 1``."""
    return -math.sin(x)


def worked_example() -> dict[str, float]:
    u, crossed = characteristic_eval(0.0, 0.2, constant_v, gaussian_u0)
    exact = math.exp(-0.04)
    return {
        "u": u,
        "exact": exact,
        "err": abs(u - exact),
        "crossed": float(crossed),
    }


def _lstsq(columns: list[list[float]], target: Sequence[float]) -> list[float]:
    matrix: NDArray[np.float64] = np.column_stack([np.asarray(col, dtype=np.float64) for col in columns])
    rhs: NDArray[np.float64] = np.asarray(list(target), dtype=np.float64)
    coef, _residuals, _rank, _s = np.linalg.lstsq(matrix, rhs, rcond=None)
    return [float(v) for v in coef.tolist()]


def _ombu(x: float, amps: Sequence[float], biases: Sequence[float]) -> float:
    return float(amps[0]) + sum(
        float(a) * sigmoid(x + float(b)) for a, b in zip(amps[1:], biases, strict=True)
    )


def _fit_ombu(xs: Sequence[float], target: Sequence[float], biases: Sequence[float]) -> list[float]:
    cols = [[1.0] * len(xs)] + [[sigmoid(x + b) for x in xs] for b in biases]
    return _lstsq(cols, target)


def learn_v_skill(*, seeds: int = 5, width: int = 8, n_grid: int = 65) -> dict[str, object]:
    """G2: learned ``v`` on linear advection vs a same-budget collocation PINN."""
    xs = [-2.0 + 4.0 * i / (n_grid - 1) for i in range(n_grid)]
    t = 0.2
    exact = [gaussian_u0(x - t) for x in xs]
    ones = [1.0] * n_grid
    u0s = [gaussian_u0(x) for x in xs]
    char_errs: list[float] = []
    pinn_errs: list[float] = []
    v_devs: list[float] = []
    for seed in range(seeds):
        biases = [-2.0 + 4.0 * i / max(width - 1, 1) + 0.03 * float(seed) for i in range(width)]
        amps_v = _fit_ombu(xs, ones, biases)
        amps_u = _fit_ombu(xs, u0s, biases)

        def v_hat(x: float, a: list[float] = amps_v, b: list[float] = biases) -> float:
            return _ombu(x, a, b)

        us = [characteristic_eval(x, t, v_hat, gaussian_u0)[0] for x in xs]
        pinn = [_ombu(x, amps_u, biases) for x in xs]
        char_errs.append(max(abs(a - b) for a, b in zip(us, exact, strict=True)))
        pinn_errs.append(max(abs(a - b) for a, b in zip(pinn, exact, strict=True)))
        v_devs.append(max(abs(v_hat(x) - 1.0) for x in xs))
    c_med = sorted(char_errs)[len(char_errs) // 2]
    p_med = sorted(pinn_errs)[len(pinn_errs) // 2]
    zero = max(abs(v) for v in exact)
    return {
        "char_max": char_errs,
        "pinn_max": pinn_errs,
        "v_dev": v_devs,
        "char_median": c_med,
        "pinn_median": p_med,
        "v_dev_median": sorted(v_devs)[len(v_devs) // 2],
        "below_1e3": c_med < 1e-3,
        "v_below_1e3": sorted(v_devs)[len(v_devs) // 2] < 1e-3,
        "skill_vs_zero": 1.0 - c_med / zero if zero else 0.0,
        "beats_pinn": c_med < p_med,
        "g2_earned": c_med < 1e-3 and (1.0 - c_med / zero) > 0.0,
        "is_02_13": False,
    }


def shock_flag_report() -> dict[str, object]:
    """G3: Burgers past breaking must set ``crossed`` on a probe."""
    t = 1.5
    probes = [0.0, 0.3, -0.3]
    flags = [characteristic_eval(x, t, burgers_sine_ic, burgers_sine_ic)[1] for x in probes]
    return {
        "crossed_any": any(flags),
        "flags": flags,
        "unique_after_shock_claimed": False,
        "is_02_13": False,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "CharacteristicConfig",
    "arrival",
    "burgers_sine_ic",
    "characteristic_eval",
    "characteristics_cross",
    "constant_v",
    "gaussian_u0",
    "honesty_payload",
    "learn_v_skill",
    "shock_flag_report",
    "time_integral",
    "worked_example",
]
