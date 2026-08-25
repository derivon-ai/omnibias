# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Designed causal ``sigma^(n)`` taps (theory 05-02 G5).

Order 0 is the logistic survival / integral-role tail
``c * sigma(tau - alpha * k)``. That is the leaky-integrator kernel: a
one-sided exponential envelope, not a mid-lag bump. Pack order is a band
selector (spec 01-07); ``n = 1`` (``sigma'``) is band-pass and is the
wrong class for an AR(1) / S4D comparison.

Taps are founding-tower evaluations (``delta -> 0`` register). This is
not temperature collapse and not ``omnibias.struct``.
"""

from __future__ import annotations

import math

from omnibias.core.locus import sigma_n


def leaky_integrator_init(rho: float) -> tuple[float, float, float]:
    """``(coeff, alpha, tau)`` near the AR(1) impulse ``(1 - rho) rho^k``.

    Far-tail match: ``sigma(tau - alpha k) ~ exp(tau - alpha k)`` so
    ``alpha = -log(rho)`` and ``coeff = 1 - rho`` at ``tau = 0``.
    """
    value = float(rho)
    if not (0.0 < value < 1.0):
        raise ValueError(f"rho must lie in (0, 1), got {rho}")
    return (1.0 - value, -math.log(value), 0.0)


def causal_transverse_taps(
    *,
    width: int,
    coeff: float,
    alpha: float,
    tau: float,
    order: int = 0,
    base: str = "sigmoid",
) -> tuple[float, ...]:
    """Causal taps for lag ``k = 0, ..., width - 1``.

    * ``order == 0``: ``c * sigma^(0)(tau - alpha * k)`` (low-pass tail).
    * ``order >= 1``: ``c * sigma^(n)(alpha * (k - tau))`` (band-pass bump).
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    scale = float(alpha)
    shift = float(tau)
    gain = float(coeff)
    taps: list[float] = []
    for lag in range(int(width)):
        k = float(lag)
        argument = shift - scale * k if order == 0 else scale * (k - shift)
        taps.append(gain * sigma_n(base, argument, order))
    return tuple(taps)
