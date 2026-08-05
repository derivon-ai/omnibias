# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Non-rigorous float oracles (RK4) used only to check soundness of the enclosures."""

from __future__ import annotations

from collections.abc import Callable, Sequence

FloatField = Callable[[Sequence[float]], list[float]]


def rk4(field: FloatField, y0: Sequence[float], t0: float, t1: float, n: int) -> list[float]:
    """Classic RK4 integration of ``y' = field(y)`` from ``t0`` to ``t1`` in ``n`` steps."""
    h = (t1 - t0) / n
    y = list(y0)
    for _ in range(n):
        k1 = field(y)
        k2 = field([y[i] + 0.5 * h * k1[i] for i in range(len(y))])
        k3 = field([y[i] + 0.5 * h * k2[i] for i in range(len(y))])
        k4 = field([y[i] + h * k3[i] for i in range(len(y))])
        y = [y[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(y))]
    return y


def harmonic_float(omega: float = 1.0) -> FloatField:
    return lambda y: [omega * y[1], -omega * y[0]]


def hopf_float(mu: float = 1.0) -> FloatField:
    def f(y: Sequence[float]) -> list[float]:
        r2 = y[0] * y[0] + y[1] * y[1]
        return [mu * y[0] - y[1] - y[0] * r2, y[0] + mu * y[1] - y[1] * r2]

    return f


def radial_float(mu: float = 1.0) -> FloatField:
    return lambda y: [mu * y[0] - y[0] ** 3]
