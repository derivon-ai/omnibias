# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The time-scale (delta) integral.

On a discrete scale the delta integral is the exact graininess-weighted sum

.. math::

    \int_a^b f(t)\,\Delta t = \sum_{t \in [a, b) \cap T} \mu(t)\, f(t),

which satisfies the fundamental theorem ``int_a^b f^Delta Delta t = f(b) - f(a)`` exactly.
On ``R`` (graininess ``0``) it is the ordinary Riemann integral, approximated here by a
composite Simpson quadrature. The discrete branch is **closed-form / exact**; the ``R``
branch is **numerical**.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.timescale._core.timescale import TimeScale


def delta_integral(
    f: Callable[[float], float], a: float, b: float, ts: TimeScale, *, quad_points: int = 1024
) -> float:
    r"""The delta integral ``int_a^b f(t) Delta t`` on the time scale ``ts``.

    Discrete scales return the exact graininess-weighted sum over grid points in ``[a, b)``;
    the continuum returns a composite-Simpson approximation with ``quad_points`` panels.
    Requires ``a <= b``.
    """
    if a > b:
        raise ValueError(f"delta_integral needs a <= b, got a={a}, b={b}")
    if a == b:
        return 0.0
    if ts.kind == "reals":
        return _simpson(f, a, b, quad_points)
    total = 0.0
    for t in ts.grid(a, b):
        if t >= b:  # the half-open [a, b) convention
            continue
        total += ts.mu(t) * f(t)
    return total


def _simpson(f: Callable[[float], float], a: float, b: float, panels: int) -> float:
    if panels % 2 == 1:
        panels += 1
    step = (b - a) / panels
    total = f(a) + f(b)
    for i in range(1, panels):
        total += (4.0 if i % 2 else 2.0) * f(a + i * step)
    return total * step / 3.0


__all__ = ["delta_integral"]
