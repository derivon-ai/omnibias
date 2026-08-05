# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Linear dynamic equations ``y^Delta = p(t) y + r(t)`` on a time scale.

Two equivalent solvers on a discrete scale:

* :func:`solve_linear_dynamic` -- the direct forward recursion
  ``y(sigma(t)) = y(t) + mu(t)(p(t) y(t) + r(t))``;
* :func:`variation_of_constants` -- the closed-form solution
  ``y(t) = e_p(t, t0) y0 + int_{t0}^{t} e_p(t, sigma(tau)) r(tau) Delta tau``.

Both unify the recurrence (``hZ``) and ODE (``R``) registers: the same equation is a linear
recurrence on a mesh and a linear ODE on the continuum, and they agree as ``mu -> 0``.
"""

from __future__ import annotations

from omnibias.timescale._core.exponential import Coefficient, _as_callable, hilger_exponential
from omnibias.timescale._core.timescale import TimeScale


def solve_linear_dynamic(
    p: Coefficient,
    r: Coefficient,
    y0: float,
    ts: TimeScale,
    t0: float,
    tn: float,
) -> list[tuple[float, float]]:
    r"""Solve ``y^Delta = p y + r``, ``y(t0) = y0`` by forward recursion over ``[t0, tn]``.

    Returns ``[(t, y(t)), ...]`` at every scale point from ``t0`` to ``tn`` (discrete scales
    only -- the continuum has no canonical step).
    """
    if ts.kind == "reals":
        raise ValueError("solve_linear_dynamic needs a discrete scale; R has no canonical step")
    pf, rf = _as_callable(p), _as_callable(r)
    grid = list(ts.grid(t0, tn))
    if not grid:
        raise ValueError(f"no scale points in [{t0}, {tn}]")
    out: list[tuple[float, float]] = [(grid[0], y0)]
    y = y0
    for t in grid[:-1]:
        y = y + ts.mu(t) * (pf(t) * y + rf(t))
        out.append((ts.sigma(t), y))
    return out


def variation_of_constants(
    p: Coefficient,
    r: Coefficient,
    y0: float,
    t: float,
    ts: TimeScale,
    t0: float,
) -> float:
    r"""Closed-form ``y(t) = e_p(t,t0) y0 + int_{t0}^t e_p(t, sigma(tau)) r(tau) Delta tau``."""
    if ts.kind == "reals":
        raise ValueError("variation_of_constants is implemented for discrete scales")
    rf = _as_callable(r)
    homogeneous = hilger_exponential(p, t, t0, ts) * y0
    particular = 0.0
    for tau in ts.grid(t0, t):
        if tau >= t:  # half-open [t0, t)
            continue
        particular += ts.mu(tau) * hilger_exponential(p, t, ts.sigma(tau), ts) * rf(tau)
    return float(homogeneous + particular)


__all__ = [
    "solve_linear_dynamic",
    "variation_of_constants",
]
