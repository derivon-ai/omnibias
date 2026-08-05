# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The Hilger exponential and the regressive ``circle-plus`` group.

For a *regressive* coefficient ``p`` (one with ``1 + mu(t) p(t) != 0`` for all ``t``), the
Hilger exponential ``e_p(t, s)`` is the unique solution of the dynamic initial value problem
``y^Delta = p(t) y``, ``y(s) = 1``. On a discrete scale it is the product

.. math::

    e_p(t, s) = \prod_{\tau \in [s, t) \cap T} \bigl(1 + \mu(\tau) p(\tau)\bigr),

and on ``R`` it is ``exp(int_s^t p)``. The regressive functions form an abelian group under
``(p oplus q)(t) = p(t) + q(t) + mu(t) p(t) q(t)`` with inverse
``(ominus p)(t) = -p(t) / (1 + mu(t) p(t))``; ``e_{p oplus q} = e_p e_q`` and
``e_{ominus p} = 1 / e_p``. As ``mu -> 0`` the cylinder transformation
``xi_mu(z) = log(1 + mu z)/mu -> z`` and ``e_p -> exp`` -- the founding continuous limit.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from omnibias.timescale._core.timescale import TimeScale

Coefficient = Callable[[float], float] | float


def _as_callable(p: Coefficient) -> Callable[[float], float]:
    if callable(p):
        return p
    value = float(p)
    return lambda _t: value


def is_regressive(p: Coefficient, ts: TimeScale, a: float, b: float) -> bool:
    """Whether ``1 + mu(t) p(t) != 0`` across the grid of ``[a, b]`` (the group condition)."""
    pf = _as_callable(p)
    if ts.kind == "reals":
        return True  # mu == 0 => 1 + 0 = 1 != 0 always
    return all(abs(1.0 + ts.mu(t) * pf(t)) > 1e-12 for t in ts.grid(a, b))


def hilger_exponential(p: Coefficient, t: float, s: float, ts: TimeScale) -> float:
    r"""The Hilger exponential ``e_p(t, s)`` on the time scale ``ts``.

    Solves ``y^Delta = p y``, ``y(s) = 1``. Supports ``t < s`` via ``e_p(t,s) = 1/e_p(s,t)``.
    """
    if t == s:
        return 1.0
    if t < s:
        return 1.0 / hilger_exponential(p, s, t, ts)
    pf = _as_callable(p)
    if ts.kind == "reals":
        return math.exp(_integrate(pf, s, t))
    product = 1.0
    for tau in ts.grid(s, t):
        if tau >= t:  # half-open [s, t)
            continue
        product *= 1.0 + ts.mu(tau) * pf(tau)
    return product


def circle_plus(p: Coefficient, q: Coefficient, ts: TimeScale) -> Callable[[float], float]:
    r"""The group sum ``(p oplus q)(t) = p(t) + q(t) + mu(t) p(t) q(t)``."""
    pf, qf = _as_callable(p), _as_callable(q)
    return lambda t: pf(t) + qf(t) + ts.mu(t) * pf(t) * qf(t)


def circle_minus(p: Coefficient, ts: TimeScale) -> Callable[[float], float]:
    r"""The group inverse ``(ominus p)(t) = -p(t) / (1 + mu(t) p(t))``."""
    pf = _as_callable(p)
    return lambda t: -pf(t) / (1.0 + ts.mu(t) * pf(t))


def cylinder(z: float, mu: float) -> float:
    r"""The Hilger cylinder transformation ``xi_mu(z) = log(1 + mu z)/mu`` (``z`` at ``mu=0``)."""
    if mu == 0.0:
        return z
    return math.log(1.0 + mu * z) / mu


def _integrate(f: Callable[[float], float], a: float, b: float, panels: int = 2048) -> float:
    if panels % 2 == 1:
        panels += 1
    step = (b - a) / panels
    total = f(a) + f(b)
    for i in range(1, panels):
        total += (4.0 if i % 2 else 2.0) * f(a + i * step)
    return total * step / 3.0


__all__ = [
    "circle_minus",
    "circle_plus",
    "cylinder",
    "hilger_exponential",
    "is_regressive",
]
