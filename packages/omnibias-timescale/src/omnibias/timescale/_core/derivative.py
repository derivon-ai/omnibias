# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The Hilger delta / nabla derivatives -- the founding collapse on a variable mesh.

The delta (forward) derivative of ``f`` at ``t`` on a time scale ``T`` is

.. math::

    f^{\Delta}(t) = \begin{cases}
      \dfrac{f(\sigma(t)) - f(t)}{\mu(t)}, & t \text{ right-scattered},\\[1ex]
      \lim_{s\to t} \dfrac{f(t) - f(s)}{t - s} = f'(t), & t \text{ right-dense}.
    \end{cases}

This is the **founding bias collapse** in its most general form. omnibias's founding move is
the ``delta -> 0`` limit in which a finite difference of many biases collapses to the smooth
derivative ``sigma^(K-1)`` (see :mod:`omnibias.difference`). Here the *graininess* ``mu(t)``
plays the role of ``delta``: on a scattered mesh ``f^Delta`` is literally a finite
difference, and as ``mu -> 0`` (the mesh becoming ``R``) it collapses to the ordinary
derivative -- the same founding derivative sense, never the ``beta -> inf`` feasibility
penalty.

The activation-aware :func:`delta_derivative_tower` makes the three registers one operator:
it **dispatches** to the closed-form omnibias derivative tower on ``R``, to the
:mod:`omnibias.difference` forward difference on ``hZ``, and to the
:mod:`omnibias.qcalculus` Jackson q-derivative on the quantum scale.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.difference import finite_difference_estimate
from omnibias.qcalculus import q_derivative as _jackson_q_derivative
from omnibias.timescale._core.timescale import TimeScale


def delta_derivative(
    f: Callable[[float], float], t: float, ts: TimeScale, *, fprime: Callable[[float], float] | None = None
) -> float:
    r"""The delta derivative ``f^Delta(t)`` of a callable on the time scale ``ts``.

    At a right-scattered ``t`` this is the exact quotient ``(f(sigma(t)) - f(t))/mu(t)``. At
    a right-dense ``t`` (e.g. anywhere on ``R``) it is the ordinary derivative ``f'(t)``,
    which needs an analytic ``fprime`` (a difference quotient cannot form it from one point).
    """
    if ts.is_right_scattered(t):
        return float((f(ts.sigma(t)) - f(t)) / ts.mu(t))
    if fprime is None:
        raise ValueError(
            f"t={t} is right-dense on this time scale; supply fprime for the derivative limit"
        )
    return float(fprime(t))


def nabla_derivative(
    f: Callable[[float], float], t: float, ts: TimeScale, *, fprime: Callable[[float], float] | None = None
) -> float:
    r"""The nabla derivative ``f^nabla(t) = (f(t) - f(rho(t)))/nu(t)`` (backward analogue)."""
    if ts.is_left_scattered(t):
        return float((f(t) - f(ts.rho(t))) / ts.nu(t))
    if fprime is None:
        raise ValueError(
            f"t={t} is left-dense on this time scale; supply fprime for the derivative limit"
        )
    return float(fprime(t))


def _mid(iv: Interval) -> float:
    return 0.5 * (iv.lo + iv.hi)


def sigma_value(name: str, t: float) -> float:
    """Closed-form activation value ``sigma(t)`` (order-0 entry of the omnibias tower)."""
    return _mid(sigma_tower_interval(name, Interval.from_value(t), 0)[0])


def delta_derivative_tower(name: str, t: float, ts: TimeScale) -> float:
    r"""Delta derivative of an omnibias activation, dispatched by the scale of ``t``.

    * ``reals`` -> the **closed-form** derivative tower ``sigma'(t)``;
    * ``h_integers`` -> the :mod:`omnibias.difference` forward difference
      ``(sigma(t+h) - sigma(t))/h``;
    * ``quantum`` -> the :mod:`omnibias.qcalculus` Jackson q-derivative
      ``(sigma(qt) - sigma(t))/((q-1)t)``.

    All three agree in the ``mu -> 0`` limit (the founding collapse to ``sigma'``).
    """
    if ts.kind == "reals":
        return _mid(sigma_tower_interval(name, Interval.from_value(t), 1)[1])
    if ts.kind == "h_integers":
        # dispatch to the founding forward-difference stencil (order 1, forward).
        return float(finite_difference_estimate(name, t, 1, ts.h, "forward").estimate)
    if ts.kind == "quantum":
        if t == 0.0:  # right-dense at 0: fall back to the closed-form derivative
            return _mid(sigma_tower_interval(name, Interval.from_value(0.0), 1)[1])
        return float(_jackson_q_derivative(lambda x: sigma_value(name, x), t, ts.q))
    # finite / general: the plain scattered quotient of the closed-form value.
    if ts.is_right_scattered(t):
        return float((sigma_value(name, ts.sigma(t)) - sigma_value(name, t)) / ts.mu(t))
    return _mid(sigma_tower_interval(name, Interval.from_value(t), 1)[1])


__all__ = [
    "delta_derivative",
    "delta_derivative_tower",
    "nabla_derivative",
    "sigma_value",
]
