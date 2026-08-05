# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Convenience vector-field builders for the validated-dynamics engines.

Every engine in this package consumes the pair ``(field, jac)`` used by the
QR-Lohner flow: ``field`` is an autonomous :data:`~omnibias.core.verified.ode.VectorField`
acting on component Taylor series, and ``jac`` a
:data:`~omnibias.core.verified.lohner.JacobianEnclosure` mapping a box to an
interval enclosure of ``DF``.  These helpers build that pair for the systems used
in the documentation and tests; user systems supply their own pair directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import IntervalMatrix
from omnibias.core.verified.lohner import (
    JacobianEnclosure,
    constant_jacobian,
    linear_field,
)
from omnibias.core.verified.ode import TaylorSeries, VectorField


def linear_system(a: Sequence[Sequence[float]]) -> tuple[VectorField, JacobianEnclosure]:
    """The pair ``(field, jac)`` for the linear flow ``y' = A y``."""
    return linear_field(a), constant_jacobian(a)


def harmonic_oscillator(omega: float = 1.0) -> tuple[VectorField, JacobianEnclosure]:
    r"""The undamped oscillator ``x' = omega y, y' = -omega x`` (a pure rotation).

    The flow rotates by ``omega t``; the monodromy over one period ``T = 2 pi /
    omega`` is the identity (both Floquet multipliers ``1``).
    """
    return linear_system([[0.0, omega], [-omega, 0.0]])


def hopf_normal_form(
    mu: float = 1.0,
) -> tuple[VectorField, JacobianEnclosure]:
    r"""Supercritical Hopf normal form with an isolated limit cycle at ``r = sqrt(mu)``.

    .. math::

        x' = \mu x - y - x (x^2 + y^2), \qquad
        y' = x + \mu y - y (x^2 + y^2).

    For ``mu = 1`` the cycle is the unit circle with period ``2 pi`` and non-trivial
    Floquet multiplier ``exp(-4 pi)`` (radial determinant ``exp(\int tr\,DF) =
    exp(-4 pi)``).
    """

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        x, y = series[0], series[1]
        r2 = x * x + y * y
        dx = x * mu - y - x * r2
        dy = x + y * mu - y * r2
        return [dx, dy]

    def jac(box: Sequence[Interval]) -> IntervalMatrix:
        x, y = box[0], box[1]
        m = Interval.point(mu)
        three = Interval.point(3.0)
        two = Interval.point(2.0)
        x2, y2 = x.pow_int(2), y.pow_int(2)
        j00 = m - (three * x2 + y2)
        j01 = Interval.point(-1.0) - two * x * y
        j10 = Interval.point(1.0) - two * x * y
        j11 = m - (x2 + three * y2)
        return [[j00, j01], [j10, j11]]

    return field, jac


def radial_logistic(mu: float = 1.0) -> tuple[VectorField, JacobianEnclosure]:
    r"""Scalar radial reduction of the Hopf form: ``r' = mu r - r^3``.

    This is the Poincare return dynamics (return time ``2 pi``) of
    :func:`hopf_normal_form` on the section ``theta = 0``.  Its fixed point
    ``r = sqrt(mu)`` is the limit cycle; the time-``2 pi`` map has the *isolated*
    fixed point used by :func:`~omnibias.dynamics.prove_periodic_orbit`.
    """

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        r = series[0]
        return [r * mu - r * r * r]

    def jac(box: Sequence[Interval]) -> IntervalMatrix:
        r = box[0]
        return [[Interval.point(mu) - Interval.point(3.0) * r.pow_int(2)]]

    return field, jac


__all__ = [
    "harmonic_oscillator",
    "hopf_normal_form",
    "linear_system",
    "radial_logistic",
]
