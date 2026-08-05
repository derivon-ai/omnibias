# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Coefficient-engine -> validated-dynamics bridge (the missing jet ingest).

Before this module ``omnibias-dynamics`` had **no** way to ingest a closed-form
derivative tower / Taylor jet: every ``(field, jac)`` pair had to be hand-written.
This bridge turns the closed-form activation tower directly into the pair the
QR-Lohner flow consumes:

* :func:`vector_field_from_sigma_tower` -- the scalar autonomous ODE
  ``x' = sigma(scale*x + bias)``, with the field built by composing the
  ``sigma`` derivative tower onto the state's time-Taylor series with the rigorous
  interval Faa di Bruno kernel (:func:`omnibias.core.verified.jet.compose_jet`),
  so a *single* tower evaluation drives every Taylor order of the step.
* :func:`sigma_oscillator_field` -- the nonlinear oscillator ``x' = y``,
  ``y' = -stiffness*sigma(x) - damping*y`` (a genuinely 2-D, rotation-like field
  where the QR-Lohner frame beats naive interval stepping on wrapping).
* :func:`discrete_periodic_point` -- the *discrete* analogue of
  :func:`~omnibias.dynamics.prove_periodic_orbit`: a Krawczyk proof of a
  period-``p`` orbit of a 1-D map ``g`` (e.g. a discovered recurrence or the
  logistic map), using the chain-rule / Newton-series derivative of the iterate.

Every enclosure is closed-form and rigorous; the flows are validated by the
existing Lohner engine and the discrete proof by the core Krawczyk test.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.jet import compose_jet
from omnibias.core.verified.kantorovich import krawczyk_certificate
from omnibias.core.verified.linalg import IntervalMatrix
from omnibias.core.verified.lohner import JacobianEnclosure
from omnibias.core.verified.ode import TaylorSeries, VectorField
from omnibias.core.verified.sigma import sigma_tower_interval

_ONE = Interval.point(1.0)
_ZERO = Interval.point(0.0)


def vector_field_from_sigma_tower(
    name: str, *, scale: IntervalLike = 1.0, bias: IntervalLike = 0.0
) -> tuple[VectorField, JacobianEnclosure]:
    r"""``(field, jac)`` for the scalar ODE ``x' = sigma(scale*x + bias)``.

    The ``field`` composes the closed-form ``sigma`` derivative tower onto the
    state's time-Taylor series via the rigorous interval Faa di Bruno kernel; the
    ``1x1`` Jacobian enclosure is ``scale * sigma'(scale*x + bias)``. ``name`` is
    any activation supported by
    :func:`~omnibias.core.verified.sigma.sigma_tower_interval`.
    """
    s = Interval.from_value(scale)
    b = Interval.from_value(bias)

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        x = series[0]
        # u(t) = scale*x(t) + bias -- affine in the time-Taylor series.
        u = [s * c for c in x.coeffs]
        u[0] = u[0] + b
        tower = sigma_tower_interval(name, u[0], len(u) - 1)
        return [TaylorSeries(compose_jet(u, list(tower)))]

    def jac(box: Sequence[Interval]) -> IntervalMatrix:
        u0 = s * box[0] + b
        return [[s * sigma_tower_interval(name, u0, 1)[1]]]

    return field, jac


def sigma_oscillator_field(
    name: str, *, stiffness: IntervalLike = 1.0, damping: IntervalLike = 0.0
) -> tuple[VectorField, JacobianEnclosure]:
    r"""``(field, jac)`` for ``x' = y``, ``y' = -stiffness*sigma(x) - damping*y``.

    A nonlinear (rotation-like) oscillator whose restoring force is the closed-form
    activation ``sigma`` -- e.g. ``name="tanh"`` gives a saturating spring. The
    ``sigma(x)`` term is built by the same rigorous tower composition as
    :func:`vector_field_from_sigma_tower`.
    """
    k = Interval.from_value(stiffness)
    c = Interval.from_value(damping)

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        x, y = series[0], series[1]
        u = list(x.coeffs)
        tower = sigma_tower_interval(name, u[0], len(u) - 1)
        sx = TaylorSeries(compose_jet(u, list(tower)))
        dy = sx * (-k) + y * (-c)
        return [y, dy]

    def jac(box: Sequence[Interval]) -> IntervalMatrix:
        sp = sigma_tower_interval(name, box[0], 1)[1]
        return [[_ZERO, _ONE], [(-k) * sp, -c]]

    return field, jac


@dataclass(frozen=True)
class DiscretePeriodicOrbit:
    """A verified period-``period`` point of a 1-D map (or a negative result)."""

    exists: bool
    period: int
    center: float
    enclosure: tuple[float, float] | None
    kappa: float | None


def discrete_periodic_point(
    g: Callable[[Interval], Interval],
    dg: Callable[[Interval], Interval],
    x_bar: float,
    period: int,
    *,
    radii: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2),
) -> DiscretePeriodicOrbit:
    r"""Prove a period-``period`` point of a 1-D map ``g`` via the Krawczyk test.

    Solves ``g^period(x) - x = 0`` with the interval map's iterate and its
    chain-rule derivative ``(g^p)'(x) = prod_i g'(x_i)`` -- the Newton-series
    derivative of the recurrence ``x_{n+1} = g(x_n)``. This is the discrete twin of
    :func:`~omnibias.dynamics.prove_periodic_orbit` (which proves *flow* periodic
    orbits): it certifies a unique period-``period`` point in a rigorously enclosed
    ball around ``x_bar``. ``g`` and ``dg`` must be sound interval extensions.
    Tries each ``radii`` (ascending), returning the first verified ball.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    def residual(box: list[Interval]) -> list[Interval]:
        x = box[0]
        for _ in range(period):
            x = g(x)
        return [x - box[0]]

    def jacobian(box: list[Interval]) -> list[list[Interval]]:
        x = box[0]
        deriv = _ONE
        for _ in range(period):
            deriv = dg(x) * deriv
            x = g(x)
        return [[deriv - _ONE]]

    # a_inv from the float chain-rule derivative of the iterate at x_bar.
    xf = Interval.point(x_bar)
    deriv_f = 1.0
    for _ in range(period):
        deriv_f *= dg(xf).mid
        xf = g(xf)
    slope = deriv_f - 1.0
    if slope == 0.0:
        return DiscretePeriodicOrbit(False, period, x_bar, None, None)
    a_inv = 1.0 / slope

    for r in radii:
        cert = krawczyk_certificate(residual, jacobian, [x_bar], [[a_inv]], r)
        if cert is not None:
            lo, hi = cert.enclosure[0]
            return DiscretePeriodicOrbit(True, period, x_bar, (lo, hi), cert.kappa)
    return DiscretePeriodicOrbit(False, period, x_bar, None, None)


__all__ = [
    "DiscretePeriodicOrbit",
    "discrete_periodic_point",
    "sigma_oscillator_field",
    "vector_field_from_sigma_tower",
]
