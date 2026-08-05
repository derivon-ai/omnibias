# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous (interval) enclosures for the least-action quantities.

Pure Python -- no torch / jax. Every quantity is returned as a guaranteed
:class:`~omnibias.core.verified.interval.Interval` produced by outward-rounded
arithmetic, so it *provably* contains the true value over the stated scope.

- :func:`action_enclosure` -- a rigorous enclosure of the action
  :math:`\int_a^b L\,dt` from interval samples of the integrand plus a rigorous
  bound on its second derivative (the honest, fudge-factor-free composite
  midpoint / trapezoid remainder from :mod:`omnibias.core.verified.quadrature`).
- :func:`euler_lagrange_enclosure` -- the Euler-Lagrange residual
  :math:`m\,\ddot q + V'(q)` of a mechanical Lagrangian
  :math:`L = \tfrac12 m\dot q^2 - V(q)`, enclosed over a phase-space *box*.
- :func:`energy_enclosure` -- the mechanical energy
  :math:`\tfrac12 m\dot q^2 + V(q)` enclosed over a box.
- :func:`action_certificate` -- an optional hash-sealed v1 certificate around an
  enclosure (via :mod:`omnibias.core.proof.certificate`), honestly scoped.

Soundness, not completeness: pass rigorous inputs (interval samples, an interval
``second_derivative`` bound) and the outputs provably bracket the truth over the
declared ``scope`` -- a local box, never a global or global-regularity-grade claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.quadrature import midpoint_integral, trapezoid_integral

_HALF = Interval.from_rational(Fraction(1, 2))


def action_enclosure(
    integrand_values: Sequence[IntervalLike],
    a: float,
    b: float,
    *,
    second_derivative: IntervalLike,
    rule: str = "midpoint",
) -> Interval:
    r"""Rigorous enclosure of the action :math:`S = \int_a^b L\,dt`.

    Parameters
    ----------
    integrand_values
        Interval enclosures of the Lagrangian ``L(t)`` along the trajectory at
        equispaced nodes: the ``n`` panel *midpoints* for ``rule="midpoint"``, or
        the ``n + 1`` panel *endpoints* for ``rule="trapezoid"``.
    a, b
        Integration limits with ``a <= b``.
    second_derivative
        A rigorous interval enclosure of ``d^2 L / dt^2`` over all of ``[a, b]``;
        it supplies the derived (fudge-factor-free) quadrature remainder.
    rule
        ``"midpoint"`` (default) or ``"trapezoid"``.

    Returns
    -------
    Interval
        A guaranteed enclosure of the exact action.
    """
    vals = [Interval.from_value(v) for v in integrand_values]
    m2 = Interval.from_value(second_derivative)
    if rule == "midpoint":
        return midpoint_integral(vals, a, b, m2)
    if rule == "trapezoid":
        return trapezoid_integral(vals, a, b, m2)
    raise ValueError(f"rule must be 'midpoint' or 'trapezoid', got {rule!r}")


def euler_lagrange_enclosure(
    qddot: IntervalLike,
    potential_gradient: IntervalLike,
    *,
    mass: IntervalLike = 1,
) -> Interval:
    r"""Enclose the Euler-Lagrange residual ``m qddot + V'(q)`` over a box.

    For the mechanical Lagrangian :math:`L = \tfrac12 m\dot q^2 - V(q)` the
    Euler-Lagrange equation is :math:`m\ddot q + V'(q) = 0`. Passing interval
    enclosures of ``qddot`` and ``V'(q)`` over a phase-space box yields a
    guaranteed enclosure of the residual there (e.g. to certify no solution --
    a residual interval bounded away from zero -- inside the box).
    """
    m = Interval.from_value(mass)
    return m * Interval.from_value(qddot) + Interval.from_value(potential_gradient)


def energy_enclosure(
    qdot: IntervalLike,
    potential: IntervalLike,
    *,
    mass: IntervalLike = 1,
) -> Interval:
    r"""Enclose the mechanical energy ``1/2 m qdot^2 + V(q)`` over a box."""
    m = Interval.from_value(mass)
    return _HALF * m * Interval.from_value(qdot).pow_int(2) + Interval.from_value(potential)


def acceleration_enclosure(
    force: IntervalLike,
    *,
    mass: IntervalLike = 1,
) -> Interval:
    r"""Enclose the forward acceleration ``qddot = F / m`` of a scalar/const-mass Lagrangian.

    The forward-dynamics dual of :func:`euler_lagrange_enclosure`: for the
    mechanical Lagrangian :math:`L = \tfrac12 m\dot q^2 - V(q)` the equation of
    motion is :math:`m\ddot q = -V'(q)`, so passing an interval enclosure of the
    generalized force ``F`` (e.g. ``-V'(q)``) and the mass ``m`` (bounded away
    from zero) yields a guaranteed enclosure of the acceleration over the box.
    The general positive-definite matrix mass ``M(q)`` (a rigorous interval
    linear solve) is a noted follow-up.
    """
    return Interval.from_value(force) / Interval.from_value(mass)


def action_certificate(
    interval: Interval,
    *,
    claim: str = "action enclosure (least-action)",
    scope: str = "local_box",
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A sealed, tamper-evident v1 certificate asserting the action lies in ``interval``.

    The ``scope`` is recorded honestly in the certificate metadata (default
    ``"local_box"``): the enclosure is a local statement, never a global claim.
    Verify with
    :func:`omnibias.core.proof.certificate.verify_certificate_digest`.
    """
    from omnibias.core.proof.certificate import interval_certificate

    cert_meta: dict[str, Any] = {"scope": scope}
    if meta is not None:
        cert_meta.update(meta)
    return dict(interval_certificate(claim, interval, meta=cert_meta))


__all__ = [
    "acceleration_enclosure",
    "action_certificate",
    "action_enclosure",
    "energy_enclosure",
    "euler_lagrange_enclosure",
]
