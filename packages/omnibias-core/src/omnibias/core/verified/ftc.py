# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous Fundamental Theorem of Calculus certificate (verified register).

This module seals the integral<->derivative link of the closed-form derivative
tower as a tamper-evident v1 certificate.  For any verified-tower activation it
certifies

.. math::

    \int_a^b \sigma^{(k)}(z)\,dz \;=\; \sigma^{(k-1)}(b) - \sigma^{(k-1)}(a)
    \qquad (k \ge 1),

the default ``k = 1`` being the textbook ``\int_a^b \sigma' = \sigma(b) -
\sigma(a)``.  The two sides are computed **independently** -- the left by
integrating a rigorous :class:`~omnibias.core.verified.taylor_model.TaylorModel`
of ``\sigma^{(k)}`` over the cell, the right by evaluating the pointwise
:func:`~omnibias.core.verified.sigma.sigma_tower_interval` at the two endpoints --
so the residual ``LHS - RHS`` enclosing ``0`` is a genuine cross-check, not a
tautology.  Everything is interval arithmetic: honest, rigorous, and never a
open-problem claim.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import TYPE_CHECKING

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.core.verified.taylor_model import TaylorModel

if TYPE_CHECKING:  # pragma: no cover - avoids a proof<->verified import cycle
    from omnibias.core.proof.certificate import Cert

_DEFAULT_ORDER = 6


def _sigma_taylor_model(
    name: str, k: int, center: float, radius: float, order: int
) -> TaylorModel:
    r"""Rigorous degree-``order`` Taylor model of ``sigma^(k)`` on the cell.

    The cell is ``[center - radius, center + radius]``.  The coefficients are the
    exact Taylor coefficients of ``sigma^(k)`` about ``center`` -- a *point* tower
    evaluation, hence tight,

    .. math::

        \text{coeffs}[j] = \frac{\sigma^{(k+j)}(\text{center})}{j!}
        \qquad (j = 0 \dots \text{order}),

    and the (flat) remainder is the Lagrange term with the ``(order+1)``-st tower
    entry enclosed over the *whole* cell,

    .. math::

        R \in \frac{\sigma^{(k+\text{order}+1)}([\text{cell}])}{(\text{order}+1)!}
        \,[-h, h]^{\text{order}+1}
        \qquad (h = \text{radius}),

    which rigorously absorbs the truncated tail for every ``x`` in the cell.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    center_tower = sigma_tower_interval(name, Interval.point(center), k + order)
    coeffs = [
        center_tower[k + j] * Interval.from_rational(Fraction(1, factorial(j)))
        for j in range(order + 1)
    ]
    cell = Interval.point(center) + Interval(-radius, radius)
    top = sigma_tower_interval(name, cell, k + order + 1)[k + order + 1]
    rel = Interval(-radius, radius)
    remainder = (
        top
        * Interval.from_rational(Fraction(1, factorial(order + 1)))
        * rel.pow_int(order + 1)
    )
    return TaylorModel(center, radius, coeffs, remainder)


def _ftc_parts(
    name: str, a: float, b: float, *, k: int, order: int
) -> tuple[Interval, Interval, Interval]:
    """Shared rigorous ``(LHS, RHS, residual)`` for the FTC identity on ``[a, b]``."""
    if k < 1:
        raise ValueError(
            f"k must be >= 1 (the endpoints evaluate sigma^(k-1)), got {k}"
        )
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if not b > a:
        raise ValueError(f"require a < b, got a={a!r}, b={b!r}")
    center = 0.5 * (a + b)
    radius = 0.5 * (b - a)
    model = _sigma_taylor_model(name, k, center, radius, order)
    lhs = model.definite_integral()  # rigorous cell integral of sigma^(k)
    # Endpoints of the model's cell, enclosed rigorously (outward rounded).
    a_iv = Interval.point(center) - Interval.point(radius)
    b_iv = Interval.point(center) + Interval.point(radius)
    rhs = (
        sigma_tower_interval(name, b_iv, k - 1)[k - 1]
        - sigma_tower_interval(name, a_iv, k - 1)[k - 1]
    )
    return lhs, rhs, lhs - rhs


def certified_ftc_residual(
    name: str, a: float, b: float, *, k: int = 1, order: int = _DEFAULT_ORDER
) -> Interval:
    r"""Rigorous enclosure of ``\int_a^b sigma^(k) - (sigma^(k-1)(b) - sigma^(k-1)(a))``.

    A sound FTC identity forces the true residual to be exactly ``0``, so a correct
    enclosure must satisfy ``residual.contains(0.0)``.  Because the two terms are
    computed by independent rigorous methods (Taylor-model integration vs pointwise
    endpoint towers), a tight residual around ``0`` is a real cross-check of the
    derivative tower.

    Parameters
    ----------
    name
        A verified-tower activation (``tanh``/``sigmoid``/``gaussian``/``sin``/
        ``cos``/``silu``/``gelu``/``softplus``).
    a, b
        Integration limits with ``a < b``.
    k
        Derivative order of the integrand ``sigma^(k)`` (default ``1``); must be
        ``>= 1`` so the endpoints evaluate ``sigma^(k-1)``.
    order
        Taylor-model degree used to enclose ``sigma^(k)`` over the cell.
    """
    return _ftc_parts(name, a, b, k=k, order=order)[2]


def ftc_certificate(
    name: str, a: float, b: float, *, k: int = 1, order: int = _DEFAULT_ORDER
) -> Cert:
    r"""Seal the FTC identity ``\int_a^b sigma^(k) = sigma^(k-1)(b) - sigma^(k-1)(a)``.

    Returns a canonical, hash-sealed v1 certificate carrying the integral
    enclosure (``LHS``), the endpoint difference (``RHS``), and their residual
    (which contains ``0`` for a sound tower).  The certificate makes no open-problem
    claim; it is a rigorous, tamper-evident interval artifact.
    """
    # Imported lazily: certificate.py imports verified.interval, so a top-level
    # import here would create a proof<->verified initialisation cycle.
    from omnibias.core.proof.certificate import encode_interval, make_certificate

    lhs, rhs, residual = _ftc_parts(name, a, b, k=k, order=order)
    payload = {
        "type": "ftc_identity",
        "activation": name,
        "k": k,
        "order": order,
        "a": float(a),
        "b": float(b),
        "integral_of_kth_derivative": encode_interval(lhs),
        "endpoint_difference": encode_interval(rhs),
        "residual": encode_interval(residual),
        "residual_contains_zero": residual.contains(0.0),
    }
    return make_certificate(
        claim="FTC: integral of sigma^(k) = sigma^(k-1)(b) - sigma^(k-1)(a)",
        payload=payload,
        honesty={
            "unproven_claim": False,
            "closed_form_integrand": True,
            "rigorous_enclosure": True,
        },
    )


__all__ = [
    "certified_ftc_residual",
    "ftc_certificate",
]
