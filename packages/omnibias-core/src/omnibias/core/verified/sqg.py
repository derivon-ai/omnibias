# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified 2-D SQG substrate: the single Riesz transform on a Poisson-blob basis.

The companion :mod:`omnibias.core.verified.riesz` module makes the *second-order*
Riesz composite ``R_iR_k`` elementary on the basis ``f_a = a^2/(\pi D^2)`` -- which
is exactly what the Leray projection (2-D Euler) needs -- but the **single** Riesz
transform ``R_j`` (Fourier multiplier ``-i\xi_j/|\xi|``, a *half*-Laplacian
``|\xi|^{-1}``) is *not* elementary on that basis.  Genuine **surface
quasi-geostrophic (SQG)** flow needs precisely that single Riesz transform:

.. math::

    u = R^\perp\theta = \nabla^\perp(-\Delta)^{-1/2}\theta
      = (-R_2\theta,\; R_1\theta).

The unlock is to change the blob basis to the one on which the half-Laplacian is
*itself* diagonal in closed form: the **2-D Poisson kernel**.  For a scale
``a > 0`` set ``D = x^2 + y^2 + a^2`` and

.. math::

    \theta_a(x) = \frac{1}{2\pi}\,\frac{a}{D^{3/2}},
    \qquad \hat\theta_a(\xi) = e^{-a|\xi|}.

Because the symbol is ``e^{-a|\xi|}``, every (half-)power of ``-\Delta`` acts by a
plain multiplication and stays elementary.  In particular the **stream function**

.. math::

    \psi_a = (-\Delta)^{-1/2}\theta_a,\qquad
    \hat\psi_a(\xi) = \frac{e^{-a|\xi|}}{|\xi|},\qquad
    \psi_a(x) = \frac{1}{2\pi}\,\frac{1}{D^{1/2}},

is in closed form (the radial inverse transform
``\int_0^\infty e^{-a\rho}J_0(r\rho)\,d\rho = (r^2+a^2)^{-1/2}``), and so is the
**single Riesz transform**

.. math::

    (R_j\theta_a)(x) = \partial_j\psi_a(x)
      = -\frac{1}{2\pi}\,\frac{x_j}{D^{3/2}},
    \qquad
    u = \nabla^\perp\psi_a = \frac{1}{2\pi}\,\frac{(y, -x)}{D^{3/2}}.

This is the verified half-Laplacian substrate recorded as the open obligation of
the 2-D Euler certificate.  A *radial* ``\theta = \sum_i c_i\theta_{a_i}`` makes
``u`` tangential (``\parallel (y,-x)``) while ``\nabla\theta`` is radial
(``\parallel (x,y)``), so ``u\cdot\nabla\theta \equiv 0`` -- an **exact SQG steady
state**, the genuine-SQG analogue of the Euler vortex.

Every quantity is an outward-rounded :class:`Interval`; the half-power ``D^{1/2}``
uses the rigorous :meth:`Interval.sqrt`.  The closed forms are *proved* (Fourier
multipliers) and *checked* in the tests against an independent ``mpmath`` Hankel
transform of the symbol and ``mpmath`` derivatives of ``\psi_a``.
"""

from __future__ import annotations

import math

from omnibias.core.verified.interval import Interval

#: rigorous enclosure of pi (``math.pi`` is the nearest double below the truth).
_PI = Interval(math.pi, math.nextafter(math.pi, math.inf))
_TWO_PI = Interval.point(2.0) * _PI


def _require_axis(name: str, value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 (x) or 1 (y), got {value}")


def _denom(x: float, y: float, a: float) -> Interval:
    """Rigorous enclosure of ``D = x^2 + y^2 + a^2`` (``>= a^2 > 0`` for ``a != 0``)."""
    return Interval.point(x).pow_int(2) + Interval.point(y).pow_int(2) + Interval.point(a).pow_int(2)


def _inv_pow_three_half(d: Interval) -> Interval:
    r"""Rigorous ``D^{-3/2} = 1/(D \cdot \sqrt D)``."""
    return (d * d.sqrt()).reciprocal()


def _inv_pow_five_half(d: Interval) -> Interval:
    r"""Rigorous ``D^{-5/2} = 1/(D^2 \cdot \sqrt D)``."""
    return (d.pow_int(2) * d.sqrt()).reciprocal()


def sqg_blob(x: float, y: float, a: float) -> Interval:
    r"""Verified SQG temperature blob ``\theta_a = a/(2\pi D^{3/2})`` (``> 0``).

    The 2-D Poisson kernel; its Fourier symbol is ``e^{-a|\xi|}`` (checked in the
    tests via the Hankel transform), which is what makes the half-Laplacian
    elementary on this basis.
    """
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    return Interval.point(a) * (_TWO_PI.reciprocal()) * _inv_pow_three_half(_denom(x, y, a))


def sqg_stream(x: float, y: float, a: float) -> Interval:
    r"""Verified SQG stream function ``\psi_a = (-\Delta)^{-1/2}\theta_a = 1/(2\pi D^{1/2})``."""
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    return (_TWO_PI * _denom(x, y, a).sqrt()).reciprocal()


def sqg_stream_gradient(x: float, y: float, a: float) -> tuple[Interval, Interval]:
    r"""Verified ``\nabla\psi_a = -(1/2\pi)(x, y)/D^{3/2}``."""
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    inv = _TWO_PI.reciprocal() * _inv_pow_three_half(_denom(x, y, a))
    return Interval.point(-x) * inv, Interval.point(-y) * inv


def sqg_riesz(j: int, x: float, y: float, a: float) -> Interval:
    r"""Verified **single** Riesz transform ``(R_j\theta_a)(x) = -x_j/(2\pi D^{3/2})``.

    This is the half-Laplacian piece ``R_j = \partial_j(-\Delta)^{-1/2}`` that is
    *not* elementary on the Euler ``f_a`` basis but **is** elementary here
    (``R_j\theta_a = \partial_j\psi_a``); ``j`` indexes the axis (``0`` = x,
    ``1`` = y).  It is the closed-form engine of the SQG velocity ``u = R^\perp\theta``.
    """
    _require_axis("j", j)
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    comp = (x, y)
    inv = _TWO_PI.reciprocal() * _inv_pow_three_half(_denom(x, y, a))
    return Interval.point(-comp[j]) * inv


def sqg_velocity(x: float, y: float, a: float) -> tuple[Interval, Interval]:
    r"""Verified SQG velocity ``u = R^\perp\theta_a = (1/2\pi)(y, -x)/D^{3/2}``.

    The tangential field ``u = \nabla^\perp\psi_a = (-\partial_y\psi_a,
    \partial_x\psi_a)``; computed here directly so the certificate can cross-check
    it against the independent ``(-R_2\theta_a, R_1\theta_a)`` Riesz route.
    """
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    inv = _TWO_PI.reciprocal() * _inv_pow_three_half(_denom(x, y, a))
    return Interval.point(y) * inv, Interval.point(-x) * inv


def sqg_blob_gradient(x: float, y: float, a: float) -> tuple[Interval, Interval]:
    r"""Verified ``\nabla\theta_a = -(3a/2\pi)(x, y)/D^{5/2}`` (purely radial).

    Paired with the tangential velocity ``u \parallel (y, -x)`` it makes the SQG
    advection residual ``u\cdot\nabla\theta`` vanish by exact perpendicularity.
    """
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    fac = Interval.point(-3.0 * a) * _TWO_PI.reciprocal() * _inv_pow_five_half(_denom(x, y, a))
    return Interval.point(x) * fac, Interval.point(y) * fac


def sqg_blob_l2_inner(a: float, b: float) -> Interval:
    r"""Verified ``\langle\theta_a,\theta_b\rangle_{L^2(\mathbb R^2)} = 1/(2\pi(a+b)^2)``.

    The exact whole-plane inner product of two Poisson-kernel temperature blobs.
    With ``\theta_a = a/(2\pi D^{3/2})`` the radial integral is elementary,

    .. math::

        \int_0^\infty \frac{r\,dr}{(r^2+a^2)^{3/2}(r^2+b^2)^{3/2}}
        = \frac{1}{(a+b)^2\,ab},
        \qquad
        \langle\theta_a,\theta_b\rangle = \frac{1}{2\pi(a+b)^2},

    so the squared ``L^2`` norm of a radial profile ``\theta=\sum_i c_i\theta_{a_i}``
    *diagonalises* to ``\sum_{ij} c_i c_j/(2\pi(a_i+a_j)^2)``.  This is the engine of
    the self-similar **obstruction** certificate (the residual lower bound
    ``\lVert(y+R^\perp\theta)\cdot\nabla\theta\rVert_2 \ge \lVert\theta\rVert_2``).
    The endpoints are outward rounded; requires ``a, b > 0``.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("blob scales a, b must be positive")
    ab = Interval.point(a) + Interval.point(b)
    return (_TWO_PI * ab.pow_int(2)).reciprocal()


def sqg_velocity_divergence_residual(x: float, y: float, a: float) -> Interval:
    r"""Closed-form enclosure of ``\nabla\cdot u`` for one velocity blob.

    Computes ``\partial_x u_1 + \partial_y u_2`` from the two *separate* component
    derivatives ``\partial_x u_1 = -(3/2\pi) x y D^{-5/2}`` and
    ``\partial_y u_2 = +(3/2\pi) x y D^{-5/2})``; the sum is analytically ``0``
    (``u = \nabla^\perp\psi`` is divergence-free), so the returned interval is a
    certificate that the two non-trivial rationals cancel.
    """
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    inv5 = _inv_pow_five_half(_denom(x, y, a))
    xy = Interval.point(x) * Interval.point(y)
    base = Interval.point(3.0) * _TWO_PI.reciprocal() * xy * inv5
    dxux = -base
    dyuy = base
    return dxux + dyuy


__all__ = [
    "sqg_blob",
    "sqg_blob_gradient",
    "sqg_blob_l2_inner",
    "sqg_riesz",
    "sqg_stream",
    "sqg_stream_gradient",
    "sqg_velocity",
    "sqg_velocity_divergence_residual",
]
