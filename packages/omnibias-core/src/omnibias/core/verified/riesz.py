# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified 2-D Riesz transform & Leray projection on a closed-form blob basis.

The 1-D self-similar machinery in :mod:`omnibias.core.verified.line` works in the
basis where the line Hilbert transform acts *exactly* (the Poisson /
conjugate-Poisson pair).  Two-dimensional model equations (2-D Euler / SQG) need
the planar analogue: the **Riesz transforms** ``R_j`` (Fourier multiplier
``-i\xi_j/|\xi|``) and the **Leray projection**
``P = I - \nabla\Delta^{-1}\nabla\cdot`` (multiplier
``\delta_{ik} - \xi_i\xi_k/|\xi|^2``) that removes the divergence of a vector
field.

The trick that makes this *closed form* (hence amenable to interval arithmetic)
is to work on a basis of **radial blobs that are the Laplacian of an explicit
Newtonian potential**.  For a scale ``a > 0`` set ``D = x^2 + y^2 + a^2`` and

.. math::

    N_a(x) = \tfrac{1}{4\pi}\,\ln D
    \quad(\text{the smoothed Newtonian potential}),\qquad
    f_a = \Delta N_a = \frac{a^2}{\pi\,D^2}\ (\ge 0),

so ``N_a = \Delta^{-1} f_a`` is known in closed form.  The *second-order* Riesz
composite that the Leray projection is built from is then **elementary**:

.. math::

    R_i R_k = -\partial_i\partial_k\Delta^{-1}
    \;\Longrightarrow\;
    (R_i R_k f_a)(x) = -\partial_i\partial_k N_a(x)
    = \frac{1}{2\pi}\,\frac{2 x_i x_k - \delta_{ik} D}{D^2}.

(The *single* Riesz transform of a radial blob is not elementary -- it carries a
half-Laplacian ``|\xi|^{-1}`` -- but the Leray projection only ever needs the
composite ``R_iR_k``, which is.)  Two consequences are used as built-in checks:

* ``R_{11} f + R_{22} f = -f`` exactly (the multiplier identity
  ``(\xi_1^2+\xi_2^2)/|\xi|^2 = 1``);
* for a vector blob ``v = (c_1, c_2) f_a`` the Leray projection
  ``(Pv)_i = c_i f_a + \sum_k (R_iR_k f_a)\,c_k`` is **divergence-free by an
  algebraic cancellation**, so ``\nabla\cdot(Pv) \equiv 0`` -- here certified by a
  closed-form interval residual that encloses ``0``.

Every quantity is an outward-rounded :class:`Interval`; the only transcendental
is the rigorous ``ln`` (the potential) and ``exp``/``ln`` in the far-field tail
bound.  The closed forms are *proved* (Fourier multipliers) and *checked* in the
tests against an independent ``mpmath`` derivative of ``N_a``.
"""

from __future__ import annotations

import math

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv, ln_iv

#: rigorous enclosure of pi (``math.pi`` is the nearest double below the truth).
_PI = Interval(math.pi, math.nextafter(math.pi, math.inf))
_TWO_PI = Interval.point(2.0) * _PI
_FOUR_PI = Interval.point(4.0) * _PI


def _require_axis(name: str, value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 (x) or 1 (y), got {value}")


def _denom(x: float, y: float, a: float) -> Interval:
    """Rigorous enclosure of ``D = x^2 + y^2 + a^2`` (``>= a^2 > 0`` for ``a != 0``)."""
    return Interval.point(x).pow_int(2) + Interval.point(y).pow_int(2) + Interval.point(a).pow_int(2)


def blob(x: float, y: float, a: float) -> Interval:
    r"""Verified radial blob ``f_a = \Delta N_a = a^2 / (\pi D^2)`` (``>= 0``)."""
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    d = _denom(x, y, a)
    return Interval.point(a).pow_int(2) * (_PI * d.pow_int(2)).reciprocal()


def blob_gradient(x: float, y: float, a: float) -> tuple[Interval, Interval]:
    r"""Verified gradient of the radial blob ``\nabla f_a``.

    From ``f_a = a^2/(\pi D^2)`` with ``D = x^2+y^2+a^2`` one differentiates to the
    closed form

    .. math::

        \nabla f_a = -\frac{4 a^2}{\pi\,D^3}\,(x, y),

    a purely *radial* field (parallel to ``(x, y)``).  This is the
    ``\nabla\omega`` of a 2-D Euler vorticity blob; paired with the tangential
    Biot--Savart velocity ``u = \nabla^\perp N_a \parallel (-y, x)`` it makes the
    steady-state residual ``u\cdot\nabla\omega`` vanish by exact perpendicularity.
    """
    if a == 0.0:
        raise ValueError("blob scale a must be non-zero")
    d = _denom(x, y, a)
    fac = Interval.point(-4.0) * Interval.point(a).pow_int(2) * (_PI * d.pow_int(3)).reciprocal()
    return Interval.point(x) * fac, Interval.point(y) * fac


def newtonian_potential(x: float, y: float, a: float) -> Interval:
    r"""Verified ``N_a = \Delta^{-1} f_a = (1/4\pi)\ln(x^2+y^2+a^2)``."""
    return ln_iv(_denom(x, y, a)) * _FOUR_PI.reciprocal()


def potential_gradient(x: float, y: float, a: float) -> tuple[Interval, Interval]:
    r"""Verified ``\nabla N_a = (1/2\pi)(x, y)/D``."""
    d = _denom(x, y, a)
    inv = (_TWO_PI * d).reciprocal()
    return Interval.point(x) * inv, Interval.point(y) * inv


def riesz_double_blob(i: int, k: int, x: float, y: float, a: float) -> Interval:
    r"""Verified second-order Riesz composite ``(R_i R_k f_a)(x)``.

    Equals ``-\partial_i\partial_k N_a = (1/2\pi)(2 x_i x_k - \delta_{ik} D)/D^2``;
    ``i, k`` index the axes (``0`` = x, ``1`` = y).  This is the closed-form
    Calderon--Zygmund building block of the Leray projection.
    """
    _require_axis("i", i)
    _require_axis("k", k)
    comp = (x, y)
    d = _denom(x, y, a)
    num = Interval.point(2.0) * Interval.point(comp[i]) * Interval.point(comp[k])
    if i == k:
        num = num - d
    return num * (_TWO_PI * d.pow_int(2)).reciprocal()


def leray_blob_field(
    c1: float, c2: float, x: float, y: float, a: float
) -> tuple[Interval, Interval]:
    r"""Verified Leray projection of the vector blob ``v = (c_1, c_2) f_a``.

    ``(Pv)_i = c_i f_a + \sum_k (R_iR_k f_a) c_k`` collapses to the closed forms

    .. math::

        (Pv)_1 = \frac{c_1(a^2+x^2-y^2) + 2 c_2 x y}{2\pi D^2},\qquad
        (Pv)_2 = \frac{c_2(a^2+y^2-x^2) + 2 c_1 x y}{2\pi D^2},

    which are divergence-free (see :func:`leray_divergence_residual`).
    """
    d = _denom(x, y, a)
    inv = (_TWO_PI * d.pow_int(2)).reciprocal()
    a2 = Interval.point(a).pow_int(2)
    x2 = Interval.point(x).pow_int(2)
    y2 = Interval.point(y).pow_int(2)
    xy = Interval.point(x) * Interval.point(y)
    c1i, c2i = Interval.point(c1), Interval.point(c2)
    pv1 = (c1i * (a2 + x2 - y2) + Interval.point(2.0) * c2i * xy) * inv
    pv2 = (c2i * (a2 + y2 - x2) + Interval.point(2.0) * c1i * xy) * inv
    return pv1, pv2


def leray_divergence_residual(
    c1: float, c2: float, x: float, y: float, a: float
) -> Interval:
    r"""Closed-form enclosure of ``\nabla\cdot(Pv)`` for the blob field above.

    Computes ``\partial_x (Pv)_1 + \partial_y (Pv)_2`` from the *separate*
    component derivatives (each a non-trivial rational function); the sum is
    analytically ``0``, so the returned interval is a certificate that the two
    independently-evaluated rationals cancel -- it must enclose ``0`` with a width
    reflecting only outward rounding.
    """
    d = _denom(x, y, a)
    inv3 = (_TWO_PI * d.pow_int(3)).reciprocal()
    a2 = Interval.point(a).pow_int(2)
    x2 = Interval.point(x).pow_int(2)
    y2 = Interval.point(y).pow_int(2)
    xi, yi = Interval.point(x), Interval.point(y)
    xy = xi * yi
    c1i, c2i = Interval.point(c1), Interval.point(c2)
    two = Interval.point(2.0)
    four = Interval.point(4.0)
    a1 = c1i * (a2 + x2 - y2) + two * c2i * xy  # numerator of 2pi D^2 (Pv)_1
    a2_num = c2i * (a2 + y2 - x2) + two * c1i * xy  # numerator of 2pi D^2 (Pv)_2
    grad_s = two * c1i * xi + two * c2i * yi  # d/dx and d/dy of the linear parts
    dx_pv1 = (grad_s * d - four * xi * a1) * inv3
    dy_pv2 = (grad_s * d - four * yi * a2_num) * inv3
    return dx_pv1 + dy_pv2


def riesz_tail_bound(
    kernel_const: float,
    decay_const: float,
    decay_power: float,
    x_trunc: float,
    core_radius: float,
) -> Interval:
    r"""Rigorous far-field bound on a 2-D Calderon--Zygmund (Riesz) convolution.

    The second-order Riesz kernel obeys ``|K(x)| <= C_K / |x|^2``.  If a profile
    ``g`` satisfies ``|g(t)| <= C |t|^{-p}`` for ``|t| >= X`` (``p > 0``) and the
    evaluation point has ``|x_0| <= \rho < X``, then the contribution of the
    region ``|t| >= X`` to ``(K * g)(x_0)`` is bounded by

    .. math::

        |\text{tail}| \le
        \frac{2\pi\,C_K\,C\,X^{-p}}{p\,(1 - \rho/X)^2},

    from ``|x_0 - t| \ge |t|(1 - \rho/X)`` and the planar polar integral
    ``\int_{|t|\ge X} |t|^{-p-2}\,dt = 2\pi X^{-p}/p``.  The returned symmetric
    interval ``[-B, B]`` is an outward-rounded enclosure of that bound, so a
    finite-basis Leray/Riesz evaluation can be upgraded to a rigorous full-plane
    statement by adding it.

    Parameters mirror the inequality: ``kernel_const = C_K`` (a bound on
    ``|K(x)|\,|x|^2``), ``decay_const = C``, ``decay_power = p``,
    ``x_trunc = X``, ``core_radius = \rho``.
    """
    if decay_power <= 0.0:
        raise ValueError("decay_power p must be > 0 for the tail to converge")
    if not (x_trunc > 0.0 and 0.0 <= core_radius < x_trunc):
        raise ValueError("require 0 <= core_radius < x_trunc and x_trunc > 0")
    if decay_const < 0.0 or kernel_const < 0.0:
        raise ValueError("decay_const C and kernel_const C_K must be non-negative")
    x_iv = Interval.point(x_trunc)
    x_neg_p = exp_iv(Interval.point(-decay_power) * ln_iv(x_iv))  # X^{-p}
    one_minus = Interval.point(1.0) - Interval.point(core_radius) * x_iv.reciprocal()
    denom = Interval.point(decay_power) * one_minus.pow_int(2)
    bound = (
        _TWO_PI
        * Interval.point(kernel_const)
        * Interval.point(decay_const)
        * x_neg_p
        * denom.reciprocal()
    )
    b = bound.hi
    return Interval(-b, b)


__all__ = [
    "blob",
    "blob_gradient",
    "leray_blob_field",
    "leray_divergence_residual",
    "newtonian_potential",
    "potential_gradient",
    "riesz_double_blob",
    "riesz_tail_bound",
]
