# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified *line* (non-periodic) Hilbert transform on the Poisson basis.

The periodic Hilbert transform (:mod:`omnibias.core.verified.spectral`) lives on a
box; a self-similar blow-up profile lives on the whole line with **algebraic**
far-field decay.  This module supplies the line operator rigorously by working in
the basis where the line Hilbert transform acts *exactly* -- the Poisson /
conjugate-Poisson pair.  For a scale ``a > 0`` define

    p_a(x) = a / (x^2 + a^2)        (even, the Poisson kernel),
    q_a(x) = x / (x^2 + a^2)        (odd,  the conjugate Poisson kernel).

With the convention ``H[f](x) = (1/pi) p.v. \int f(t)/(x - t) dt`` the Hilbert
transform rotates the pair (a fact equivalent to ``hat{p_a}(xi) = pi e^{-a|xi|}``
and the multiplier ``-i sgn(xi)``):

    H[p_a] = q_a,      H[q_a] = -p_a.

So an odd profile ``Omega = sum_i c_i q_{a_i}`` has the **exact** closed forms

    Omega'    = sum_i c_i q_{a_i}',
    U' = H[Omega] = - sum_i c_i p_{a_i},
    U  = \int_0^x U'  = - sum_i c_i atan(x / a_i),

with no quadrature and no truncated tail -- the nonlocal operator is exact, and the
``|x|^{-1}`` decay of ``q_a`` is the physical self-similar far field.  Every value is
returned as an outward-rounded :class:`Interval`; the only transcendental is the
rigorous ``atan`` enclosure (the velocity primitive).

The rotation identities are *proved* (Fourier multiplier) and *checked* in the
tests against an independent high-precision principal-value quadrature.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import atan_iv, exp_iv, ln_iv

#: rigorous enclosure of pi (math.pi is the nearest double below the true value).
_PI = Interval(math.pi, math.nextafter(math.pi, math.inf))


def _denom(x: float, a: float) -> Interval:
    """Rigorous enclosure of ``x^2 + a^2`` (>= a^2 > 0 for a != 0)."""
    return Interval.point(x).pow_int(2) + Interval.point(a).pow_int(2)


def poisson_kernel(x: float, a: float) -> Interval:
    """Verified ``p_a(x) = a / (x^2 + a^2)`` (even)."""
    return Interval.point(a) * _denom(x, a).reciprocal()


def conjugate_poisson(x: float, a: float) -> Interval:
    """Verified ``q_a(x) = x / (x^2 + a^2)`` (odd); decays like ``1/x``."""
    return Interval.point(x) * _denom(x, a).reciprocal()


def conjugate_poisson_deriv(x: float, a: float) -> Interval:
    """Verified ``q_a'(x) = (a^2 - x^2) / (x^2 + a^2)^2`` (even)."""
    num = Interval.point(a).pow_int(2) - Interval.point(x).pow_int(2)
    return num * _denom(x, a).pow_int(2).reciprocal()


def poisson_kernel_deriv(x: float, a: float) -> Interval:
    """Verified ``p_a'(x) = -2 a x / (x^2 + a^2)^2`` (odd)."""
    num = Interval.point(-2.0) * Interval.point(a) * Interval.point(x)
    return num * _denom(x, a).pow_int(2).reciprocal()


def poisson_primitive(x: float, a: float) -> Interval:
    r"""Verified ``\int_0^x p_a = atan(x / a)`` (odd, bounded)."""
    ratio = Interval.point(x) * Interval.point(a).reciprocal()
    return atan_iv(ratio)


def hilbert_of_poisson(x: float, a: float) -> Interval:
    """Exact line Hilbert transform of the Poisson kernel: ``H[p_a] = q_a``."""
    return conjugate_poisson(x, a)


def hilbert_of_conjugate(x: float, a: float) -> Interval:
    """Exact line Hilbert transform of the conjugate kernel: ``H[q_a] = -p_a``."""
    return -poisson_kernel(x, a)


def hilbert_tail_bound(decay_const: float, decay_power: float, x_trunc: float,
                       core_radius: float) -> Interval:
    r"""Rigorous far-field bound on the line Hilbert transform of a decaying profile.

    Closes the *continuum tail* obligation for the nonlocal operator.  If a profile
    obeys ``|Omega(t)| <= C |t|^{-p}`` for ``|t| >= X`` (``p > 0``) and the evaluation
    point satisfies ``|x0| <= rho < X``, then the contribution of ``|t| >= X`` to
    ``H[Omega](x0) = (1/pi) p.v. int Omega(t)/(x0 - t) dt`` is bounded by

        |tail| <= (2 C X^{-p}) / (pi p (1 - rho/X)),

    derived from ``|x0 - t| >= |t|(1 - rho/X)`` and
    ``int_{|t|>=X} |t|^{-p-1} dt = 2 X^{-p}/p``.  The returned symmetric
    :class:`Interval` ``[-B, B]`` is an outward-rounded enclosure of that bound, so a
    finite-domain (or finite-basis) Hilbert evaluation can be upgraded to a rigorous
    full-line statement by adding it.

    Parameters mirror the inequality: ``decay_const = C``, ``decay_power = p``,
    ``x_trunc = X`` (truncation radius), ``core_radius = rho`` (max ``|x0|``).
    """
    if decay_power <= 0.0:
        raise ValueError("decay_power p must be > 0 for the tail to converge")
    if not (x_trunc > 0.0 and 0.0 <= core_radius < x_trunc):
        raise ValueError("require 0 <= core_radius < x_trunc and x_trunc > 0")
    if decay_const < 0.0:
        raise ValueError("decay_const C must be non-negative")
    x_iv = Interval.point(x_trunc)
    x_neg_p = exp_iv(Interval.point(-decay_power) * ln_iv(x_iv))  # X^{-p}
    one_minus = Interval.point(1.0) - Interval.point(core_radius) * x_iv.reciprocal()
    denom = _PI * Interval.point(decay_power) * one_minus
    bound = Interval.point(2.0) * Interval.point(decay_const) * x_neg_p * denom.reciprocal()
    b = bound.hi
    return Interval(-b, b)


def _matrix(fn: object, x_nodes: Sequence[float], a_values: Sequence[float],
            ) -> list[list[Interval]]:
    f = fn  # narrow for mypy
    assert callable(f)
    return [[f(float(x), float(a)) for a in a_values] for x in x_nodes]


def conjugate_poisson_matrix(x_nodes: Sequence[float],
                             a_values: Sequence[float]) -> list[list[Interval]]:
    """``[[q_{a_i}(x_j)]]`` -- the odd profile design matrix."""
    return _matrix(conjugate_poisson, x_nodes, a_values)


def conjugate_poisson_deriv_matrix(x_nodes: Sequence[float],
                                   a_values: Sequence[float]) -> list[list[Interval]]:
    """``[[q_{a_i}'(x_j)]]`` -- the profile-derivative design matrix."""
    return _matrix(conjugate_poisson_deriv, x_nodes, a_values)


def poisson_matrix(x_nodes: Sequence[float],
                   a_values: Sequence[float]) -> list[list[Interval]]:
    """``[[p_{a_i}(x_j)]]`` -- appears in ``U' = H[Omega] = -sum c_i p_{a_i}``."""
    return _matrix(poisson_kernel, x_nodes, a_values)


def poisson_primitive_matrix(x_nodes: Sequence[float],
                             a_values: Sequence[float]) -> list[list[Interval]]:
    """``[[atan(x_j / a_i)]]`` -- the velocity primitive ``U = -sum c_i atan(x/a_i)``."""
    return _matrix(poisson_primitive, x_nodes, a_values)


# --------------------------------------------------------------------------- #
# Even-profile layer (the CCF-on-the-line representation)                      #
# --------------------------------------------------------------------------- #
# A smooth even profile is represented as a finite even Poisson-basis sum
#   Theta(x) = sum_i c_i p_{a_i}(x)           (even, decays like |x|^{-2}),
# for which the line Hilbert transform is exact term-by-term (H[p_a] = q_a):
#   H[Theta](x)  = sum_i c_i q_{a_i}(x)       (odd),
#   Theta'(x)    = sum_i c_i p_{a_i}'(x)      (odd),
#   (H Theta)'(x)= H[Theta'](x) = sum_i c_i q_{a_i}'(x)  (even).
# These four are exactly the quantities the self-similar CCF residual needs:
#   E(Theta, lam) = (1+lam) y Theta' - lam Theta + s (H Theta) Theta'   (transport)
#                 + s Theta (H Theta)'                                   (flux extra).


def _profile_sum(
    kernel: object,
    x: float,
    coeffs: Sequence[float],
    scales: Sequence[float],
) -> Interval:
    f = kernel
    assert callable(f)
    if len(coeffs) != len(scales):
        raise ValueError("coeffs and scales must have the same length")
    if not coeffs:
        raise ValueError("need at least one (coeff, scale) term")
    if any(a == 0.0 for a in scales):
        raise ValueError("Poisson scales a must be non-zero")
    acc = Interval.point(0.0)
    for c, a in zip(coeffs, scales, strict=True):
        acc = acc + Interval.point(float(c)) * f(float(x), float(a))
    return acc


def even_profile(x: float, coeffs: Sequence[float], scales: Sequence[float]) -> Interval:
    r"""Verified even profile ``Theta(x) = sum_i c_i p_{a_i}(x)``."""
    return _profile_sum(poisson_kernel, x, coeffs, scales)


def even_profile_deriv(x: float, coeffs: Sequence[float], scales: Sequence[float]) -> Interval:
    r"""Verified ``Theta'(x) = sum_i c_i p_{a_i}'(x)`` (odd)."""
    return _profile_sum(poisson_kernel_deriv, x, coeffs, scales)


def hilbert_even_profile(x: float, coeffs: Sequence[float], scales: Sequence[float]) -> Interval:
    r"""Verified **exact** ``H[Theta](x) = sum_i c_i q_{a_i}(x)`` (odd)."""
    return _profile_sum(conjugate_poisson, x, coeffs, scales)


def hilbert_even_profile_deriv(
    x: float, coeffs: Sequence[float], scales: Sequence[float]
) -> Interval:
    r"""Verified **exact** ``(H Theta)'(x) = H[Theta'](x) = sum_i c_i q_{a_i}'(x)`` (even)."""
    return _profile_sum(conjugate_poisson_deriv, x, coeffs, scales)


def even_profile_tail_constant(
    coeffs: Sequence[float], scales: Sequence[float]
) -> tuple[float, float]:
    r"""Rigorous far-field decay ``(C, p)`` with ``|Theta(x)| <= C |x|^{-p}``.

    Since ``p_a(x) = a/(x^2+a^2) <= |a|/x^2``, the even profile obeys
    ``|Theta(x)| <= (sum_i |c_i| |a_i|) |x|^{-2}``; returns that outward-rounded
    constant and ``p = 2`` for feeding :func:`hilbert_tail_bound`.
    """
    if len(coeffs) != len(scales):
        raise ValueError("coeffs and scales must have the same length")
    acc = Interval.point(0.0)
    for c, a in zip(coeffs, scales, strict=True):
        acc = acc + Interval.point(float(c)).abs() * Interval.point(float(a)).abs()
    return acc.hi, 2.0


__all__ = [
    "conjugate_poisson",
    "conjugate_poisson_deriv",
    "conjugate_poisson_deriv_matrix",
    "conjugate_poisson_matrix",
    "even_profile",
    "even_profile_deriv",
    "even_profile_tail_constant",
    "hilbert_even_profile",
    "hilbert_even_profile_deriv",
    "hilbert_of_conjugate",
    "hilbert_of_poisson",
    "hilbert_tail_bound",
    "poisson_kernel",
    "poisson_kernel_deriv",
    "poisson_matrix",
    "poisson_primitive",
    "poisson_primitive_matrix",
]
