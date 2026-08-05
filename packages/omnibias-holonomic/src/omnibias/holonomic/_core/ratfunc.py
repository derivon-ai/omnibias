# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact univariate rational functions over the rationals ``Q(x)``.

A :data:`RatFunc` is a normalised pair ``(num, den)`` of :mod:`.rational_poly`
polynomials with ``den`` monic, ``gcd(num, den) = 1``, and the zero element carried as
``((), (1,))``. This is the coefficient field the Ore-operator Euclidean algorithm
(:mod:`.oreops`) and the exact linear-relation finder (:mod:`.relations`) work over --
polynomial coefficients are not a field, so right division of Ore operators needs genuine
``Q(x)`` arithmetic. Everything is exact; there is no rounding anywhere.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.holonomic._core.rational_poly import (
    Poly,
    is_zero,
    padd,
    pdivmod,
    pgcd,
    pmul,
    pshift,
    psub,
)

#: A rational function ``num / den`` (both ascending-coefficient polynomials).
RatFunc = tuple[Poly, Poly]

_ONE: Poly = (Fraction(1),)
_ZERO: RatFunc = ((), _ONE)


def rf_zero() -> RatFunc:
    """The zero rational function ``0/1``."""
    return _ZERO


def rf_normalize(num: Poly, den: Poly) -> RatFunc:
    """Reduce ``num/den`` to lowest terms with a monic denominator (exact)."""
    if is_zero(den):
        raise ZeroDivisionError("rational function with zero denominator")
    if is_zero(num):
        return _ZERO
    g = pgcd(num, den)
    if len(g) > 1:  # degree > 0
        num = pdivmod(num, g)[0]
        den = pdivmod(den, g)[0]
    lead = den[-1]
    if lead != 1:
        num = tuple(c / lead for c in num)
        den = tuple(c / lead for c in den)
    return (num, den)


def rf_from_poly(p: Poly) -> RatFunc:
    """Embed a polynomial as ``p/1``."""
    return rf_normalize(p, _ONE)


def rf_from_rational(v: Fraction | int) -> RatFunc:
    """Embed a scalar as ``v/1``."""
    f = v if isinstance(v, Fraction) else Fraction(v)
    return rf_normalize((f,) if f != 0 else (), _ONE)


def rf_is_zero(a: RatFunc) -> bool:
    """Whether ``a`` is the zero rational function."""
    return bool(is_zero(a[0]))


def rf_add(a: RatFunc, b: RatFunc) -> RatFunc:
    """Exact sum ``a + b``."""
    (an, ad), (bn, bd) = a, b
    return rf_normalize(padd(pmul(an, bd), pmul(bn, ad)), pmul(ad, bd))


def rf_sub(a: RatFunc, b: RatFunc) -> RatFunc:
    """Exact difference ``a - b``."""
    (an, ad), (bn, bd) = a, b
    return rf_normalize(psub(pmul(an, bd), pmul(bn, ad)), pmul(ad, bd))


def rf_mul(a: RatFunc, b: RatFunc) -> RatFunc:
    """Exact product ``a * b``."""
    (an, ad), (bn, bd) = a, b
    return rf_normalize(pmul(an, bn), pmul(ad, bd))


def rf_div(a: RatFunc, b: RatFunc) -> RatFunc:
    """Exact quotient ``a / b`` (``b != 0``)."""
    (an, ad), (bn, bd) = a, b
    if is_zero(bn):
        raise ZeroDivisionError("division by the zero rational function")
    return rf_normalize(pmul(an, bd), pmul(ad, bn))


def rf_neg(a: RatFunc) -> RatFunc:
    """Exact negation ``-a``."""
    num, den = a
    return (tuple(-c for c in num), den)


def rf_shift(a: RatFunc, c: Fraction | int) -> RatFunc:
    """Compose ``a(x + c)`` (used by the shift Ore algebra's ``sigma``)."""
    num, den = a
    return rf_normalize(pshift(num, c), pshift(den, c))


__all__ = [
    "RatFunc",
    "rf_add",
    "rf_div",
    "rf_from_poly",
    "rf_from_rational",
    "rf_is_zero",
    "rf_mul",
    "rf_neg",
    "rf_normalize",
    "rf_shift",
    "rf_sub",
    "rf_zero",
]
