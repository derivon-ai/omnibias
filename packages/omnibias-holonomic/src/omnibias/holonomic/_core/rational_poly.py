# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact univariate polynomial arithmetic over the rationals.

A polynomial is an ascending tuple of :class:`~fractions.Fraction` coefficients
``(c_0, c_1, ..., c_d)`` meaning ``c_0 + c_1 x + ... + c_d x^d`` (the empty tuple is the
zero polynomial). Everything here is exact -- addition, multiplication, Euclidean division,
GCD, the shift ``p(x + c)``, evaluation, and the **dispersion set** used by Gosper's
algorithm. No floats, no rounding.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

Poly = tuple[Fraction, ...]
Rational = Fraction | int


def to_poly(coeffs: Sequence[Rational]) -> Poly:
    """Normalise a coefficient sequence into a trimmed rational polynomial."""
    out = [c if isinstance(c, Fraction) else Fraction(c) for c in coeffs]
    return _trim(out)


def _trim(coeffs: list[Fraction]) -> Poly:
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    return tuple(coeffs)


def degree(p: Poly) -> int:
    """Degree of ``p`` (``-1`` for the zero polynomial)."""
    return len(p) - 1


def is_zero(p: Poly) -> bool:
    return len(p) == 0


def padd(a: Poly, b: Poly) -> Poly:
    n = max(len(a), len(b))
    out = [Fraction(0)] * n
    for i, c in enumerate(a):
        out[i] += c
    for i, c in enumerate(b):
        out[i] += c
    return _trim(out)


def psub(a: Poly, b: Poly) -> Poly:
    return padd(a, pscale(b, Fraction(-1)))


def pscale(p: Poly, s: Rational) -> Poly:
    s = s if isinstance(s, Fraction) else Fraction(s)
    if s == 0:
        return ()
    return tuple(c * s for c in p)


def pmul(a: Poly, b: Poly) -> Poly:
    if not a or not b:
        return ()
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            out[i + j] += ca * cb
    return _trim(out)


def peval(p: Poly, x: Rational) -> Fraction:
    x = x if isinstance(x, Fraction) else Fraction(x)
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * x + c
    return acc


def pderiv(p: Poly) -> Poly:
    return _trim([c * i for i, c in enumerate(p)][1:]) if len(p) > 1 else ()


def pshift(p: Poly, c: Rational) -> Poly:
    """Compose ``p(x + c)`` (Horner in the shifted variable)."""
    c = c if isinstance(c, Fraction) else Fraction(c)
    result: Poly = ()
    for coeff in reversed(p):
        result = padd(pmul(result, (c, Fraction(1))), (coeff,))
    return result


def pdivmod(a: Poly, b: Poly) -> tuple[Poly, Poly]:
    """Euclidean division ``a = q b + r`` with ``deg r < deg b`` (exact, ``b != 0``)."""
    if is_zero(b):
        raise ZeroDivisionError("polynomial division by zero")
    a_coeffs = list(a)
    q = [Fraction(0)] * max(0, len(a) - len(b) + 1)
    b_lead = b[-1]
    while len(a_coeffs) >= len(b):
        deg_diff = len(a_coeffs) - len(b)
        factor = a_coeffs[-1] / b_lead
        q[deg_diff] = factor
        for i, cb in enumerate(b):
            a_coeffs[deg_diff + i] -= factor * cb
        a_coeffs = _trim_list(a_coeffs)
    return _trim(q), tuple(a_coeffs)


def _trim_list(coeffs: list[Fraction]) -> list[Fraction]:
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs


def pgcd(a: Poly, b: Poly) -> Poly:
    """Monic GCD of ``a`` and ``b`` (the zero polynomial for ``gcd(0, 0)``)."""
    while not is_zero(b):
        _, r = pdivmod(a, b)
        a, b = b, r
    return pmonic(a)


def pmonic(p: Poly) -> Poly:
    if is_zero(p):
        return ()
    return pscale(p, Fraction(1) / p[-1])


def dispersion_set(a: Poly, b: Poly, *, bound: int = 64) -> list[int]:
    r"""Non-negative integers ``j`` with ``deg gcd(a(x), b(x + j)) > 0`` (the dispersion set).

    Used by Gosper's algorithm to build the ``a/b/c`` normal form. Bounded search over
    ``0 <= j <= bound`` (ample for the polynomial degrees that arise in practice).
    """
    if is_zero(a) or is_zero(b):
        return []
    out: list[int] = []
    for j in range(bound + 1):
        if degree(pgcd(a, pshift(b, j))) > 0:
            out.append(j)
    return out


__all__ = [
    "Poly",
    "degree",
    "dispersion_set",
    "is_zero",
    "padd",
    "pderiv",
    "pdivmod",
    "peval",
    "pgcd",
    "pmonic",
    "pmul",
    "pscale",
    "pshift",
    "psub",
    "to_poly",
]
