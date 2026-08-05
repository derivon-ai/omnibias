# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Scoped exact polynomial factorisation over ``Q`` (square-free + rational roots).

This is the factorisation substrate Petkovsek's ``Hyper`` (:mod:`.petkovsek`) needs to
enumerate the monic divisors of the trailing/leading coefficients. It is deliberately
**regime-scoped and honestly labelled**: it extracts the exact square-free structure (Yun's
algorithm) and every *rational-root / linear* factor, but does **not** factor irreducible
quadratic-or-higher parts (no Zassenhaus / van Hoeij). That covers the hypergeometric-term
solutions whose ratio is built from linear factors -- factorials, binomials, geometrics,
Pochhammer products -- which is the overwhelming majority; solutions requiring an
irreducible non-linear factor are out of scope and reported as such by the caller.
Everything here is exact.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import gcd, lcm

from omnibias.holonomic._core.rational_poly import (
    Poly,
    degree,
    is_zero,
    pderiv,
    pdivmod,
    peval,
    pgcd,
    pmonic,
    pmul,
    psub,
)


def _int_divisors(n: int) -> list[int]:
    n = abs(n)
    if n == 0:
        return [1]
    return [d for d in range(1, n + 1) if n % d == 0]


def rational_roots(p: Poly) -> list[Fraction]:
    """All distinct rational roots of ``p`` (exact rational-root enumeration)."""
    if is_zero(p):
        raise ValueError("the zero polynomial has every root")
    coeffs = list(p)
    k = 0
    while coeffs[k] == 0:
        k += 1
    roots: set[Fraction] = set()
    if k > 0:
        roots.add(Fraction(0))
    tail = tuple(coeffs[k:])  # constant term now non-zero
    if degree(tail) == 0:
        return sorted(roots)
    denom_lcm = 1
    for c in tail:
        denom_lcm = lcm(denom_lcm, c.denominator)
    ints = [int(c * denom_lcm) for c in tail]
    a0, am = ints[0], ints[-1]
    for pnum in _int_divisors(a0):
        for qden in _int_divisors(am):
            g = gcd(pnum, qden)
            for sign in (1, -1):
                cand = Fraction(sign * pnum // g, qden // g)
                if peval(tail, cand) == 0:
                    roots.add(cand)
    return sorted(roots)


def roots_with_multiplicity(p: Poly) -> list[tuple[Fraction, int]]:
    """Rational roots of ``p`` paired with their exact multiplicities."""
    out: list[tuple[Fraction, int]] = []
    cur = p
    for root in rational_roots(p):
        mult = 0
        linear = (-root, Fraction(1))  # (x - root)
        while degree(cur) > 0:
            q, r = pdivmod(cur, linear)
            if not is_zero(r):
                break
            cur = q
            mult += 1
        if mult:
            out.append((root, mult))
    return out


def square_free(p: Poly) -> list[tuple[Poly, int]]:
    r"""Yun's square-free decomposition: ``p = c prod_i g_i^i`` with ``g_i`` square-free.

    Returns the ``(g_i, i)`` with ``deg g_i > 0`` (monic), exact over ``Q``.
    """
    if is_zero(p):
        raise ValueError("cannot square-free-decompose the zero polynomial")
    f = pmonic(p)
    if degree(f) == 0:
        return []
    fp = pderiv(f)
    a = pgcd(f, fp)
    b, _ = pdivmod(f, a)
    c, _ = pdivmod(fp, a)
    d = psub(c, pderiv(b))
    out: list[tuple[Poly, int]] = []
    i = 1
    while degree(b) > 0:
        g = pgcd(b, d)
        if degree(g) > 0:
            out.append((pmonic(g), i))
        b, _ = pdivmod(b, g)
        c, _ = pdivmod(d, g)
        d = psub(c, pderiv(b))
        i += 1
    return out


def linear_factorization(p: Poly) -> tuple[list[tuple[Fraction, int]], Poly]:
    """``(roots_with_mult, cofactor)`` where ``cofactor`` has no rational roots (monic)."""
    roots = roots_with_multiplicity(p)
    cur = pmonic(p)
    for root, mult in roots:
        linear = (-root, Fraction(1))
        for _ in range(mult):
            cur, _ = pdivmod(cur, linear)
    return roots, pmonic(cur)


def monic_linear_divisors(p: Poly) -> list[Poly]:
    r"""Every monic product of the *rational-root linear factors* of ``p`` (incl. ``1``).

    These are the candidate ``a(n)`` / ``b(n)`` in Petkovsek's ``Hyper``. If ``p`` has an
    irreducible non-linear part it simply does not contribute divisors (scoped).
    """
    if is_zero(p):
        return [(Fraction(1),)]
    roots = roots_with_multiplicity(p)
    ranges = [range(mult + 1) for _root, mult in roots]
    divisors: list[Poly] = []
    for exponents in (product(*ranges) if roots else [()]):
        factor: Poly = (Fraction(1),)
        for (root, _mult), e in zip(roots, exponents, strict=True):
            linear = (-root, Fraction(1))
            for _ in range(e):
                factor = pmul(factor, linear)
        divisors.append(factor)
    unique: list[Poly] = []
    for d in divisors:
        if d not in unique:
            unique.append(d)
    return unique


__all__ = [
    "linear_factorization",
    "monic_linear_divisors",
    "rational_roots",
    "roots_with_multiplicity",
    "square_free",
]
