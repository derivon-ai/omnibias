# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gosper's algorithm: indefinite hypergeometric summation with an exact certificate.

A term ``t(k)`` is *hypergeometric* when the ratio ``t(k+1)/t(k) = num(k)/den(k)`` is a
fixed rational function of ``k``. Gosper's algorithm decides whether ``t`` has a
hypergeometric antidifference ``T`` (``T(k+1) - T(k) = t(k)``) and, when it does, returns the
**rational certificate** ``R(k)`` with ``T(k) = R(k) t(k)`` -- so
``sum_{k=a}^{b-1} t(k) = R(b) t(b) - R(a) t(a)`` in closed form.

The construction is exact (Gosper-Petkovšek normal form + a bounded-degree polynomial
solve over the rationals); it returns ``summable=False`` rather than an unsound guess when
no hypergeometric antidifference exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.holonomic._core.linalg import solve_exact
from omnibias.holonomic._core.rational_poly import (
    Poly,
    degree,
    dispersion_set,
    pdivmod,
    peval,
    pgcd,
    pmul,
    pshift,
    psub,
    to_poly,
)


@dataclass(frozen=True)
class GosperResult:
    """The outcome of Gosper's algorithm on a hypergeometric ratio ``num/den``."""

    summable: bool
    #: certificate numerator/denominator: ``R(k) = cert_num(k) / cert_den(k)``, ``T = R t``.
    cert_num: Poly = ()
    cert_den: Poly = ()

    def certificate(self, k: Fraction | int) -> Fraction:
        """Evaluate the rational certificate ``R(k)`` (``T(k) = R(k) t(k)``)."""
        if not self.summable:
            raise ValueError("not Gosper-summable; no certificate")
        return Fraction(peval(self.cert_num, k) / peval(self.cert_den, k))


def gosper_normal_form(num: Poly, den: Poly) -> tuple[Poly, Poly, Poly]:
    r"""Gosper-Petkovšek form ``(a, b, c)`` with ``num/den = (a/b)(c(k+1)/c(k))``.

    Guarantees ``gcd(a(k), b(k + j)) = 1`` for every non-negative integer ``j``.
    """
    a, b = num, den
    g0 = pgcd(a, b)
    if degree(g0) > 0:
        a = pdivmod(a, g0)[0]
        b = pdivmod(b, g0)[0]
    c: Poly = (Fraction(1),)
    while True:
        shifts = [j for j in dispersion_set(a, b) if j >= 1]
        if not shifts:
            break
        j = shifts[0]
        g = pgcd(a, pshift(b, j))
        a = pdivmod(a, g)[0]
        b = pdivmod(b, pshift(g, -j))[0]
        for i in range(1, j + 1):
            c = pmul(c, pshift(g, -i))
    return a, b, c


def _degree_bound(bigA: Poly, bigB: Poly, bigC: Poly) -> int:
    dA, dB, dC = degree(bigA), degree(bigB), degree(bigC)
    if dA != dB:
        return int(dC - max(dA, dB))
    if dA < 0:  # both zero (cannot happen for valid input)
        return -1
    la, lb = bigA[-1], bigB[-1]
    if la != lb:
        return int(dC - dA)
    d = dA
    a_sub = bigA[d - 1] if d - 1 >= 0 else Fraction(0)
    b_sub = bigB[d - 1] if d - 1 >= 0 else Fraction(0)
    jstar = (b_sub - a_sub) / la
    cand = dC - (d - 1)
    if jstar.denominator == 1 and jstar >= 0:
        return int(max(int(jstar), cand))
    return int(cand)


def gosper_sum(num: Poly, den: Poly) -> GosperResult:
    r"""Run Gosper's algorithm on the term ratio ``t(k+1)/t(k) = num(k)/den(k)``.

    Returns a :class:`GosperResult`; ``summable`` is ``True`` iff a hypergeometric
    antidifference exists, in which case ``R(k) = cert_num/cert_den`` satisfies
    ``R(k+1) num(k)/den(k) ... `` giving ``T(k) = R(k) t(k)`` with ``T(k+1)-T(k)=t(k)``.
    """
    num = to_poly(num)
    den = to_poly(den)
    if degree(den) < 0:
        raise ValueError("den must be non-zero")
    a, b, c = gosper_normal_form(num, den)
    bigA = a
    bigB = pshift(b, -1)  # b(k - 1)
    bigC = c
    bound = _degree_bound(bigA, bigB, bigC)
    if bound < 0:
        return GosperResult(summable=False)

    # Solve A(k) x(k+1) - B(k) x(k) = C(k) for x of degree <= bound.
    cols: list[Poly] = []
    for i in range(bound + 1):
        xi = tuple(Fraction(1) if e == i else Fraction(0) for e in range(i + 1))
        contrib = psub(pmul(bigA, pshift(xi, 1)), pmul(bigB, xi))
        cols.append(contrib)
    max_deg = max((degree(col) for col in cols), default=-1)
    max_deg = max(max_deg, degree(bigC), 0)
    matrix = [[(col[p] if p < len(col) else Fraction(0)) for col in cols] for p in range(max_deg + 1)]
    rhs = [(bigC[p] if p < len(bigC) else Fraction(0)) for p in range(max_deg + 1)]
    sol = solve_exact(matrix, rhs)
    if sol is None:
        return GosperResult(summable=False)
    x = to_poly(sol)
    if degree(x) < 0:  # x == 0 only valid if C == 0
        if degree(bigC) >= 0:
            return GosperResult(summable=False)
    # certificate R(k) = b(k-1) x(k) / c(k)
    cert_num = pmul(pshift(b, -1), x)
    cert_den = c
    return GosperResult(summable=True, cert_num=cert_num, cert_den=cert_den)


__all__ = [
    "GosperResult",
    "gosper_normal_form",
    "gosper_sum",
]
