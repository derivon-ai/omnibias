# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Proper hypergeometric terms ``F(n, k)`` with exact bivariate term ratios.

A :class:`ProperTerm` is a product of factorials of integer-linear arguments (plus optional
geometric factors) -- the "proper hypergeometric" shape Zeilberger's theorem applies to:

.. math::

    F(n, k) = z_n^{\,n} z_k^{\,k} \prod_r \bigl[(a_r n + b_r k + c_r)!\bigr]^{e_r}.

Its two term ratios ``F(n+1,k)/F(n,k)`` and ``F(n,k+1)/F(n,k)`` are then **exact bivariate
rational functions** (:mod:`.poly2`) -- each factorial contributes a rising/falling product
of linear factors -- which is exactly what the creative-telescoping certificate solver
needs. Values are exact ``Fraction``s and vanish outside the finite ``k``-support (a
factorial of a negative argument), matching binomial conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import factorial

from omnibias.holonomic._core.poly2 import (
    Poly2,
    Rat2,
    p2_const,
    p2_linear,
    p2_mul,
    r2_mul,
    r2_one,
)

#: A factorial factor ``(a_n * n + a_k * k + c)!`` raised to ``exp`` (``exp`` usually +-1).
FactFactor = tuple[int, int, int, int]


@dataclass(frozen=True)
class ProperTerm:
    """A proper hypergeometric term (product of linear-argument factorials + geometrics)."""

    factors: tuple[FactFactor, ...]
    geom_n: Fraction = field(default=Fraction(1))
    geom_k: Fraction = field(default=Fraction(1))

    def value(self, n: int, k: int) -> Fraction:
        """Exact ``F(n, k)`` (zero outside the finite support)."""
        result = self.geom_n**n * self.geom_k**k
        for a_n, a_k, c, exp in self.factors:
            arg = a_n * n + a_k * k + c
            if arg < 0:
                return Fraction(0)
            result *= Fraction(factorial(arg)) ** exp
        return result

    def sum_over_k(self, n: int, lo: int, hi: int) -> Fraction:
        """Exact ``sum_{k=lo}^{hi} F(n, k)``."""
        return sum((self.value(n, k) for k in range(lo, hi + 1)), Fraction(0))

    def power(self, e: int) -> ProperTerm:
        """The termwise power ``F ** e`` (``e >= 1``)."""
        if e < 1:
            raise ValueError("power needs e >= 1")
        factors: list[FactFactor] = []
        for a_n, a_k, c, exp in self.factors:
            factors.append((a_n, a_k, c, exp * e))
        return ProperTerm(tuple(factors), self.geom_n**e, self.geom_k**e)

    def times(self, other: ProperTerm) -> ProperTerm:
        """The termwise product ``F * G``."""
        return ProperTerm(
            self.factors + other.factors,
            self.geom_n * other.geom_n,
            self.geom_k * other.geom_k,
        )

    def ratio_n(self) -> Rat2:
        """``F(n+1, k) / F(n, k)`` as an exact bivariate rational function."""
        ratio = r2_one()
        if self.geom_n != 1:
            ratio = r2_mul(ratio, (p2_const(self.geom_n), p2_const(1)))
        for a_n, a_k, c, exp in self.factors:
            ratio = r2_mul(ratio, _factor_ratio(a_n, a_k, c, exp, shift_in="n"))
        return ratio

    def ratio_k(self) -> Rat2:
        """``F(n, k+1) / F(n, k)`` as an exact bivariate rational function."""
        ratio = r2_one()
        if self.geom_k != 1:
            ratio = r2_mul(ratio, (p2_const(self.geom_k), p2_const(1)))
        for a_n, a_k, c, exp in self.factors:
            ratio = r2_mul(ratio, _factor_ratio(a_n, a_k, c, exp, shift_in="k"))
        return ratio


def _rising_product(base: Poly2, step: int) -> Rat2:
    """``(L + step)! / L!`` for the linear form ``base = L``; rising if step>0 else falling."""
    if step == 0:
        return r2_one()
    if step > 0:
        num = p2_const(1)
        for t in range(1, step + 1):
            num = p2_mul(num, _shift_const(base, t))
        return (num, p2_const(1))
    den = p2_const(1)
    for t in range(0, -step):
        den = p2_mul(den, _shift_const(base, -t))
    return (p2_const(1), den)


def _shift_const(base: Poly2, delta: int) -> Poly2:
    """``base + delta`` (add an integer constant to a linear form)."""
    out = dict(base)
    out[(0, 0)] = out.get((0, 0), Fraction(0)) + Fraction(delta)
    return {kk: v for kk, v in out.items() if v != 0}


def _factor_ratio(a_n: int, a_k: int, c: int, exp: int, *, shift_in: str) -> Rat2:
    """Ratio contribution of ``(a_n n + a_k k + c)! ** exp`` under a unit shift in n or k."""
    base = p2_linear(a_n, a_k, c)
    step = a_n if shift_in == "n" else a_k
    ratio = _rising_product(base, step)
    if exp < 0:
        ratio = (ratio[1], ratio[0])  # invert
        exp = -exp
    result = r2_one()
    for _ in range(exp):
        result = r2_mul(result, ratio)
    return result


def binomial_nk() -> ProperTerm:
    """The binomial coefficient ``C(n, k) = n! / (k! (n-k)!)``."""
    return ProperTerm(((1, 0, 0, 1), (0, 1, 0, -1), (1, -1, 0, -1)))


def geometric_k(base: Fraction | int) -> ProperTerm:
    """The geometric term ``base ** k``."""
    return ProperTerm((), geom_k=base if isinstance(base, Fraction) else Fraction(base))


__all__ = [
    "FactFactor",
    "ProperTerm",
    "binomial_nk",
    "geometric_k",
]
