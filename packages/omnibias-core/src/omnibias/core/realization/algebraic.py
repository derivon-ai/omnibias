# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Portable real algebraic values with exact Sturm root isolation.

The primitive polynomial need not be irreducible. Equality/sign refer to its
one root in the isolating interval. Division requires a unit modulo that
polynomial; callers can refine a reducible primitive representation first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from omnibias.core.realization.polynomial import Rational, rational

UPoly = tuple[Fraction, ...]


def trim(p: Sequence[Rational]) -> UPoly:
    out = list(map(rational, p))
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)


def add(a: UPoly, b: UPoly) -> UPoly:
    return trim(
        [
            (a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0))
            for i in range(max(len(a), len(b)))
        ]
    )


def scale(a: UPoly, c: Rational) -> UPoly:
    return trim([v * rational(c) for v in a])


def multiply(a: UPoly, b: UPoly) -> UPoly:
    if not a or not b:
        return ()
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return trim(out)


def divide(a: UPoly, b: UPoly) -> tuple[UPoly, UPoly]:
    if not b:
        raise ZeroDivisionError("zero polynomial divisor")
    rem = list(a)
    q = [Fraction(0)] * max(0, len(a) - len(b) + 1)
    while rem and len(rem) >= len(b):
        j = len(rem) - len(b)
        coeff = rem[-1] / b[-1]
        q[j] = coeff
        for i, c in enumerate(b):
            rem[i + j] -= coeff * c
        rem = list(trim(rem))
    return trim(q), trim(rem)


def gcd(a: UPoly, b: UPoly) -> UPoly:
    while b:
        a, b = b, divide(a, b)[1]
    return scale(a, 1 / a[-1]) if a else ()


def evaluate(p: UPoly, x: Rational) -> Fraction:
    v = Fraction(0)
    for c in reversed(p):
        v = v * rational(x) + c
    return v


def derivative(p: UPoly) -> UPoly:
    return trim([i * c for i, c in enumerate(p)][1:])


def sturm_sequence(p: Sequence[Rational]) -> tuple[UPoly, ...]:
    a = trim(p)
    if len(a) < 2:
        raise ValueError("root counting requires a nonconstant polynomial")
    b = derivative(a)
    seq = [a, b]
    while b:
        rem = scale(divide(a, b)[1], -1)
        if not rem:
            break
        seq.append(rem)
        a, b = b, rem
    return tuple(seq)


def root_count(p: Sequence[Rational], lo: Rational, hi: Rational) -> int:
    """Distinct roots strictly between non-root rational endpoints."""
    a, lower, upper = trim(p), rational(lo), rational(hi)
    if lower >= upper or not evaluate(a, lower) or not evaluate(a, upper):
        raise ValueError("ordered non-root endpoints required")
    if len(a) <= 1:
        return 0
    seq = sturm_sequence(a)

    def variations(x: Fraction) -> int:
        signs = [(v > 0) for q in seq if (v := evaluate(q, x)) != 0]
        return sum(a != b for a, b in zip(signs[:-1], signs[1:], strict=True))

    return variations(lower) - variations(upper)


@dataclass(frozen=True)
class RealAlgebraicField:
    polynomial: tuple[Rational, ...]
    lo: Rational
    hi: Rational

    def __post_init__(self) -> None:
        p = trim(self.polynomial)
        if len(p) < 2 or len(gcd(p, derivative(p))) != 1:
            raise ValueError("primitive polynomial must be nonconstant and square-free")
        if root_count(p, self.lo, self.hi) != 1:
            raise ValueError("primitive interval must isolate exactly one real root")
        object.__setattr__(self, "polynomial", p)
        object.__setattr__(self, "lo", rational(self.lo))
        object.__setattr__(self, "hi", rational(self.hi))

    def number(self, coefficients: Sequence[Rational]) -> AlgebraicNumber:
        return AlgebraicNumber(self, divide(trim(coefficients), trim(self.polynomial))[1])

    def constant(self, value: Rational) -> AlgebraicNumber:
        return self.number((rational(value),))

    @property
    def generator(self) -> AlgebraicNumber:
        return self.number((0, 1))

    @classmethod
    def sqrt(cls, value: Rational) -> RealAlgebraicField:
        q = rational(value)
        if q <= 0:
            raise ValueError("positive radicand required")
        n, d = isqrt(q.numerator), isqrt(q.denominator)
        if n * n == q.numerator and d * d == q.denominator:
            r = Fraction(n, d)
            return cls((-r, 1), r - 1, r + 1)
        lower = isqrt(q.numerator // q.denominator)
        return cls((-q, 0, 1), lower, lower + 1)


@dataclass(frozen=True)
class AlgebraicNumber:
    field: RealAlgebraicField
    coefficients: UPoly

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coefficients", divide(trim(self.coefficients), trim(self.field.polynomial))[1]
        )

    def _coerce(self, other: AlgebraicNumber | Rational) -> AlgebraicNumber:
        out = other if isinstance(other, AlgebraicNumber) else self.field.constant(other)
        if out.field != self.field:
            raise ValueError("algebraic primitive elements differ")
        return out

    def __add__(self, other: AlgebraicNumber | Rational) -> AlgebraicNumber:
        return self.field.number(add(self.coefficients, self._coerce(other).coefficients))

    def __radd__(self, other: Rational) -> AlgebraicNumber:
        return self + other

    def __neg__(self) -> AlgebraicNumber:
        return self.field.number(scale(self.coefficients, -1))

    def __sub__(self, other: AlgebraicNumber | Rational) -> AlgebraicNumber:
        return self + -self._coerce(other)

    def __rsub__(self, other: Rational) -> AlgebraicNumber:
        return -self + other

    def __mul__(self, other: AlgebraicNumber | Rational) -> AlgebraicNumber:
        return self.field.number(multiply(self.coefficients, self._coerce(other).coefficients))

    def __rmul__(self, other: Rational) -> AlgebraicNumber:
        return self * other

    def inverse(self) -> AlgebraicNumber:
        a, b = trim(self.field.polynomial), self.coefficients
        u: UPoly = ()
        v: UPoly = (Fraction(1),)
        while b:
            q, rem = divide(a, b)
            a, b, u, v = b, rem, v, add(u, scale(multiply(q, v), -1))
        if len(a) != 1:
            raise ZeroDivisionError(
                "nonunit denominator; refine the primitive polynomial if necessary"
            )
        return self.field.number(scale(u, 1 / a[0]))

    def __truediv__(self, other: AlgebraicNumber | Rational) -> AlgebraicNumber:
        return self * self._coerce(other).inverse()

    def __pow__(self, n: int) -> AlgebraicNumber:
        if type(n) is not int:
            raise TypeError("algebraic powers must be integers")
        if n < 0:
            return self.inverse() ** -n
        out, base = self.field.constant(1), self
        while n:
            if n & 1:
                out = out * base
            n //= 2
            if n:
                base = base * base
        return out

    def is_zero(self) -> bool:
        if not self.coefficients:
            return True
        common = gcd(trim(self.field.polynomial), self.coefficients)
        return len(common) > 1 and root_count(common, self.field.lo, self.field.hi) == 1

    def sign(self, *, max_bisections: int = 4096) -> int:
        if self.is_zero():
            return 0
        lo, hi = rational(self.field.lo), rational(self.field.hi)
        primitive = trim(self.field.polynomial)
        for _ in range(max_bisections):
            # Exact interval Horner on rational endpoints.
            a = b = Fraction(0)
            for c in reversed(self.coefficients):
                products = (a * lo, a * hi, b * lo, b * hi)
                a, b = min(products) + c, max(products) + c
            if a > 0:
                return 1
            if b < 0:
                return -1
            mid = (lo + hi) / 2
            if evaluate(primitive, mid) == 0:
                value = evaluate(self.coefficients, mid)
                return 1 if value > 0 else -1
            if root_count(primitive, lo, mid):
                hi = mid
            else:
                lo = mid
        raise ArithmeticError("algebraic sign refinement budget exhausted")


__all__ = ["AlgebraicNumber", "RealAlgebraicField", "UPoly", "root_count", "sturm_sequence"]
