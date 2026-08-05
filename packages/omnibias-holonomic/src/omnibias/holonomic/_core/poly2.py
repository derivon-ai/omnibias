# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact bivariate polynomials and rational functions over ``Q[n, k]``.

Creative telescoping (:mod:`.zeilberger`) is inherently two-variable -- the summation index
``k`` and the recurrence parameter ``n`` -- so the telescoping certificate ``G(n, k)`` and
the identity ``sum_i c_i(n) F(n+i,k) = G(n,k+1) - G(n,k)`` live in ``Q(n, k)``. A
:data:`Poly2` is a sparse ``dict`` mapping a monomial exponent pair ``(i, j)`` (power of
``n``, power of ``k``) to its rational coefficient; a :data:`Rat2` is an *unreduced*
``(num, den)`` pair (no bivariate GCD is taken -- identities are checked by
cross-multiplication, which needs none). Everything is exact.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

#: A bivariate polynomial: ``{(i, j): coeff}`` for ``coeff * n^i * k^j`` (zero coeffs dropped).
Poly2 = dict[tuple[int, int], Fraction]


def p2_zero() -> Poly2:
    """The zero polynomial."""
    return {}


def p2_const(c: Fraction | int) -> Poly2:
    """The constant polynomial ``c``."""
    f = c if isinstance(c, Fraction) else Fraction(c)
    return {} if f == 0 else {(0, 0): f}


def p2_linear(a_n: int, a_k: int, const: int) -> Poly2:
    """The linear form ``a_n * n + a_k * k + const``."""
    out: Poly2 = {}
    if a_n:
        out[(1, 0)] = Fraction(a_n)
    if a_k:
        out[(0, 1)] = Fraction(a_k)
    if const:
        out[(0, 0)] = Fraction(const)
    return out


def p2_is_zero(p: Mapping[tuple[int, int], Fraction]) -> bool:
    """Whether ``p`` is identically zero."""
    return all(v == 0 for v in p.values())


def _clean(p: Poly2) -> Poly2:
    return {kk: v for kk, v in p.items() if v != 0}


def p2_add(a: Poly2, b: Poly2) -> Poly2:
    """Sum ``a + b``."""
    out: Poly2 = dict(a)
    for key, v in b.items():
        out[key] = out.get(key, Fraction(0)) + v
    return _clean(out)


def p2_sub(a: Poly2, b: Poly2) -> Poly2:
    """Difference ``a - b``."""
    out: Poly2 = dict(a)
    for key, v in b.items():
        out[key] = out.get(key, Fraction(0)) - v
    return _clean(out)


def p2_neg(a: Poly2) -> Poly2:
    """Negation ``-a``."""
    return {key: -v for key, v in a.items()}


def p2_scale(a: Poly2, c: Fraction | int) -> Poly2:
    """Scalar multiple ``c * a``."""
    f = c if isinstance(c, Fraction) else Fraction(c)
    if f == 0:
        return {}
    return {key: v * f for key, v in a.items()}


def p2_mul(a: Poly2, b: Poly2) -> Poly2:
    """Product ``a * b``."""
    out: Poly2 = {}
    for (i1, j1), v1 in a.items():
        if v1 == 0:
            continue
        for (i2, j2), v2 in b.items():
            if v2 == 0:
                continue
            key = (i1 + i2, j1 + j2)
            out[key] = out.get(key, Fraction(0)) + v1 * v2
    return _clean(out)


def p2_pow(a: Poly2, e: int) -> Poly2:
    """Non-negative integer power ``a ** e``."""
    if e < 0:
        raise ValueError("p2_pow needs a non-negative exponent")
    result = p2_const(1)
    base = a
    while e:
        if e & 1:
            result = p2_mul(result, base)
        e >>= 1
        if e:
            base = p2_mul(base, base)
    return result


def p2_eval(p: Mapping[tuple[int, int], Fraction], n: Fraction | int, k: Fraction | int) -> Fraction:
    """Evaluate ``p(n, k)`` exactly."""
    nn = n if isinstance(n, Fraction) else Fraction(n)
    kk = k if isinstance(k, Fraction) else Fraction(k)
    total = Fraction(0)
    for (i, j), v in p.items():
        total += v * nn**i * kk**j
    return total


def p2_shift_k(p: Poly2, delta: int) -> Poly2:
    """Substitute ``k -> k + delta``."""
    out: Poly2 = {}
    for (i, j), v in p.items():
        # (k + delta)^j = sum_t C(j, t) delta^{j-t} k^t
        for t in range(j + 1):
            from math import comb

            coeff = v * comb(j, t) * Fraction(delta) ** (j - t)
            key = (i, t)
            out[key] = out.get(key, Fraction(0)) + coeff
    return _clean(out)


def p2_shift_n(p: Poly2, delta: int) -> Poly2:
    """Substitute ``n -> n + delta``."""
    out: Poly2 = {}
    for (i, j), v in p.items():
        for t in range(i + 1):
            from math import comb

            coeff = v * comb(i, t) * Fraction(delta) ** (i - t)
            key = (t, j)
            out[key] = out.get(key, Fraction(0)) + coeff
    return _clean(out)


def p2_degree_k(p: Poly2) -> int:
    """Degree of ``p`` in ``k`` (``-1`` for zero)."""
    degs = [j for (_i, j), v in p.items() if v != 0]
    return max(degs) if degs else -1


def p2_degree_n(p: Poly2) -> int:
    """Degree of ``p`` in ``n`` (``-1`` for zero)."""
    degs = [i for (i, _j), v in p.items() if v != 0]
    return max(degs) if degs else -1


#: An unreduced bivariate rational function ``num / den``.
Rat2 = tuple[Poly2, Poly2]


def r2_from_poly(p: Poly2) -> Rat2:
    """Embed a polynomial as ``p / 1``."""
    return (dict(p), p2_const(1))


def r2_one() -> Rat2:
    """The rational function ``1``."""
    return (p2_const(1), p2_const(1))


def r2_mul(a: Rat2, b: Rat2) -> Rat2:
    """Product (unreduced)."""
    return (p2_mul(a[0], b[0]), p2_mul(a[1], b[1]))


def r2_add(a: Rat2, b: Rat2) -> Rat2:
    """Sum (unreduced, common denominator by cross-multiplication)."""
    return (p2_add(p2_mul(a[0], b[1]), p2_mul(b[0], a[1])), p2_mul(a[1], b[1]))


def r2_sub(a: Rat2, b: Rat2) -> Rat2:
    """Difference (unreduced)."""
    return (p2_sub(p2_mul(a[0], b[1]), p2_mul(b[0], a[1])), p2_mul(a[1], b[1]))


def r2_shift_k(a: Rat2, delta: int) -> Rat2:
    """Substitute ``k -> k + delta`` in both numerator and denominator."""
    return (p2_shift_k(a[0], delta), p2_shift_k(a[1], delta))


def r2_shift_n(a: Rat2, delta: int) -> Rat2:
    """Substitute ``n -> n + delta`` in both numerator and denominator."""
    return (p2_shift_n(a[0], delta), p2_shift_n(a[1], delta))


def r2_is_zero(a: Rat2) -> bool:
    """Whether the rational function is identically zero (numerator vanishes)."""
    return p2_is_zero(a[0])


def r2_equal(a: Rat2, b: Rat2) -> bool:
    """Whether ``a == b`` as rational functions (cross-multiplied polynomial identity)."""
    return p2_is_zero(p2_sub(p2_mul(a[0], b[1]), p2_mul(b[0], a[1])))


__all__ = [
    "Poly2",
    "Rat2",
    "p2_add",
    "p2_const",
    "p2_degree_k",
    "p2_degree_n",
    "p2_eval",
    "p2_is_zero",
    "p2_linear",
    "p2_mul",
    "p2_neg",
    "p2_pow",
    "p2_scale",
    "p2_shift_k",
    "p2_shift_n",
    "p2_sub",
    "p2_zero",
    "r2_add",
    "r2_equal",
    "r2_from_poly",
    "r2_is_zero",
    "r2_mul",
    "r2_one",
    "r2_shift_k",
    "r2_shift_n",
    "r2_sub",
]
