# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded exact Chebyshev calculus for derivative enclosure consumers.

All operations charge the caller's shared ``_Arithmetic`` meter. Coefficients
use ordinary Chebyshev normalization; differentiated coefficients represent
raw derivatives, not Taylor coefficients. No numerical backend is imported.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.gaussian_jet import _Arithmetic

Rows = tuple[tuple[Fraction, ...], ...]


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, excluding bool")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _order(value: int, name: str, arithmetic: _Arithmetic, *, minimum: int = 0) -> int:
    result = _integer(value, name, minimum=minimum)
    if result > arithmetic.limits.max_order:
        raise ValueError(f"{name} exceeds the derivative resource budget")
    return result


def _power(value: Fraction, exponent: int, arithmetic: _Arithmetic) -> Fraction:
    """Checked nonnegative integer exponentiation in logarithmic work."""
    result = Fraction(1)
    while exponent:
        if exponent & 1:
            result = arithmetic.mul(result, value)
        exponent //= 2
        if exponent:
            value = arithmetic.mul(value, value)
    return result


def _binomial(n: int, k: int, arithmetic: _Arithmetic) -> Fraction:
    """Form one positive coefficient without unchecked large integer scratch."""
    k = min(k, n-k)
    result = Fraction(1)
    for j in range(1, k+1):
        numerator = arithmetic.mul(result, arithmetic.exact(n-k+j))
        result = arithmetic._coefficient(numerator/j)
    return result


def differentiated_chebyshev_coefficients(
    coefficients: Rows, order: int, half_width: Fraction, arithmetic: _Arithmetic,
) -> Rows:
    """Return raw physical-t derivatives of sum c_n*T_n(t/half_width).

    Output rows retain the natural degree ``max(0, degree-order)``. The zero
    polynomial has one row; trailing zero rows are otherwise not trimmed.
    The finite table is bounded by entries/work, while ``order`` is also
    bounded by ``max_order``. Inputs must be nonempty rectangular tuples.
    """
    derivative = _order(order, "order", arithmetic)
    width = arithmetic.exact(half_width)
    if width <= 0:
        raise ValueError("half_width must be strictly positive")
    if not isinstance(coefficients, tuple) or not coefficients:
        raise ValueError("coefficients must be a nonempty tuple of rows")
    if not isinstance(coefficients[0], tuple) or not coefficients[0]:
        raise ValueError("coefficient rows must be nonempty tuples")
    count, dimension = len(coefficients), len(coefficients[0])
    arithmetic._entries(6*count*dimension + 16)
    arithmetic._work(8*(min(derivative, count-1)+1)*count*dimension + 16)
    if any(not isinstance(row, tuple) or len(row) != dimension for row in coefficients):
        raise ValueError("coefficient rows must be rectangular tuples")
    current = tuple(tuple(arithmetic.exact(value) for value in row) for row in coefficients)
    if derivative >= count:
        return ((Fraction(0),)*dimension,)
    for _ in range(derivative):
        degree = len(current)-1
        following = [[Fraction(0)]*dimension for _ in range(degree)]
        for n in range(degree-1, -1, -1):
            factor = arithmetic.exact(2*(n+1))
            for axis in range(dimension):
                value = arithmetic.mul(factor, current[n+1][axis])
                if n+2 < degree:
                    value = arithmetic.add(value, following[n+2][axis])
                following[n][axis] = value
        current = tuple(tuple(arithmetic._coefficient((arithmetic._coefficient(value/2) if n == 0 else value)/width)
                              for value in row) for n, row in enumerate(following))
    return current


def chebyshev_derivative_norm(degree: int, order: int, arithmetic: _Arithmetic) -> Fraction:
    """Exact sup on [-1,1] of |T_degree^(order)|, attained at an endpoint."""
    n = _integer(degree, "degree")
    k = _order(order, "order", arithmetic)
    exact_n = arithmetic.exact(n)
    arithmetic._entries(16)
    arithmetic._work(8*(k+1))
    if n < k:
        return Fraction(0)
    result = Fraction(1)
    if k:
        squared = arithmetic.mul(exact_n, exact_n)
        for j in range(k):
            factor = arithmetic.add(squared, -arithmetic.exact(j*j))
            numerator = arithmetic.mul(result, factor)
            result = arithmetic._coefficient(numerator/(2*j+1))
    return result


def geometric_power_tail(r: Fraction, start: int, power: int, arithmetic: _Arithmetic) -> Fraction:
    """Exactly sum n**power*r**n for n>=start, 0<=r<1, in O(power²).

    Define H_p=sum_{j>=0}(start+j)**p*r**j. The positive recurrence
    (1-r)*H_p=start**p+r*sum_{k<p}binom(p,k)*H_k avoids cancellation.
    In particular 0**0 is one, as appropriate for the zeroth moment.
    """
    first = _integer(start, "start")
    exponent = _order(power, "power", arithmetic)
    ratio = arithmetic.exact(r)
    exact_start = arithmetic.exact(first)
    if not 0 <= ratio < 1:
        raise ValueError("r must satisfy 0 <= r < 1")
    arithmetic._entries(4*(exponent+1) + 16)
    arithmetic._work(12*(exponent+1)**2 + 8*(first.bit_length()+1))
    if ratio == 0:
        return Fraction(int(first == 0 and exponent == 0))
    complement = arithmetic.add(Fraction(1), -ratio)
    moments = [arithmetic._coefficient(1/complement)]
    start_power = Fraction(1)
    for p in range(1, exponent+1):
        start_power = arithmetic.mul(start_power, exact_start)
        total, binomial = Fraction(0), Fraction(1)
        for k in range(p):
            total = arithmetic.add(total, arithmetic.mul(binomial, moments[k]))
            if k+1 < p:
                binomial = arithmetic._coefficient(arithmetic.mul(binomial, arithmetic.exact(p-k))/(k+1))
        numerator = arithmetic.add(start_power, arithmetic.mul(ratio, total))
        moments.append(arithmetic._coefficient(numerator/complement))
    return arithmetic.mul(_power(ratio, first, arithmetic), moments[-1])


def chebyshev_derivative_tail(r: Fraction, start: int, order: int, arithmetic: _Arithmetic) -> Fraction:
    r"""Exactly sum ||T_n^(order)||_infinity*r**n for n>=start in O(order).

    Require 0<=r<1, order>=1 and start>=order. Write k=order, K=start,
    and g_j=r**j/(1-r)**(j+1). Vandermonde's identity gives the positive sum
    2**(k-1)*(k-1)!*r**K*sum_{j=0}^{2*k-1} binom(K+k-1,2*k-1-j)
    *((K+j)*g_j+(j+1)*g_(j+1)). Only the requested k is bounded by
    max_order; the auxiliary index 2*k is covered by the work preflight.
    """
    k = _order(order, "order", arithmetic, minimum=1)
    first = _integer(start, "start", minimum=k)
    ratio = arithmetic.exact(r)
    arithmetic.exact(first)
    if not 0 <= ratio < 1:
        raise ValueError("r must satisfy 0 <= r < 1")
    arithmetic._entries(32)
    arithmetic._work(48*(k+1) + 8*(first.bit_length()+1))
    if ratio == 0:
        return Fraction(0)
    complement = arithmetic.add(Fraction(1), -ratio)
    geometric = arithmetic._coefficient(1/complement)
    step = arithmetic._coefficient(ratio/complement)
    upper, lower = first+k-1, 2*k-1
    arithmetic.exact(upper)
    binomial = _binomial(upper, lower, arithmetic)
    total = Fraction(0)
    for j in range(2*k):
        following = arithmetic.mul(geometric, step)
        term = arithmetic.add(arithmetic.mul(arithmetic.exact(first+j), geometric),
                              arithmetic.mul(arithmetic.exact(j+1), following))
        total = arithmetic.add(total, arithmetic.mul(binomial, term))
        if j+1 < 2*k:
            numerator = arithmetic.mul(binomial, arithmetic.exact(lower-j))
            binomial = arithmetic._coefficient(numerator/(upper-lower+j+1))
        geometric = following
    prefactor = Fraction(1)
    for j in range(1, k):
        prefactor = arithmetic.mul(prefactor, arithmetic.exact(2*j))
    return arithmetic.mul(arithmetic.mul(prefactor, _power(ratio, first, arithmetic)), total)


__all__ = ["chebyshev_derivative_norm", "chebyshev_derivative_tail",
           "differentiated_chebyshev_coefficients", "geometric_power_tail"]
