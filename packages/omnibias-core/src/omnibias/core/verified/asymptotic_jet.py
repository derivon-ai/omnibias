# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified confluent power/root primitives and exact fixed-product calculus.

These are finite analytic calculations on explicit coefficient data.
``power_compensator`` uses the entire beta-moment expansion of
``log(x) * integral_0^1 exp((b+t*(a-b))*log(x)) dt``. No division by
``a-b`` occurs, including when parameter intervals cross the diagonal.
``signed_root_primitive`` integrates ``1/(1+delta*t*t)`` through delta=0;
negative delta is admitted only before either real pole.

All transcendental calls require the directed backend. Polynomial weighted
derivatives are exact over Q and retain the acceleration of the curved
path ``(omega*exp(-s), u*exp(s))``. No boolean hypothesis can substitute
for either an enclosure calculation or a polynomial identity.
"""

from __future__ import annotations

import math
from fractions import Fraction

from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.transcend import certificate_mode, exp_iv, ln_iv

_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


def _finite(value: IntervalLike, name: str) -> Interval:
    result = Interval.from_value(value)
    if not math.isfinite(result.lo) or not math.isfinite(result.hi):
        raise ValueError(f"{name} must have finite endpoints")
    return result


def _order(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _options(terms: int, cells: int) -> None:
    if type(terms) is not int or not 1 <= terms <= 256:
        raise ValueError("terms must be an integer in [1,256]")
    if type(cells) is not int or not 1 <= cells <= 4096:
        raise ValueError("cells must be an integer in [1,4096]")


def _checked(value: Interval) -> Interval:
    if not math.isfinite(value.lo) or not math.isfinite(value.hi):
        raise ArithmeticError("enclosure overflow; restrict the parameter domain")
    return value


def _beta(p: int, q: int) -> Interval:
    return Interval.from_rational(
        Fraction(math.factorial(p) * math.factorial(q), math.factorial(p + q + 1))
    )


def _beta_exponential(z: Interval, p: int, q: int, terms: int) -> Interval:
    r"""Enclose integral t^p (1-t)^q exp(z*t) dt for |z|<=1/2.

    Taylor's integral remainder is at most
    ``B(p+1,q+1) exp(|z|) |z|^(N+1)/(N+1)!`` after degree N.
    """
    result = _ZERO
    power = _ONE
    for n in range(terms + 1):
        result = result + power * _beta(p + n, q) * Fraction(1, math.factorial(n))
        power = power * z
    radius = Interval.point(max(abs(z.lo), abs(z.hi)))
    remainder = (
        _beta(p, q) * exp_iv(radius) * radius ** (terms + 1)
        * Fraction(1, math.factorial(terms + 1))
    )
    return result + Interval(-remainder.hi, remainder.hi)


def power_compensator(
    a: IntervalLike,
    b: IntervalLike,
    x: IntervalLike,
    *,
    a_order: int = 0,
    b_order: int = 0,
    log_order: int = 0,
    terms: int = 20,
    cells: int = 64,
) -> Interval:
    r"""Enclose ``partial_a^p partial_b^q (x partial_x)^k C(a,b;x)``.

    ``C=(x**a-x**b)/(a-b)`` off the diagonal and ``C(a,a;x)=x**a log(x)``
    on it. x must be strictly positive. Derivatives are unnormalized.
    The independent a,b derivatives on the diagonal retain the beta factor;
    they are not derivatives along the constraint a=b.

    Near confluence an entire series with an explicit tail is evaluated.
    Elsewhere a partition of the integral gives a sound range enclosure.
    The returned interval includes every input in the supplied boxes. Total
    derivative order is limited to 16 as an explicit arithmetic budget.
    Underflow is enclosed; overflow raises instead of returning a false bound.
    """
    for order, name in ((a_order, "a_order"), (b_order, "b_order"), (log_order, "log_order")):
        _order(order, name)
    if a_order + b_order + log_order > 16:
        raise ValueError("total derivative order exceeds the arithmetic budget 16")
    _options(terms, cells)
    aa, bb, xx = _finite(a, "a"), _finite(b, "b"), _finite(x, "x")
    if xx.lo <= 0:
        raise ValueError("x must be strictly positive")
    p, q, k = a_order, b_order, log_order
    m = p + q + 1
    with certificate_mode():
        logx = ln_iv(xx)
        split = aa - bb
        z = _checked(split * logx)
        result = _ZERO
        diagonal = aa.lo == aa.hi == bb.lo == bb.hi
        small = max(abs(z.lo), abs(z.hi)) <= 0.5
        for j in range(min(k, m) + 1):
            ell = k - j
            factor = math.comb(k, j) * math.factorial(m) // math.factorial(m - j)
            if diagonal:
                moment = _beta(p, q) * aa**ell * exp_iv(_checked(aa * logx))
            elif small:
                moment = _ZERO
                for r in range(ell + 1):
                    moment = moment + (
                        math.comb(ell, r) * bb ** (ell - r) * split**r
                        * _beta_exponential(z, p + r, q, terms)
                    )
                moment = moment * exp_iv(_checked(bb * logx))
            else:
                moment = _ZERO
                for index in range(cells):
                    t = Interval(
                        Interval.from_rational(Fraction(index, cells)).lo,
                        Interval.from_rational(Fraction(index + 1, cells)).hi,
                    )
                    exponent = bb + t * split
                    moment = moment + (
                        t**p * (_ONE - t)**q * exponent**ell
                        * exp_iv(_checked(exponent * logx)) / cells
                    )
            result = result + factor * logx ** (m - j) * moment
        return _checked(result)


def signed_root_primitive(
    delta: IntervalLike,
    value: IntervalLike,
    *,
    delta_order: int = 0,
    endpoint_order: int = 0,
    terms: int = 24,
    cells: int = 64,
) -> Interval:
    r"""Enclose derivatives of ``S(delta,v)=integral_0^v dt/(1+delta*t²)``.

    S is ``atan(sqrt(delta)*v)/sqrt(delta)`` for positive delta, v at
    delta=0, and ``atanh(sqrt(-delta)*v)/sqrt(-delta)`` for negative delta
    before the pole. The same real integral handles intervals crossing zero.

    ``delta_order=p`` and ``endpoint_order=0 or 1`` give unnormalized mixed
    derivatives. For endpoint order zero the integrand is
    ``(-1)^p p! v^(2p+1) t^(2p)/(1+delta*v²*t²)^(p+1)`` on [0,1].
    Near zero a binomial series has a geometric bound on its omitted tail;
    elsewhere interval integration is used. The pole test covers the entire
    parameter and endpoint boxes, including negative endpoints.
    """
    _order(delta_order, "delta_order")
    _order(endpoint_order, "endpoint_order")
    if delta_order > 16:
        raise ValueError("delta_order exceeds the arithmetic budget 16")
    if endpoint_order > 1:
        raise NotImplementedError("endpoint_order above one is not implemented")
    _options(terms, cells)
    dd, vv = _finite(delta, "delta"), _finite(value, "value")
    p = delta_order
    y = _checked(dd * vv**2)
    if (_ONE + Interval(min(0.0, y.lo), max(0.0, y.hi))).lo <= 0:
        raise ValueError("the integration domain may meet a real pole")
    factor = (-1) ** p * math.factorial(p)
    if endpoint_order:
        return _checked(factor * vv ** (2 * p) / (_ONE + y) ** (p + 1))
    if dd.lo == dd.hi == 0.0:
        return _checked(Fraction(factor, 2 * p + 1) * vv ** (2 * p + 1))
    radius = max(abs(y.lo), abs(y.hi))
    integral = _ZERO
    if radius <= 0.25:
        degree = max(terms, p + 1)
        power = _ONE
        for n in range(degree + 1):
            integral = integral + Fraction(math.comb(n + p, p), 2 * n + 2 * p + 1) * power
            power = power * (-y)
        r = Interval.point(radius)
        first = Fraction(math.comb(degree + 1 + p, p), 2 * (degree + 1) + 2 * p + 1) * r ** (degree + 1)
        ratio = r * Fraction(degree + p + 2, degree + 2)
        tail = first / (_ONE - ratio)
        integral = integral + Interval(-tail.hi, tail.hi)
    else:
        for index in range(cells):
            t = Interval(
                Interval.from_rational(Fraction(index, cells)).lo,
                Interval.from_rational(Fraction(index + 1, cells)).hi,
            )
            integral = integral + t ** (2 * p) / (_ONE + y * t**2) ** (p + 1) / cells
    return _checked(factor * vv ** (2 * p + 1) * integral)


def weighted_scale_derivative(
    polynomial: SparsePolynomial,
    *,
    order: int = 1,
    omega_axis: int = 0,
    u_axis: int = 1,
) -> SparsePolynomial:
    r"""Exact ``(u partial_u - omega partial_omega)^order polynomial`` over Q.

    A monomial omega^i u^j has eigenvalue j-i. This includes every path
    acceleration term: at second order, ``omega² f_ww - 2 omega u f_wu
    + u² f_uu + omega f_w + u f_u``. Other polynomial axes are fixed.
    """
    _order(order, "order")
    if (type(omega_axis) is not int or type(u_axis) is not int
            or omega_axis == u_axis or not 0 <= min(omega_axis, u_axis)
            or max(omega_axis, u_axis) >= polynomial.nvars):
        raise ValueError("two distinct polynomial coordinate axes are required")
    return SparsePolynomial(
        polynomial.nvars,
        {idx: coefficient * (idx[u_axis] - idx[omega_axis])**order
         for idx, coefficient in polynomial.terms},
        budget=polynomial.budget,
    )


def verify_fixed_product_derivative(
    source: SparsePolynomial, claimed: SparsePolynomial, *, order: int = 1
) -> bool:
    """Replay the claimed fixed-product derivative as an exact polynomial identity.

    This checks supplied operands; it does not accept a caller's proof flag.
    A straight directional Hessian generally fails the second-order check.
    """
    expected = weighted_scale_derivative(source, order=order)
    return claimed.nvars == expected.nvars and claimed.terms == expected.terms


def fixed_product_jet(
    polynomial: SparsePolynomial,
    omega: IntervalLike,
    u: IntervalLike,
    *,
    order: int = 2,
) -> tuple[Interval, ...]:
    """Enclose unnormalized derivatives on (omega exp(-s),u exp(s)) at s=0.

    The supplied polynomial has exactly two variables ordered (omega,u).
    Both scales must be positive. Their product is fixed along each path;
    no independent variation of epsilon is inserted into these derivatives.
    """
    _order(order, "order")
    if polynomial.nvars != 2:
        raise ValueError("a two-variable polynomial ordered (omega,u) is required")
    ww, uu = _finite(omega, "omega"), _finite(u, "u")
    if min(ww.lo, uu.lo) <= 0:
        raise ValueError("fixed-product scales must be strictly positive")
    result = []
    for k in range(order + 1):
        derivative = weighted_scale_derivative(polynomial, order=k)
        value = _ZERO
        for idx, coefficient in derivative.terms:
            value = value + coefficient * ww**idx[0] * uu**idx[1]
        result.append(_checked(value))
    return tuple(result)


__all__ = [
    "fixed_product_jet",
    "power_compensator",
    "signed_root_primitive",
    "verify_fixed_product_derivative",
    "weighted_scale_derivative",
]
