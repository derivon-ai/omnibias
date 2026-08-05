# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified Euler-Maclaurin summation and the special functions it yields.

Euler-Maclaurin is *exactly* the composition this repo already owns: the exact
Bernoulli numbers (:func:`omnibias.core.verified.coeffs.bernoulli_number_exact`,
read off the closed-form ``tanh`` tower) weighting high-order derivative
enclosures, closed with a **rigorous** remainder. For ``p`` correction terms,

.. math::

    \sum_{k=a}^{b} f(k) = \int_a^b f
        + \tfrac12\,(f(a) + f(b))
        + \sum_{j=1}^{p} \frac{B_{2j}}{(2j)!}\,\bigl(f^{(2j-1)}(b) - f^{(2j-1)}(a)\bigr)
        + R_p,

with the standard bound (``max_{[0,1]} |\tilde B_{2p}| = |B_{2p}|``)

.. math::

    |R_p| \le \frac{|B_{2p}|}{(2p)!}\int_a^b |f^{(2p)}(x)|\,dx
        \le \frac{|B_{2p}|}{(2p)!}\,(b-a)\,\max_{[a,b]}|f^{(2p)}|.

:func:`euler_maclaurin_sum` takes a caller-supplied derivative-enclosure oracle
``f^{(k)}`` (any function enclosable over an interval -- not just the activation
dictionary) plus a certified enclosure of ``\int_a^b f`` and returns a guaranteed
enclosure of the finite sum. :func:`log_gamma_iv` and :func:`digamma_iv` are the
special functions this unlocks: Stirling / asymptotic series with a remainder
enclosed by the classical *first-omitted-term* bound (valid on the positive real
axis), tightened by shifting the argument up via the ``Gamma`` / ``psi``
recurrences. Everything is **closed-form + numerical** honest: the coefficients
are exact, the enclosures outward-rounded, the remainder rigorously bounded.
"""

from __future__ import annotations

from collections.abc import Callable
from math import factorial

from omnibias.core.verified.coeffs import bernoulli_number_exact
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, ln_iv

#: A derivative-enclosure oracle: ``deriv(k, x)`` encloses ``f^{(k)}`` over ``x``.
DerivativeOracle = Callable[[int, Interval], Interval]

_HALF = Interval.point(0.5)


def euler_maclaurin_sum(
    deriv: DerivativeOracle,
    integral: Interval,
    a: int,
    b: int,
    *,
    terms: int,
) -> Interval:
    r"""Guaranteed enclosure of ``sum_{k=a}^{b} f(k)`` via Euler-Maclaurin.

    Parameters
    ----------
    deriv:
        Oracle returning a guaranteed enclosure of ``f^{(k)}`` over an interval;
        ``deriv(0, x)`` is ``f`` itself. It must accept a *box* argument (used to
        bound ``f^{(2*terms)}`` over ``[a, b]`` for the remainder).
    integral:
        A guaranteed enclosure of ``\int_a^b f(x) dx`` (the caller supplies it in
        closed form, e.g. for a ``p``-series or a log term).
    a, b:
        Integer summation limits, inclusive, with ``a <= b``.
    terms:
        The number ``p >= 1`` of Bernoulli correction pairs; the remainder uses
        ``f^{(2p)}``.
    """
    if b < a:
        raise ValueError(f"require a <= b, got a={a}, b={b}")
    if terms < 1:
        raise ValueError(f"terms (p) must be >= 1, got {terms}")

    a_iv = Interval.from_rational(a)
    b_iv = Interval.from_rational(b)
    total = integral + (deriv(0, a_iv) + deriv(0, b_iv)) * _HALF

    for j in range(1, terms + 1):
        coeff = Interval.from_rational(bernoulli_number_exact(2 * j) / factorial(2 * j))
        total = total + coeff * (deriv(2 * j - 1, b_iv) - deriv(2 * j - 1, a_iv))

    # Remainder: |R_p| <= |B_{2p}|/(2p)! * (b - a) * max_[a,b] |f^{(2p)}|.
    box = Interval(float(a), float(b))
    f2p = deriv(2 * terms, box)
    bound_iv = (
        Interval.from_rational(abs(bernoulli_number_exact(2 * terms)) / factorial(2 * terms))
        * Interval.from_rational(b - a)
        * Interval.point(f2p.mag)
    )
    bound = bound_iv.hi
    return total + Interval(-bound, bound)


def _half_ln_two_pi() -> Interval:
    """Rigorous enclosure of ``(1/2) ln(2 pi)`` (composes :data:`PI_IV`)."""
    return _HALF * ln_iv(PI_IV * 2)


def _stirling_log_gamma(w: Interval, terms: int) -> Interval:
    r"""Stirling series for ``ln Gamma(w)`` with a first-omitted-term remainder.

    ``ln Gamma(w) = (w - 1/2) ln w - w + (1/2) ln(2 pi)
    + sum_{j=1}^{p} B_{2j}/((2j)(2j-1)) w^{-(2j-1)} + R``. On the positive real
    axis ``R`` is bounded in magnitude by, and shares the sign of, the first
    omitted term, so ``R`` lies in the hull of ``0`` and that term -- rigorous for
    every real ``w`` in the interval.
    """
    ln_w = ln_iv(w)
    main = (w - _HALF) * ln_w - w + _half_ln_two_pi()

    series = Interval.point(0.0)
    for j in range(1, terms + 1):
        coeff = Interval.from_rational(bernoulli_number_exact(2 * j) / ((2 * j) * (2 * j - 1)))
        series = series + coeff * w.pow_int(2 * j - 1).reciprocal()

    p = terms + 1
    first_omitted = Interval.from_rational(
        bernoulli_number_exact(2 * p) / ((2 * p) * (2 * p - 1))
    ) * w.pow_int(2 * p - 1).reciprocal()
    remainder = Interval(min(0.0, first_omitted.lo), max(0.0, first_omitted.hi))
    return main + series + remainder


def _shift_count(lo: float, shift_to: float) -> int:
    """Smallest ``n >= 0`` with ``lo + n >= shift_to``."""
    n = 0
    while lo + n < shift_to:
        n += 1
    return n


def log_gamma_iv(x: Interval, *, terms: int = 6, shift_to: float = 8.0) -> Interval:
    r"""Guaranteed enclosure of ``ln Gamma(x)`` for a strictly positive interval ``x``.

    The argument is shifted up to ``>= shift_to`` via
    ``ln Gamma(x) = ln Gamma(x + n) - sum_{k=0}^{n-1} ln(x + k)`` so the Stirling
    series is tight, then :func:`_stirling_log_gamma` closes it with a rigorous
    remainder. Increase ``terms`` / ``shift_to`` to tighten.
    """
    if x.lo <= 0.0:
        raise ValueError("log_gamma_iv requires a strictly positive interval")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    n = _shift_count(x.lo, shift_to)
    shift = Interval.point(0.0)
    for k in range(n):
        shift = shift + ln_iv(x + Interval.from_rational(k))
    w = x + Interval.from_rational(n)
    return _stirling_log_gamma(w, terms) - shift


def _stirling_digamma(w: Interval, terms: int) -> Interval:
    r"""Asymptotic series for ``psi(w)`` with a first-omitted-term remainder.

    ``psi(w) = ln w - 1/(2w) - sum_{j=1}^{p} B_{2j}/(2j) w^{-2j} + R``; on the
    positive real axis ``R`` lies in the hull of ``0`` and the first omitted term.
    """
    main = ln_iv(w) - _HALF * w.reciprocal()

    series = Interval.point(0.0)
    for j in range(1, terms + 1):
        coeff = Interval.from_rational(bernoulli_number_exact(2 * j) / (2 * j))
        series = series - coeff * w.pow_int(2 * j).reciprocal()

    p = terms + 1
    first_omitted = -(
        Interval.from_rational(bernoulli_number_exact(2 * p) / (2 * p)) * w.pow_int(2 * p).reciprocal()
    )
    remainder = Interval(min(0.0, first_omitted.lo), max(0.0, first_omitted.hi))
    return main + series + remainder


def digamma_iv(x: Interval, *, terms: int = 6, shift_to: float = 8.0) -> Interval:
    r"""Guaranteed enclosure of the digamma ``psi(x)`` for a strictly positive ``x``.

    Shifted up via ``psi(x) = psi(x + n) - sum_{k=0}^{n-1} 1/(x + k)`` before the
    asymptotic series, mirroring :func:`log_gamma_iv`.
    """
    if x.lo <= 0.0:
        raise ValueError("digamma_iv requires a strictly positive interval")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    n = _shift_count(x.lo, shift_to)
    shift = Interval.point(0.0)
    for k in range(n):
        shift = shift + (x + Interval.from_rational(k)).reciprocal()
    w = x + Interval.from_rational(n)
    return _stirling_digamma(w, terms) - shift


__all__ = [
    "DerivativeOracle",
    "digamma_iv",
    "euler_maclaurin_sum",
    "log_gamma_iv",
]
