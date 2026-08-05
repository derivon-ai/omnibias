# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sheffer-sequence *generation* and the umbral operator layer (exact rational arithmetic).

Where :mod:`omnibias.difference._core.umbral` supplies the umbral *transforms* (Newton
interpolation, binomial / Stirling transforms, the formal-power-series substrate, Riordan
arrays, and Sheffer *classification*), this module supplies the *generative* heart of the
umbral calculus:

* :func:`sheffer_sequence` -- build the polynomial sequence ``s_0 .. s_n`` of the Sheffer
  pair ``(g, f)`` from its generating function ``(1/g(fbar(t))) exp(x fbar(t))``, where
  ``fbar`` is the compositional inverse of the delta series ``f``;
* :func:`associated_sequence` -- the binomial-type (``g = 1``) special case;
* :func:`umbral_composition` -- the umbral (substitutional) composition of two sequences;
* :func:`shift_polynomial` / :func:`delta_operator_apply` / :func:`pincherle_derivative` --
  the shift operator ``E^a``, a delta operator ``Q = f(D)``, and the Pincherle derivative
  ``Q' = QX - XQ = f'(D)``;
* :func:`umbral_functional` -- the linear functional ``<L | x^k> = mu_k`` on polynomials.

Everything runs in exact :class:`~fractions.Fraction` arithmetic (**closed-form**), so the
umbral identities hold exactly. This is the founding ``delta -> 0`` derivative register of
:mod:`omnibias.difference`; it is **not** the ``beta -> inf`` feasibility penalty of the
optimization packages nor the ``q -> 1`` deformation of :mod:`omnibias.qcalculus` -- same
"collapse" word, different limits, never conflated. There is no ``autodiff-exact`` path here.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import comb, factorial

from omnibias.difference._core.umbral import (
    compose_series,
    compositional_inverse,
    series_reciprocal,
    sheffer_classify,
)

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def _poly_derivative(coeffs: Sequence[Fraction]) -> list[Fraction]:
    """Ordinary derivative of a polynomial in ascending-coefficient form."""
    return [k * coeffs[k] for k in range(1, len(coeffs))]


def _trim(coeffs: list[Fraction]) -> tuple[Fraction, ...]:
    """Drop trailing zeros, keeping at least a single (possibly zero) coefficient."""
    out = list(coeffs)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def _series_powers(series: Sequence[Rational], order: int) -> list[list[Fraction]]:
    r"""Powers ``series^0 .. series^order`` as ``t``-coefficient lists truncated to ``order``.

    ``series`` must be a delta series (zero constant term) so ``series^m`` starts at ``t^m``.
    """
    base = [Fraction(0)] * (order + 1)
    for i, c in enumerate(series):
        if i > order:
            break
        base[i] = _frac(c)
    powers = [[Fraction(0)] * (order + 1) for _ in range(order + 1)]
    powers[0][0] = Fraction(1)  # series^0 = 1
    for m in range(1, order + 1):
        prev, cur = powers[m - 1], powers[m]
        for i in range(order + 1):
            if prev[i] == 0:
                continue
            for k in range(order + 1 - i):
                if base[k]:
                    cur[i + k] += prev[i] * base[k]
    return powers


def sheffer_sequence(
    g_coeffs: Sequence[Rational], f_coeffs: Sequence[Rational], n: int
) -> list[tuple[Fraction, ...]]:
    r"""The Sheffer sequence ``s_0(x) .. s_n(x)`` for the pair ``(g, f)`` (exact).

    ``g`` is an invertible series (``g(0) != 0``) and ``f`` a delta series (``f(0) = 0``,
    ``f'(0) != 0``); both are ascending ordinary-power-series coefficient lists. The sequence
    is read off the generating function

    .. math::

        \sum_{n\ge 0} s_n(x)\,\frac{t^n}{n!} = \frac{1}{g(\bar f(t))}\, e^{x\,\bar f(t)},

    where ``fbar`` is the compositional inverse of ``f``. Returns a list of ascending
    ``x``-coefficient tuples. Special cases: ``f = t`` gives the Appell sequence of ``1/g``
    (see :func:`omnibias.difference.appell_sequence`); ``g = 1`` gives the associated
    (binomial-type) sequence (see :func:`associated_sequence`).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    sheffer_classify(g_coeffs, f_coeffs)  # validates the (g, f) pair, raises otherwise
    fbar = compositional_inverse(f_coeffs, n)
    a_series = series_reciprocal(compose_series(g_coeffs, fbar, n), n)  # 1 / g(fbar)
    a_coeffs = [a_series[j] if j < len(a_series) else Fraction(0) for j in range(n + 1)]
    fbar_pow = _series_powers(fbar, n)  # fbar^m coefficients, m = 0 .. n
    result: list[tuple[Fraction, ...]] = []
    for degree in range(n + 1):
        poly = [Fraction(0)] * (degree + 1)
        for m in range(degree + 1):
            acc = Fraction(0)
            for j in range(degree - m + 1):
                aj = a_coeffs[j]
                if aj:
                    acc += aj * fbar_pow[m][degree - j]
            poly[m] = acc / factorial(m)
        fact = factorial(degree)
        result.append(tuple(fact * c for c in poly))
    return result


def associated_sequence(
    f_coeffs: Sequence[Rational], n: int
) -> list[tuple[Fraction, ...]]:
    r"""The associated (binomial-type) sequence of the delta series ``f`` (exact).

    The ``g = 1`` Sheffer case: ``sum_n p_n(x) t^n/n! = exp(x fbar(t))``, so ``p_n`` satisfies
    the binomial identity ``p_n(x + y) = sum_k C(n, k) p_k(x) p_{n-k}(y)``. ``f = t`` gives the
    monomials ``x^n``; ``f = e^t - 1`` gives the falling factorials; ``f = log(1 + t)`` gives
    the Bell / Touchard polynomials (rows of Stirling numbers of the second kind).
    """
    return sheffer_sequence((1,), f_coeffs, n)


def umbral_composition(
    s_seq: Sequence[Sequence[Rational]], r_seq: Sequence[Sequence[Rational]]
) -> list[tuple[Fraction, ...]]:
    r"""Umbral composition ``(s # r)_n(x) = sum_k s_{n,k} r_k(x)`` (exact).

    ``s_seq`` is a polynomial sequence given as ascending-coefficient rows (``s_{n,k}`` is the
    ``x^k`` coefficient of ``s_n``); ``r_seq`` is another such sequence, substituted for the
    powers of ``x``. ``r_seq`` must supply a row for every degree that appears in ``s_seq``.
    For Sheffer sequences this is the group operation dual to composing the delta series.
    """
    out: list[tuple[Fraction, ...]] = []
    for row in s_seq:
        poly: list[Fraction] = [Fraction(0)]
        for k, s_nk in enumerate(row):
            if not s_nk:
                continue
            if k >= len(r_seq):
                raise ValueError(f"r_seq needs a row for degree {k}")
            r_k = r_seq[k]
            if len(r_k) > len(poly):
                poly.extend([Fraction(0)] * (len(r_k) - len(poly)))
            for j, c in enumerate(r_k):
                poly[j] += _frac(s_nk) * _frac(c)
        out.append(tuple(poly))
    return out


def shift_polynomial(coeffs: Sequence[Rational], a: Rational) -> tuple[Fraction, ...]:
    r"""Apply the shift operator ``E^a``: return the coefficients of ``P(x + a)`` (exact)."""
    c = [_frac(v) for v in coeffs]
    af = _frac(a)
    out = [Fraction(0)] * len(c)
    for i, ci in enumerate(c):
        if not ci:
            continue
        for j in range(i + 1):
            out[j] += ci * comb(i, j) * af ** (i - j)
    return tuple(out)


def delta_operator_apply(
    f_coeffs: Sequence[Rational], coeffs: Sequence[Rational]
) -> tuple[Fraction, ...]:
    r"""Apply the operator ``Q = f(D) = sum_k f_k D^k`` to a polynomial (exact on polynomials).

    ``f_coeffs`` are the ascending coefficients of the indicator series ``f`` (a delta series
    for a genuine delta operator, but any series is accepted); ``coeffs`` are the ascending
    coefficients of the polynomial. ``D`` is the ordinary derivative.
    """
    f = [_frac(v) for v in f_coeffs]
    result = [Fraction(0)] * len(coeffs)
    deriv = [_frac(v) for v in coeffs]  # D^0 p
    for fk in f:
        if fk and deriv:
            for j, c in enumerate(deriv):
                result[j] += fk * c
        deriv = _poly_derivative(deriv)
    return _trim(result)


def pincherle_derivative(f_coeffs: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""The Pincherle derivative of ``Q = f(D)``: the series of ``Q' = QX - XQ = f'(D)`` (exact).

    Returns the ascending coefficients of ``f'`` (the formal derivative of the indicator
    series). The operator identity ``[f(D), X] = f'(D)`` follows from ``[D, X] = 1``.
    """
    return tuple(_poly_derivative([_frac(v) for v in f_coeffs]))


def umbral_functional(moments: Sequence[Rational], coeffs: Sequence[Rational]) -> Fraction:
    r"""The linear functional ``<L | sum_k p_k x^k> = sum_k p_k mu_k`` (exact).

    ``moments`` are ``mu_k = <L | x^k>``; ``coeffs`` are the ascending coefficients of the
    polynomial. With ``mu_k = a^k`` this is evaluation at ``a``; with ``mu_k = k!`` it is the
    Borel/umbral evaluation. Needs a moment for every coefficient.
    """
    p = [_frac(v) for v in coeffs]
    mu = [_frac(v) for v in moments]
    if len(p) > len(mu):
        raise ValueError("need a moment mu_k for every polynomial coefficient")
    return sum((p[k] * mu[k] for k in range(len(p))), Fraction(0))


__all__ = [
    "associated_sequence",
    "delta_operator_apply",
    "pincherle_derivative",
    "sheffer_sequence",
    "shift_polynomial",
    "umbral_composition",
    "umbral_functional",
]
