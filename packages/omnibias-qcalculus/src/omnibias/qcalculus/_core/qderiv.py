# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Jackson calculus: the q-derivative and q-integral.

The q-derivative (Jackson derivative)

.. math::

    (D_q f)(x) = \frac{f(qx) - f(x)}{(q - 1)x} \qquad (x \neq 0),

and its inverse, the q-integral (Jackson integral). Two honesty registers:

* **closed-form / exact** -- on *polynomials* the operators are exact: ``D_q x^n =
  [n]_q x^{n-1}`` and ``\int_0^x t^n d_q t = x^{n+1}/[n+1]_q``, in rational arithmetic.
* **numerical** -- on a general callable, ``q_derivative`` is the difference quotient and
  ``q_integral`` is the truncated Jackson sum ``(1-q) x sum_{k>=0} q^k f(x q^k)``.

As ``q -> 1`` the q-derivative recovers the ordinary derivative ``f'`` (the distinct
``q -> 1`` limit, not the ``delta -> 0`` founding collapse).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from omnibias.qcalculus._core.qnumbers import q_bracket

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def q_derivative(f: Callable[[float], float], x: float, q: float) -> float:
    r"""Numerical Jackson q-derivative ``(f(qx) - f(x)) / ((q-1)x)`` (``x != 0``, ``q != 1``).

    At ``x = 0`` the q-derivative is the ordinary ``f'(0)``, which a difference quotient
    cannot form from one sample -- pass a non-zero ``x`` (or use the polynomial operator).
    """
    if q == 1.0:
        raise ValueError("q_derivative needs q != 1 (the q -> 1 limit is the ordinary derivative)")
    if x == 0.0:
        raise ValueError("numerical q_derivative is undefined at x = 0; use q_derivative_poly")
    return (f(q * x) - f(x)) / ((q - 1.0) * x)


def q_derivative_poly(coeffs: Sequence[Rational], q: Rational) -> tuple[Fraction, ...]:
    r"""Exact q-derivative of a polynomial ``sum_i c_i x^i``: ``D_q x^i = [i]_q x^{i-1}``.

    Returns the ascending coefficients of the degree-lowered polynomial (empty for a
    constant). Exact rational arithmetic.
    """
    c = [_frac(v) for v in coeffs]
    if len(c) <= 1:
        return ()
    return tuple(c[i] * q_bracket(i, q) for i in range(1, len(c)))


def q_antiderivative_poly(coeffs: Sequence[Rational], q: Rational) -> tuple[Fraction, ...]:
    r"""Exact q-antiderivative of ``sum_i c_i x^i``: ``\int x^i d_q x = x^{i+1}/[i+1]_q``.

    Returns the ascending coefficients (constant term ``0``); the exact inverse of
    :func:`q_derivative_poly` up to the integration constant.
    """
    c = [_frac(v) for v in coeffs]
    out = [Fraction(0)]
    for i, ci in enumerate(c):
        out.append(ci / q_bracket(i + 1, q))
    return tuple(out)


def q_integral(
    f: Callable[[float], float], a: float, b: float, q: float, *, terms: int = 200
) -> float:
    r"""Numerical Jackson q-integral ``\int_a^b f(x) d_q x`` for ``0 < q < 1``.

    Uses ``\int_0^c f d_q x = (1 - q) c sum_{k>=0} q^k f(c q^k)`` (a geometric point set
    ``c q^k -> 0``) and ``\int_a^b = \int_0^b - \int_0^a``. ``terms`` truncates the sum;
    for a bounded ``f`` the omitted tail is ``O(q^{terms})``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"numerical q_integral needs 0 < q < 1, got q={q}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")

    def jackson_0(c: float) -> float:
        if c == 0.0:
            return 0.0
        total = 0.0
        power = 1.0  # q^k
        for _ in range(terms):
            total += power * f(c * power)
            power *= q
        return (1.0 - q) * c * total

    return jackson_0(b) - jackson_0(a)


__all__ = [
    "q_antiderivative_poly",
    "q_derivative",
    "q_derivative_poly",
    "q_integral",
]
