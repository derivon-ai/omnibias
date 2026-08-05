# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Bernoulli numbers read off the closed-form ``tanh`` tower (exact rationals).

The odd derivatives of ``tanh`` at the origin are the (signed) **tangent
numbers** ``tanh^(2m-1)(0)``, read straight off the constant term of the exact
``tanh`` derivative-tower polynomial
(:func:`omnibias.core.verified.coeffs.tanh_poly_coeffs_exact`). The Taylor series
``tanh(x) = sum_{m>=1} 2^{2m}(2^{2m}-1) B_{2m}/(2m)! x^{2m-1}`` then gives the
even Bernoulli numbers exactly:

.. math::

    B_{2m} = \tanh^{(2m-1)}(0)\;\frac{2m}{2^{2m}\,(2^{2m}-1)}.

Convention: ``B_0 = 1``, ``B_1 = -1/2`` (the "first Bernoulli number"; matches
``mpmath.bernoulli``), and ``B_{2m+1} = 0`` for ``m >= 1``.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from omnibias.core.verified.coeffs import bernoulli_number_exact


def bernoulli_number(n: int) -> Fraction:
    """The ``n``-th Bernoulli number ``B_n`` (exact ``Fraction``; ``B_1 = -1/2``).

    A thin re-export of :func:`omnibias.core.verified.coeffs.bernoulli_number_exact`
    -- the shared source of truth also consumed by the certified Euler-Maclaurin
    engine -- so the Bernoulli numbers are defined once (off the ``tanh`` tower) and
    never forked per package.
    """
    return bernoulli_number_exact(n)


def bernoulli_polynomial(n: int) -> tuple[Fraction, ...]:
    r"""Coefficients of the Bernoulli polynomial ``B_n(x) = sum_k C(n,k) B_{n-k} x^k``.

    Returns ``(c_0, ..., c_n)`` with ``B_n(x) = sum_k c_k x^k``. This is an Appell
    sequence: ``B_n'(x) = n B_{n-1}(x)`` and ``B_n(0) = B_n``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return tuple(Fraction(comb(n, k)) * bernoulli_number(n - k) for k in range(n + 1))


def power_sum_coeffs(p: int) -> tuple[Fraction, ...]:
    r"""Faulhaber coefficients of ``S_p(N) = sum_{i=0}^{N-1} i^p`` as a polynomial in ``N``.

    Returns ``(c_0, ..., c_{p+1})`` with ``S_p(N) = sum_j c_j N^j``, via
    ``S_p(N) = (B_{p+1}(N) - B_{p+1}) / (p + 1)``. (Sum runs to ``N - 1``, the
    convention matched to ``B_1 = -1/2``.)
    """
    if p < 0:
        raise ValueError(f"p must be >= 0, got {p}")
    out = [Fraction(0)] * (p + 2)
    for k in range(1, p + 2):
        out[k] = Fraction(comb(p + 1, k)) * bernoulli_number(p + 1 - k) / Fraction(p + 1)
    return tuple(out)


__all__ = [
    "bernoulli_number",
    "bernoulli_polynomial",
    "power_sum_coeffs",
]
