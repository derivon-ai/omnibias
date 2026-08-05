# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact-integer derivative-tower polynomial coefficients + interval Horner.

:mod:`omnibias.core.polynomials` returns the activation derivative-tower
coefficients as ``float`` tuples; they are mathematically *integers* (Eulerian /
Legendre / Hermite recurrences) but lose exactness once ``n`` is large enough
that ``n!``-scale values exceed ``2**53``.  For a computer-assisted proof the
coefficients must stay exact, so this module reproduces the identical
recurrences in Python ``int`` arithmetic.

These are bit-for-bit the same sequences as the float versions wherever the
float versions are still exact -- :mod:`tests` asserts that overlap -- so the
verified tower composes with the float fast-paths without any drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import comb

from omnibias.core.verified.interval import Interval

#: Bound on each memo. These generators are keyed on a caller-supplied order, and
#: an unbounded memo would let one hostile or mistyped call pin arbitrarily much
#: memory for the life of the process.
_CACHE_SIZE: int = 256


@lru_cache(maxsize=_CACHE_SIZE)
def sigmoid_poly_coeffs_exact(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of ``P_n(s) = sigma^(n)(z)``, ``s=sigmoid(z)``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    # Iterative, so a tall tower costs memory but never a RecursionError.
    coeffs = [0, 1]
    for _ in range(n):
        deriv = [k * coeffs[k] for k in range(1, len(coeffs))]
        out = [0] * (len(deriv) + 2)
        for i, c in enumerate(deriv):
            out[i + 1] += c
            out[i + 2] -= c
        coeffs = out
    return tuple(coeffs)


@lru_cache(maxsize=_CACHE_SIZE)
def tanh_poly_coeffs_exact(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of ``T_n(t) = tanh^(n)(z)``, ``t=tanh(z)``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    # Iterative, so a tall tower costs memory but never a RecursionError.
    coeffs = [0, 1]
    for _ in range(n):
        deriv = [k * coeffs[k] for k in range(1, len(coeffs))]
        out = [0] * (len(deriv) + 2)
        for i, c in enumerate(deriv):
            out[i] += c
            out[i + 2] -= c
        coeffs = out
    return tuple(coeffs)


@lru_cache(maxsize=_CACHE_SIZE)
def bernoulli_number_exact(n: int) -> Fraction:
    r"""The ``n``-th Bernoulli number ``B_n`` as an exact :class:`~fractions.Fraction`.

    Read off the closed-form ``tanh`` tower: the odd derivative ``tanh^(2m-1)(0)``
    is the signed tangent number ``tanh_poly_coeffs_exact(2m-1)[0]``, and the Taylor
    series ``tanh(x) = sum_m 2^{2m}(2^{2m}-1) B_{2m}/(2m)! x^{2m-1}`` gives

    .. math::

        B_{2m} = \tanh^{(2m-1)}(0)\;\frac{2m}{2^{2m}\,(2^{2m}-1)}.

    Convention ``B_0 = 1``, ``B_1 = -1/2`` (matches ``mpmath.bernoulli``), and
    ``B_{2m+1} = 0`` for ``m >= 1``. This is the shared source of truth for the
    Bernoulli numbers used by :mod:`omnibias.core.verified.euler_maclaurin` and
    re-exported by :mod:`omnibias.difference`.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return Fraction(1)
    if n == 1:
        return Fraction(-1, 2)
    if n % 2 == 1:
        return Fraction(0)
    m = n // 2
    tangent = tanh_poly_coeffs_exact(2 * m - 1)[0]  # tanh^(2m-1)(0), an exact integer
    return Fraction(tangent) * Fraction(2 * m) / Fraction(2 ** (2 * m) * (2 ** (2 * m) - 1))


@lru_cache(maxsize=_CACHE_SIZE)
def euler_number_exact(n: int) -> int:
    r"""The ``n``-th Euler (secant) number ``E_n`` as an exact integer.

    Read off the closed-form ``sech`` tower: ``sech^(n)(0) = Q_n(0) sech(0) =
    Q_n(0)``, so ``E_n`` is the constant term
    :func:`sech_poly_coeffs_exact`\ ``(n)[0]``. Convention ``E_0 = 1``,
    ``E_{2m+1} = 0``, ``E_2 = -1``, ``E_4 = 5``, ``E_6 = -61`` (matches
    ``mpmath.eulernum``). This is the shared source of truth for the Euler numbers
    used by the Dirichlet ``beta`` special values and re-exported by
    :mod:`omnibias.difference`.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return int(sech_poly_coeffs_exact(n)[0])


@lru_cache(maxsize=_CACHE_SIZE)
def bernoulli_polynomial_exact(n: int, x: Fraction | int) -> Fraction:
    r"""The Bernoulli polynomial ``B_n(x)`` at a rational ``x`` as an exact Fraction.

    .. math:: B_n(x) = \sum_{k=0}^{n} \binom{n}{k} B_k\, x^{n-k},

    with the ``B_1 = -1/2`` convention (matches ``mpmath.bernpoly``). Hence
    ``B_n(0) = B_n``, ``B_1(x) = x - 1/2``, ``B_2(x) = x^2 - x + 1/6``. This is the
    shared source for the Hurwitz-zeta negative-integer special values
    ``zeta(-n, a) = -B_{n+1}(a)/(n+1)`` and the generalized Bernoulli numbers below.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    xf = Fraction(x)
    total = Fraction(0)
    for k in range(n + 1):
        total += comb(n, k) * bernoulli_number_exact(k) * xf ** (n - k)
    return total


def generalized_bernoulli_exact(n: int, chi: Sequence[int | Fraction]) -> Fraction:
    r"""Generalized Bernoulli number ``B_{n,chi}`` for a *real* Dirichlet character.

    ``chi`` is the periodic value table of a real character modulo ``f = len(chi)``:
    ``chi[a]`` is ``chi(a)`` for ``a = 0..f-1`` (``chi[0] = chi(0)``, typically ``0``
    for a non-principal character). Then

    .. math:: B_{n,\chi} = f^{\,n-1} \sum_{a=1}^{f} \chi(a)\, B_n(a/f),

    an exact :class:`~fractions.Fraction` (real characters give rational values).
    The Dirichlet ``L`` special values read this off as ``L(1-n, chi) = -B_{n,chi}/n``
    (``n >= 1``). ``chi = (0, 1)`` (trivial character mod 1 shifted) recovers
    ``B_{n,chi} = B_n``.
    """
    f = len(chi)
    if f < 1:
        raise ValueError("character table must be non-empty")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    total = Fraction(0)
    for a in range(1, f + 1):
        chi_a = Fraction(chi[a % f])
        if chi_a == 0:
            continue
        total += chi_a * bernoulli_polynomial_exact(n, Fraction(a, f))
    return Fraction(f) ** (n - 1) * total


@lru_cache(maxsize=_CACHE_SIZE)
def hermite_coeffs_exact(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of the probabilist's Hermite polynomial ``He_n``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    if n == 0:
        return (1,)
    # Iterative, so a tall tower costs memory but never a RecursionError.
    prev2: tuple[int, ...] = (1,)
    prev1: tuple[int, ...] = (0, 1)
    for m in range(2, n + 1):
        out = [0] * (m + 1)
        for k, c in enumerate(prev1):
            out[k + 1] += c
        for k, c in enumerate(prev2):
            out[k] -= (m - 1) * c
        prev2, prev1 = prev1, tuple(out)
    return prev1


@lru_cache(maxsize=_CACHE_SIZE)
def sech_poly_coeffs_exact(n: int) -> tuple[int, ...]:
    r"""Exact integer coefficients of ``Q_n(t)`` with ``sech^(n)(z) = Q_n(t) sech(z)``.

    Same tanh/sech Riccati recurrence as
    :func:`omnibias.core.polynomials.sech_polynomial_coeffs` in exact ``int``
    arithmetic: ``Q_0 = 1``, ``Q_{n+1}(t) = (1 - t^2) Q_n'(t) - t Q_n(t)``. The
    constant term ``Q_n(0)`` is the exact ``n``-th Euler (secant) number ``E_n``.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    # Iterative, so a tall tower costs memory but never a RecursionError.
    coeffs: tuple[int, ...] = (1,)
    for _ in range(n):
        deriv = [k * coeffs[k] for k in range(1, len(coeffs))]
        out = [0] * (len(coeffs) + 1)
        for i, c in enumerate(deriv):
            out[i] += c
            out[i + 2] -= c
        for i, c in enumerate(coeffs):
            out[i + 1] -= c
        coeffs = tuple(out)
    return coeffs


def horner_interval(coeffs: Sequence[int], x: Interval) -> Interval:
    """Rigorously evaluate ``sum_k coeffs[k] * x**k`` by interval Horner."""
    if not coeffs:
        return Interval.point(0.0)
    acc = Interval.from_rational(int(coeffs[-1]))
    for c in reversed(coeffs[:-1]):
        acc = acc * x + Interval.from_rational(int(c))
    return acc


__all__ = [
    "bernoulli_number_exact",
    "bernoulli_polynomial_exact",
    "euler_number_exact",
    "generalized_bernoulli_exact",
    "hermite_coeffs_exact",
    "horner_interval",
    "sech_poly_coeffs_exact",
    "sigmoid_poly_coeffs_exact",
    "tanh_poly_coeffs_exact",
]
