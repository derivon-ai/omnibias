# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact q-combinatorics: q-numbers, q-factorial, Gaussian binomial, q-Pochhammer.

The building blocks of q-calculus, in **exact** arithmetic (closed-form):

* at a numeric ``q`` (a :class:`~fractions.Fraction`) everything is an exact rational;
* as a **polynomial in ``q``**, the q-number ``[n]_q = 1 + q + ... + q^{n-1}`` and the
  Gaussian binomial ``[n choose k]_q`` are integer polynomials (the latter via the
  q-Pascal recurrence ``[n,k] = [n-1,k-1] + q^k [n-1,k]``).

Every object reduces to its ordinary-calculus counterpart as ``q -> 1``
(``[n]_q -> n``, ``[n choose k]_q -> C(n, k)``) -- the **distinct** ``q -> 1`` limit,
never the ``delta -> 0`` founding collapse of :mod:`omnibias.difference` nor the
``beta -> inf`` feasibility penalty.
"""

from __future__ import annotations

from fractions import Fraction

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def q_bracket(n: int, q: Rational) -> Fraction:
    r"""The q-number ``[n]_q = (1 - q^n)/(1 - q) = 1 + q + ... + q^{n-1}`` (exact, ``n >= 0``).

    The summation form is used so ``q = 1`` gives ``n`` with no division by zero.
    """
    if n < 0:
        raise ValueError(f"q_bracket needs n >= 0, got {n}")
    qq = _frac(q)
    total = Fraction(0)
    power = Fraction(1)
    for _ in range(n):
        total += power
        power *= qq
    return total


def q_bracket_poly(n: int) -> tuple[int, ...]:
    r"""The q-number ``[n]_q`` as an integer polynomial in ``q`` (``1, 1, ..., 1``; ``n`` ones)."""
    if n < 0:
        raise ValueError(f"q_bracket_poly needs n >= 0, got {n}")
    return tuple(1 for _ in range(n))


def q_factorial(n: int, q: Rational) -> Fraction:
    r"""The q-factorial ``[n]_q! = prod_{k=1}^{n} [k]_q`` (exact, ``n >= 0``)."""
    if n < 0:
        raise ValueError(f"q_factorial needs n >= 0, got {n}")
    result = Fraction(1)
    for k in range(1, n + 1):
        result *= q_bracket(k, q)
    return result


def q_binomial(n: int, k: int, q: Rational) -> Fraction:
    r"""The Gaussian / q-binomial ``[n choose k]_q = [n]!/([k]![n-k]!)`` (exact).

    Zero for ``k < 0`` or ``k > n``. At ``q = 1`` this is the ordinary ``C(n, k)``.
    """
    if k < 0 or k > n:
        return Fraction(0)
    return q_factorial(n, q) / (q_factorial(k, q) * q_factorial(n - k, q))


def q_binomial_poly(n: int, k: int) -> tuple[int, ...]:
    r"""The Gaussian binomial ``[n choose k]_q`` as an integer polynomial in ``q``.

    Computed by the q-Pascal recurrence ``[n,k]_q = [n-1,k-1]_q + q^k [n-1,k]_q`` (so the
    coefficients are the *non-negative* integers counting subspaces / inversions), which
    avoids any division. Returns the ascending coefficient tuple.
    """
    if k < 0 or k > n:
        return ()
    if k == 0 or k == n:
        return (1,)
    below = q_binomial_poly(n - 1, k - 1)  # [n-1, k-1]
    right = q_binomial_poly(n - 1, k)  # q^k * [n-1, k]
    size = max(len(below), k + len(right))
    coeffs = [0] * size
    for i, c in enumerate(below):
        coeffs[i] += c
    for i, c in enumerate(right):
        coeffs[i + k] += c
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return tuple(coeffs)


def q_pochhammer(a: Rational, q: Rational, n: int) -> Fraction:
    r"""The finite q-Pochhammer symbol ``(a; q)_n = prod_{k=0}^{n-1} (1 - a q^k)`` (exact)."""
    if n < 0:
        raise ValueError(f"q_pochhammer needs n >= 0, got {n}")
    af, qf = _frac(a), _frac(q)
    result = Fraction(1)
    power = Fraction(1)  # q^k
    for _ in range(n):
        result *= 1 - af * power
        power *= qf
    return result


__all__ = [
    "q_binomial",
    "q_binomial_poly",
    "q_bracket",
    "q_bracket_poly",
    "q_factorial",
    "q_pochhammer",
]
