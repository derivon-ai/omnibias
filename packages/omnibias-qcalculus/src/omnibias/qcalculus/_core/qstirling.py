# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""q-Stirling numbers and the q-factorial bases (Carlitz convention, exact).

The q-deformation of the two Stirling triangles -- the change-of-basis matrices between the
monomials and the q-falling / q-rising factorials, i.e. the combinatorial heart of the
q-umbral calculus:

* **q-Stirling second kind** ``S_q(n, k) = S_q(n-1, k-1) + [k]_q S_q(n-1, k)``;
* **q-Stirling first kind (unsigned)** ``c_q(n, k) = c_q(n-1, k-1) + [n-1]_q c_q(n-1, k)``,
  signed ``s_q(n, k) = (-1)^{n-k} c_q(n, k)``.

The two kinds are inverse matrices (``sum_j s_q(n, j) S_q(j, m) = delta_{n,m}``), and the
q-factorial polynomials read straight off the first kind:
``[x]_{n,q} = prod_{i=0}^{n-1}(x - [i]_q) = sum_k s_q(n, k) x^k`` (falling) and
``prod_{i=0}^{n-1}(x + [i]_q) = sum_k c_q(n, k) x^k`` (rising).

Everything is exact :class:`~fractions.Fraction` arithmetic at a numeric ``q``. As
``q -> 1`` every object reduces to its ordinary Stirling / factorial counterpart -- the
**distinct** ``q -> 1`` limit, never the ``delta -> 0`` founding collapse of
:mod:`omnibias.difference` nor the ``beta -> inf`` feasibility penalty.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.qcalculus._core.qnumbers import q_bracket

Rational = Fraction | int


def _second_table(n: int, q: Fraction) -> list[list[Fraction]]:
    table = [[Fraction(0)] * (n + 1) for _ in range(n + 1)]
    table[0][0] = Fraction(1)
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            table[i][j] = table[i - 1][j - 1] + q_bracket(j, q) * table[i - 1][j]
    return table


def _first_unsigned_table(n: int, q: Fraction) -> list[list[Fraction]]:
    table = [[Fraction(0)] * (n + 1) for _ in range(n + 1)]
    table[0][0] = Fraction(1)
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            table[i][j] = table[i - 1][j - 1] + q_bracket(i - 1, q) * table[i - 1][j]
    return table


def q_stirling_second(n: int, k: int, q: Rational) -> Fraction:
    r"""q-Stirling number of the second kind ``S_q(n, k)`` (exact; ``-> S(n, k)`` as ``q -> 1``)."""
    if n < 0 or k < 0:
        raise ValueError(f"n, k must be >= 0, got n={n}, k={k}")
    if k > n:
        return Fraction(0)
    return _second_table(n, Fraction(q))[n][k]


def q_stirling_first_unsigned(n: int, k: int, q: Rational) -> Fraction:
    r"""Unsigned q-Stirling number of the first kind ``c_q(n, k)`` (exact; ``-> c(n, k)`` at ``q=1``)."""
    if n < 0 or k < 0:
        raise ValueError(f"n, k must be >= 0, got n={n}, k={k}")
    if k > n:
        return Fraction(0)
    return _first_unsigned_table(n, Fraction(q))[n][k]


def q_stirling_first_signed(n: int, k: int, q: Rational) -> Fraction:
    r"""Signed q-Stirling number of the first kind ``s_q(n, k) = (-1)^{n-k} c_q(n, k)`` (exact)."""
    return Fraction((-1) ** (n - k)) * q_stirling_first_unsigned(n, k, q)


def q_stirling_second_row(n: int, q: Rational) -> tuple[Fraction, ...]:
    r"""The row ``(S_q(n, 0), ..., S_q(n, n))`` (q-Stirling second kind)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return tuple(_second_table(n, Fraction(q))[n])


def q_stirling_first_signed_row(n: int, q: Rational) -> tuple[Fraction, ...]:
    r"""The row ``(s_q(n, 0), ..., s_q(n, n))`` (signed q-Stirling first kind)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    row = _first_unsigned_table(n, Fraction(q))[n]
    return tuple(Fraction((-1) ** (n - k)) * c for k, c in enumerate(row))


def q_falling_factorial_coeffs(n: int, q: Rational) -> tuple[Fraction, ...]:
    r"""Coefficients of the q-falling factorial ``[x]_{n,q} = prod_{i=0}^{n-1}(x - [i]_q)`` (exact).

    Returns ``(c_0, ..., c_n)`` with ``[x]_{n,q} = sum_k c_k x^k``. Computed directly by the
    product (an independent cross-check of :func:`q_stirling_first_signed_row`); at ``q -> 1``
    this is the ordinary falling factorial ``(x)_n``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    qq = Fraction(q)
    poly = [Fraction(1)]
    for i in range(n):
        bracket = q_bracket(i, qq)
        nxt = [Fraction(0)] * (len(poly) + 1)
        for k, c in enumerate(poly):
            nxt[k + 1] += c
            nxt[k] -= bracket * c
        poly = nxt
    return tuple(poly)


def q_rising_factorial_coeffs(n: int, q: Rational) -> tuple[Fraction, ...]:
    r"""Coefficients of the q-rising factorial ``prod_{i=0}^{n-1}(x + [i]_q)`` (exact).

    Returns ``(c_0, ..., c_n)``; these are the unsigned q-Stirling first-kind row
    ``c_q(n, .)``. At ``q -> 1`` this is the ordinary rising factorial ``x^(n)``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    qq = Fraction(q)
    poly = [Fraction(1)]
    for i in range(n):
        bracket = q_bracket(i, qq)
        nxt = [Fraction(0)] * (len(poly) + 1)
        for k, c in enumerate(poly):
            nxt[k + 1] += c
            nxt[k] += bracket * c
        poly = nxt
    return tuple(poly)


__all__ = [
    "q_falling_factorial_coeffs",
    "q_rising_factorial_coeffs",
    "q_stirling_first_signed",
    "q_stirling_first_signed_row",
    "q_stirling_first_unsigned",
    "q_stirling_second",
    "q_stirling_second_row",
]
