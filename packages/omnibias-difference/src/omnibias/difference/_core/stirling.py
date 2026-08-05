# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Stirling numbers read off the Bell / Faa di Bruno tower (closed-form, exact).

The partial exponential Bell polynomial :math:`B_{n,k}` in
:mod:`omnibias.core.bell` -- the combinatorial engine of the closed-form
derivative tower -- specialises to both Stirling triangles:

* **Stirling second kind** ``S(n, k) = B_{n,k}(1, 1, ..., 1)`` (evaluate at all
  ones): the number of partitions of ``n`` labelled items into ``k`` blocks. Its
  row-sum is the Bell number.
* **Stirling first kind (signed)** ``s(n, k) = B_{n,k}(0!, -1!, 2!, -3!, ...)``
  (evaluate at ``x_i = (-1)^{i-1} (i-1)!``); the **unsigned**
  ``c(n, k) = |s(n, k)| = B_{n,k}(0!, 1!, 2!, ...)`` counts permutations of ``n``
  with ``k`` cycles.

The Stirling numbers are the change-of-basis matrices between the monomials and
the falling/rising factorials -- the heart of umbral finite-difference calculus:
``x^n = sum_k S(n,k) (x)_k`` and ``(x)_n = sum_k s(n,k) x^k``.
"""

from __future__ import annotations

from collections.abc import Callable
from math import factorial

from omnibias.core.bell import bell_number as _bell_number
from omnibias.core.bell import bell_partial


def bell_number(n: int) -> int:
    """The ``n``-th Bell number ``sum_k S(n, k)`` (re-exported from ``omnibias.core.bell``)."""
    return _bell_number(n)


def _bell_partial_eval(n: int, k: int, x: Callable[[int], int]) -> int:
    """Evaluate ``B_{n,k}`` at ``x_i = x(i)`` (1-indexed) in exact integer arithmetic."""
    total = 0
    for exps, coeff in bell_partial(n, k).items():
        term = coeff
        for i, e in enumerate(exps, start=1):
            if e:
                term *= x(i) ** e
        total += term
    return total


def stirling_second(n: int, k: int) -> int:
    """Stirling number of the second kind ``S(n, k) = B_{n,k}(1, ..., 1)``."""
    if n < 0 or k < 0:
        raise ValueError(f"n, k must be >= 0, got n={n}, k={k}")
    return _bell_partial_eval(n, k, lambda _i: 1)


def stirling_first_signed(n: int, k: int) -> int:
    """Signed Stirling number of the first kind ``s(n, k) = B_{n,k}((-1)^{i-1}(i-1)!)``."""
    if n < 0 or k < 0:
        raise ValueError(f"n, k must be >= 0, got n={n}, k={k}")
    return _bell_partial_eval(n, k, lambda i: (-1) ** (i - 1) * factorial(i - 1))


def stirling_first_unsigned(n: int, k: int) -> int:
    """Unsigned Stirling number of the first kind ``c(n, k) = |s(n, k)| = B_{n,k}((i-1)!)``."""
    if n < 0 or k < 0:
        raise ValueError(f"n, k must be >= 0, got n={n}, k={k}")
    return _bell_partial_eval(n, k, lambda i: factorial(i - 1))


def stirling_second_row(n: int) -> tuple[int, ...]:
    """The row ``(S(n, 0), ..., S(n, n))``."""
    return tuple(stirling_second(n, k) for k in range(n + 1))


def stirling_first_signed_row(n: int) -> tuple[int, ...]:
    """The row ``(s(n, 0), ..., s(n, n))`` (signed first kind)."""
    return tuple(stirling_first_signed(n, k) for k in range(n + 1))


def falling_factorial_coeffs(n: int) -> tuple[int, ...]:
    r"""Coefficients of the falling factorial ``(x)_n = x (x-1) ... (x-n+1)``.

    Returns ``(c_0, ..., c_n)`` with ``(x)_n = sum_k c_k x^k``; these are the signed
    Stirling numbers of the first kind ``c_k = s(n, k)``. Computed directly by the
    product ``prod_{j=0}^{n-1} (x - j)`` (an independent cross-check of
    :func:`stirling_first_signed`).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    poly = [1]
    for j in range(n):
        nxt = [0] * (len(poly) + 1)
        for k, c in enumerate(poly):
            nxt[k + 1] += c
            nxt[k] -= j * c
        poly = nxt
    return tuple(poly)


def rising_factorial_coeffs(n: int) -> tuple[int, ...]:
    r"""Coefficients of the rising factorial ``x^(n) = x (x+1) ... (x+n-1)``.

    Returns ``(c_0, ..., c_n)`` with ``x^(n) = sum_k c_k x^k``; these are the
    unsigned Stirling numbers of the first kind ``c_k = |s(n, k)|``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    poly = [1]
    for j in range(n):
        nxt = [0] * (len(poly) + 1)
        for k, c in enumerate(poly):
            nxt[k + 1] += c
            nxt[k] += j * c
        poly = nxt
    return tuple(poly)


__all__ = [
    "bell_number",
    "falling_factorial_coeffs",
    "rising_factorial_coeffs",
    "stirling_first_signed",
    "stirling_first_signed_row",
    "stirling_first_unsigned",
    "stirling_second",
    "stirling_second_row",
]
