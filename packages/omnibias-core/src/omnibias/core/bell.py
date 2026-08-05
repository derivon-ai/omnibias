# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bell polynomials and Faà di Bruno coefficients (pure Python).

This module supplies the combinatorial core of the *multi-layer jet
composition* primitive. It is backend-agnostic (no torch / jax / numpy), in the
same spirit as :mod:`omnibias.core.polynomials`: only integer combinatorics live
here, so the torch and jax jet kernels are bit-identical by construction.

Faà di Bruno's formula expresses the ``n``-th derivative of a composition
``b(t) = sigma(u(t))`` in terms of the derivatives of ``sigma`` and ``u``:

.. math::

    b^{(n)} = \\sum_{k=1}^{n} \\sigma^{(k)}\\!\\big(u^{(0)}\\big)\\,
        B_{n,k}\\!\\big(u^{(1)}, u^{(2)}, \\dots, u^{(n-k+1)}\\big),

where :math:`B_{n,k}` is the *partial (incomplete) exponential Bell polynomial*

.. math::

    B_{n,k}(x_1,\\dots,x_{n-k+1}) = \\sum \\frac{n!}{j_1!\\,j_2!\\cdots}\\,
        \\prod_{i\\ge 1}\\Big(\\frac{x_i}{i!}\\Big)^{j_i},

the sum taken over all non-negative integer sequences :math:`(j_i)` with
:math:`\\sum_i j_i = k` and :math:`\\sum_i i\\,j_i = n`. Each such sequence is a
partition of ``n`` into exactly ``k`` parts; its coefficient
:math:`n! / \\big(\\prod_i (i!)^{j_i}\\, j_i!\\big)` is an integer.

Representation
--------------
A Bell polynomial is returned as a ``dict`` mapping an *exponent tuple*
``e = (e_1, ..., e_n)`` (the exponent of :math:`x_i = u^{(i)}`) to its integer
coefficient. Trailing zeros are kept so every key has length ``n`` and the
partial polynomials for different ``k`` share the same key space, making
:func:`bell_complete` a straight merge.

The heavy combinatorics are cached as immutable tuples of ``(exps, coeff)``
pairs; the public functions return fresh ``dict`` / ``list`` copies so callers
may mutate them safely.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from math import factorial

ExpKey = tuple[int, ...]

#: Ceiling on the order accepted by the *partition-enumerating* routines
#: (:func:`bell_partial`, :func:`bell_complete`, :func:`faa_di_bruno_terms`).
#: Their output has ``p(n)`` terms, which is exponential -- ``p(64)`` is already
#: 1.7 million and ``p(100)`` is 1.9e8 -- so an unbounded order is a memory
#: exhaustion vector rather than a slow answer. Real towers use single digits.
MAX_BELL_ORDER: int = 64

#: Ceiling on :func:`bell_number`, which uses the ``O(n^2)`` Bell triangle rather
#: than enumerating partitions and so tolerates a far higher order.
MAX_BELL_NUMBER_ORDER: int = 4096

#: Bound on each memo: unbounded memoisation keyed on a caller-supplied order
#: lets one hostile or mistyped call pin arbitrarily much memory for the life of
#: the process.
_CACHE_SIZE: int = 256


def _check_bell_order(n: int, ceiling: int = MAX_BELL_ORDER) -> None:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n > ceiling:
        raise ValueError(
            f"n must be <= {ceiling}, got {n}; enumerating the partitions of a "
            "larger n exhausts memory (see MAX_BELL_ORDER)"
        )


def _partitions_into_k(n: int, k: int) -> Iterator[tuple[int, ...]]:
    """Yield non-increasing tuples of ``k`` positive ints summing to ``n``."""
    if k == 0:
        if n == 0:
            yield ()
        return
    if k > n:
        return

    def rec(remaining: int, parts_left: int, max_part: int) -> Iterator[tuple[int, ...]]:
        if parts_left == 0:
            if remaining == 0:
                yield ()
            return
        # Largest first part keeps the remaining ``parts_left - 1`` parts >= 1.
        hi = min(max_part, remaining - (parts_left - 1))
        for first in range(hi, 0, -1):
            for rest in rec(remaining - first, parts_left - 1, first):
                yield (first, *rest)

    yield from rec(n, k, n)


def _exps(parts: tuple[int, ...], n: int) -> ExpKey:
    """Exponent tuple ``(e_1, ..., e_n)`` of a partition (``e_i`` = #parts == i)."""
    e = [0] * n
    for p in parts:
        e[p - 1] += 1
    return tuple(e)


def _coeff(parts: tuple[int, ...], n: int) -> int:
    """Bell coefficient ``n! / prod_i ((i!)^{j_i} * j_i!)`` for one partition."""
    mult = Counter(parts)
    c = factorial(n)
    for size, j in mult.items():
        c //= (factorial(size) ** j) * factorial(j)
    return c


@lru_cache(maxsize=_CACHE_SIZE)
def _bell_partial_pairs(n: int, k: int) -> tuple[tuple[ExpKey, int], ...]:
    _check_bell_order(n)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    pairs: list[tuple[ExpKey, int]] = []
    for parts in _partitions_into_k(n, k):
        pairs.append((_exps(parts, n), _coeff(parts, n)))
    pairs.sort()
    return tuple(pairs)


@lru_cache(maxsize=_CACHE_SIZE)
def _bell_complete_pairs(n: int) -> tuple[tuple[ExpKey, int], ...]:
    _check_bell_order(n)
    acc: dict[ExpKey, int] = {}
    for k in range(n + 1):
        for e, c in _bell_partial_pairs(n, k):
            acc[e] = acc.get(e, 0) + c
    return tuple(sorted(acc.items()))


@lru_cache(maxsize=_CACHE_SIZE)
def _faa_di_bruno_pairs(n: int) -> tuple[tuple[int, ExpKey, int], ...]:
    _check_bell_order(n)
    terms: list[tuple[int, ExpKey, int]] = []
    for k in range(1, n + 1):
        for e, c in _bell_partial_pairs(n, k):
            terms.append((k, e, c))
    return tuple(terms)


def bell_partial(n: int, k: int) -> dict[ExpKey, int]:
    """Partial exponential Bell polynomial :math:`B_{n,k}`.

    Returns a ``dict`` mapping an exponent tuple ``(e_1, ..., e_n)`` (the
    exponent of ``x_i``) to its integer coefficient. ``B_{0,0} = 1`` (key
    ``()``); ``B_{n,k} = 0`` (empty dict) when ``k > n`` or when exactly one of
    ``n``/``k`` is zero.

    Examples
    --------
    ``B_{n,1}`` is the single monomial ``x_n`` with coefficient 1;
    ``B_{n,n}`` is ``x_1^n`` with coefficient 1.
    """
    return dict(_bell_partial_pairs(n, k))


def bell_complete(n: int) -> dict[ExpKey, int]:
    """Complete exponential Bell polynomial :math:`B_n = \\sum_{k} B_{n,k}`.

    Returns a ``dict`` keyed by exponent tuple of length ``n``. Evaluated at
    ``x_i = 1`` for all ``i`` it equals the Bell number ``Bell(n)``
    (see :func:`bell_number`).
    """
    return dict(_bell_complete_pairs(n))


@lru_cache(maxsize=_CACHE_SIZE)
def bell_number(n: int) -> int:
    """The ``n``-th Bell number ``Bell(n) = B_n(1, 1, ..., 1)``.

    Computed by the **Bell triangle** (Aitken's array): ``O(n^2)`` exact
    big-integer additions. This returns the identical integer as
    ``sum(bell_complete(n).values())`` but without enumerating the ``p(n)``
    partition terms of the complete Bell polynomial (which is exponential in
    ``n`` -- ``p(100) ~ 1.9e8``), so it stays fast for large ``n``.
    """
    _check_bell_order(n, MAX_BELL_NUMBER_ORDER)
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for value in row:
            nxt.append(nxt[-1] + value)
        row = nxt
    return row[0]


def faa_di_bruno_terms(n: int) -> list[tuple[int, ExpKey, int]]:
    """Faà di Bruno decomposition of the ``n``-th composition derivative.

    Returns a list of ``(k, exps, coeff)`` triples such that

    .. math::

        (\\sigma \\circ u)^{(n)}
            = \\sum (\\text{coeff})\\;\\sigma^{(k)}(u^{(0)})\\;
              \\prod_{i=1}^{n} \\big(u^{(i)}\\big)^{e_i},

    where ``exps = (e_1, ..., e_n)``. For ``n = 0`` the list is empty (the
    zeroth "derivative" is the value ``sigma(u^{(0)})`` itself, handled by the
    caller).
    """
    return list(_faa_di_bruno_pairs(n))


__all__ = [
    "bell_complete",
    "bell_number",
    "bell_partial",
    "faa_di_bruno_terms",
]
