# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified series summation -- the verified ``Sigma`` operator.

omnibias's closed-form derivative tower, the Wilson-plaquette character /
cluster expansion, and Neumann series are all *series*; turning a truncated
float sum into a theorem-grade enclosure needs a rigorous bound on the omitted
tail.  This module supplies that bound for the geometric-majorant case -- exactly
the situation of a convergent expansion with a proven consecutive-ratio bound
``|a_{n+1} / a_n| <= q < 1`` valid past the truncation index.

Given such a rigorous ``q`` (an :class:`Interval` with ``q.hi < 1``) the omitted
tail of a series truncated after ``a_N`` obeys

.. math::

    \Bigl| \sum_{k \ge 1} a_{N+k} \Bigr|
        \le |a_N| \sum_{k \ge 1} q^{k}
        = \frac{|a_N|\, q}{1 - q},

an outward-rounded :class:`Interval`.  The full sum is the interval sum of the
retained terms plus that symmetric tail enclosure.  Nothing here assumes a sign
pattern: the bound is on the *absolute* tail, so it is valid for sign-indefinite
series too.  :func:`geometric_series_closed_form` gives the analytic limit
``a / (1 - r)`` of a pure geometric series, the ground truth a certified partial
sum must enclose.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from omnibias.core.verified.interval import Interval, IntervalLike, sum_intervals


def _check_ratio(q: Interval) -> None:
    """Validate a geometric ratio *magnitude* bound ``0 <= q < 1``."""
    if q.lo < 0.0:
        raise ValueError(f"geometric ratio bound q must be >= 0, got q.lo={q.lo!r}")
    if q.hi >= 1.0:
        raise ValueError(
            f"geometric ratio bound q must be < 1 to converge, got q.hi={q.hi!r}"
        )


def geometric_tail_enclosure(last_term: IntervalLike, ratio: IntervalLike) -> Interval:
    r"""Symmetric enclosure of ``sum_{k>=1} a_{N+k}`` from the last retained term.

    ``last_term`` encloses the final *retained* term ``a_N``; ``ratio`` is a
    rigorous magnitude bound ``q`` (``0 <= q < 1``) on every subsequent
    consecutive ratio ``|a_{n+1} / a_n|``.  Returns ``[-B, B]`` with
    ``B = (|a_N| * q / (1 - q)).hi`` (outward rounded) -- a guaranteed enclosure
    of the omitted tail, regardless of the sign pattern of the terms.
    """
    term = Interval.from_value(last_term)
    q = Interval.from_value(ratio)
    _check_ratio(q)
    one_minus = Interval.point(1.0) - q
    bound = term.abs() * q * one_minus.reciprocal()
    b = bound.hi
    return Interval(-b, b)


def certified_geometric_series_sum(
    terms: Sequence[IntervalLike], ratio: IntervalLike
) -> Interval:
    r"""Enclosure of ``sum_{n>=0} a_n`` from retained terms ``a_0..a_N`` and a tail.

    ``terms`` are enclosures of the retained terms ``a_0, ..., a_N``; ``ratio`` is
    a rigorous magnitude bound ``q`` (``0 <= q < 1``) on every *omitted*
    consecutive ratio.  The result is
    ``sum(terms) + geometric_tail_enclosure(terms[-1], q)``.

    The caller is responsible for ``q`` being an a-priori rigorous bound (e.g. the
    convergence ratio of a character expansion); this routine does not infer it
    from the retained terms.
    """
    ivs = [Interval.from_value(t) for t in terms]
    if not ivs:
        raise ValueError("certified_geometric_series_sum needs >= 1 retained term")
    partial = sum_intervals(ivs)
    tail = geometric_tail_enclosure(ivs[-1], ratio)
    return partial + tail


def certified_ratio_series_sum(
    term: Callable[[int], IntervalLike], ratio: IntervalLike, *, num_terms: int
) -> Interval:
    r"""Enclosure of ``sum_{n>=0} a_n`` for terms ``a_n = term(n)`` under a ratio bound.

    A thin driver over :func:`certified_geometric_series_sum`: it evaluates the
    ``num_terms`` retained terms ``a_0, ..., a_{num_terms-1}`` from the ``term``
    callable and applies the geometric tail majorant, where ``ratio`` is a rigorous
    magnitude bound ``q`` (``0 <= q < 1``) on every *omitted* consecutive ratio
    ``|a_{n+1} / a_n|`` past the last retained term. Shared by the basic
    hypergeometric (``omnibias-qcalculus``) and polylogarithm
    (``core.verified.polylog``) enclosures, whose ``a_n`` are ``mpmath``-free
    products of exact rationals and certified transcendental base points.
    """
    if num_terms < 1:
        raise ValueError(f"num_terms must be >= 1, got {num_terms}")
    terms = [Interval.from_value(term(k)) for k in range(num_terms)]
    return certified_geometric_series_sum(terms, ratio)


def geometric_series_closed_form(first: IntervalLike, ratio: IntervalLike) -> Interval:
    r"""Exact enclosure of ``sum_{n>=0} first * ratio^n = first / (1 - ratio)``.

    Requires ``|ratio| < 1`` (checked as ``-1 < ratio.lo`` and ``ratio.hi < 1``).
    Unlike :func:`certified_geometric_series_sum` this is the analytic limit of a
    *pure* geometric series; it is the ground truth a truncated certified sum must
    contain.
    """
    a = Interval.from_value(first)
    r = Interval.from_value(ratio)
    if r.hi >= 1.0 or r.lo <= -1.0:
        raise ValueError(f"geometric closed form requires |ratio| < 1, got {r!r}")
    return a * (Interval.point(1.0) - r).reciprocal()


__all__ = [
    "certified_geometric_series_sum",
    "certified_ratio_series_sum",
    "geometric_series_closed_form",
    "geometric_tail_enclosure",
]
