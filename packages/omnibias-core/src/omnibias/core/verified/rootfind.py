# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified scalar root isolation -- the rigorous rate/eigenvalue selector.

A self-similar blow-up *rate* is an eigenvalue fixed by a scalar
*connection condition* ``g(c) = 0`` (residual minimisation alone leaves it
degenerate -- see the certified-evidence research log).  This module pins such a ``c``
*rigorously*:

* :func:`certified_sign_change` proves a root exists in ``[a, b]`` via the
  intermediate value theorem with certified endpoint signs.
* :func:`interval_newton` proves a root **exists and is unique** in an interval
  by the interval-Newton test ``N(X) \subseteq \mathrm{int}(X)`` and returns a
  tight enclosure of it.

Both consume callables returning :class:`Interval` enclosures of ``g`` (and, for
Newton, ``g'``), so they compose directly with the verified jet / sigma tower.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from omnibias.core.verified.interval import Interval

GFn = Callable[[Interval], Interval]


class NewtonResult(TypedDict):
    status: str
    enclosure: tuple[float, float]
    width: float
    iterations: int
    unique: bool


def certified_sign_change(g: GFn, a: float, b: float) -> bool:
    """True iff ``g`` provably changes sign across ``[a, b]`` (so a root exists)."""
    ga = g(Interval.point(a))
    gb = g(Interval.point(b))
    a_neg = ga.hi < 0.0
    a_pos = ga.lo > 0.0
    b_neg = gb.hi < 0.0
    b_pos = gb.lo > 0.0
    return (a_neg and b_pos) or (a_pos and b_neg)


def interval_newton(
    g: GFn,
    gprime: GFn,
    bracket: tuple[float, float],
    *,
    max_iter: int = 100,
    tol: float = 1e-14,
) -> NewtonResult:
    """Isolate a unique root of ``g`` in ``bracket`` via the interval-Newton test."""
    x = Interval(bracket[0], bracket[1])
    unique = False
    iterations = 0
    status = "max_iter"
    while iterations < max_iter:
        iterations += 1
        m = Interval.point(x.mid)
        gm = g(m)
        gp = gprime(x)
        if gp.contains_zero():
            # Derivative enclosure straddles zero: cannot apply the test here.
            status = "derivative_contains_zero" if not unique else "unique_root"
            break
        n_op = m - gm * gp.reciprocal()
        # Uniqueness test: N(X) strictly inside X.
        if x.lo < n_op.lo and n_op.hi < x.hi:
            unique = True
        try:
            x_new = n_op.intersect(x)
        except ValueError:
            status = "no_root"
            x = n_op
            break
        x = x_new
        if x.width <= tol:
            status = "unique_root" if unique else "converged_not_certified"
            break
    return {
        "status": status,
        "enclosure": (x.lo, x.hi),
        "width": x.width,
        "iterations": iterations,
        "unique": unique,
    }


def bisection_bracket(
    g: GFn, a: float, b: float, *, iters: int = 60
) -> tuple[float, float]:
    """Shrink a certified sign-change bracket by interval-safe bisection."""
    if not certified_sign_change(g, a, b):
        raise ValueError("no certified sign change on the initial bracket")
    ga = g(Interval.point(a))
    a_pos = ga.lo > 0.0
    lo, hi = a, b
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        gm = g(Interval.point(mid))
        if gm.contains_zero():
            break  # cannot certify the sign of g(mid); stop refining.
        mid_pos = gm.lo > 0.0
        if mid_pos == a_pos:
            lo = mid
        else:
            hi = mid
    return (lo, hi)


__all__ = [
    "GFn",
    "NewtonResult",
    "bisection_bracket",
    "certified_sign_change",
    "interval_newton",
]
