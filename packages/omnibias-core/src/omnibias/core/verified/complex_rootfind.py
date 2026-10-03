# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Parametric complex interval Newton on rectangular enclosures.

For a family ``g(z, p)`` whose parameter ``p`` ranges over a fixed box, the
interval-Newton operator

.. math::

   N(X) = m - g(m,P) / \partial_z g(X,P)

proves one root for every parameter and uniqueness in ``X`` when
``N(X)`` lies strictly inside ``X`` and the derivative enclosure excludes
zero.  Callers remain responsible for supplying sound complex enclosures of
the function and derivative on the declared parameter box.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval

ComplexFn = Callable[[ComplexInterval], ComplexInterval]
ComplexNewtonStatus = Literal[
    "unique_root",
    "derivative_contains_zero",
    "no_root",
    "max_iter",
]


@dataclass(frozen=True)
class ComplexNewtonResult:
    """Outcome of a rectangular parametric complex interval-Newton proof."""

    status: ComplexNewtonStatus
    enclosure: ComplexInterval
    iterations: int
    unique_for_every_parameter: bool


def _interior(inner: ComplexInterval, outer: ComplexInterval) -> bool:
    return bool(
        outer.re.lo < inner.re.lo
        and inner.re.hi < outer.re.hi
        and outer.im.lo < inner.im.lo
        and inner.im.hi < outer.im.hi
    )


def _intersection(
    left: ComplexInterval,
    right: ComplexInterval,
) -> ComplexInterval | None:
    re_lo = max(left.re.lo, right.re.lo)
    re_hi = min(left.re.hi, right.re.hi)
    im_lo = max(left.im.lo, right.im.lo)
    im_hi = min(left.im.hi, right.im.hi)
    if re_lo > re_hi or im_lo > im_hi:
        return None
    return ComplexInterval(Interval(re_lo, re_hi), Interval(im_lo, im_hi))


def _width(value: ComplexInterval) -> float:
    return float(max(value.re.width, value.im.width))


def parametric_complex_interval_newton(
    function: ComplexFn,
    derivative: ComplexFn,
    domain: ComplexInterval,
    *,
    max_iter: int = 20,
    tol: float = 1e-13,
) -> ComplexNewtonResult:
    """Prove a unique complex root branch over an implicit parameter box.

    ``function`` and ``derivative`` may close over interval parameters.  A
    ``unique_root`` result then applies separately to every parameter in that
    box, not merely to one representative value.
    """
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not 0.0 < tol < float("inf"):
        raise ValueError("tol must be finite and positive")
    current = domain
    unique = False
    for iteration in range(1, max_iter + 1):
        midpoint = complex(current.re.mid, current.im.mid)
        midpoint_box = ComplexInterval.point(midpoint)
        derivative_box = derivative(current)
        if derivative_box.modulus().lo <= 0.0:
            return ComplexNewtonResult(
                "derivative_contains_zero",
                current,
                iteration,
                False,
            )
        try:
            newton = midpoint_box - function(midpoint_box) / derivative_box
        except ZeroDivisionError:
            return ComplexNewtonResult(
                "derivative_contains_zero",
                current,
                iteration,
                False,
            )
        if _interior(newton, current):
            unique = True
        narrowed = _intersection(current, newton)
        if narrowed is None:
            return ComplexNewtonResult("no_root", newton, iteration, False)
        current = narrowed
        if unique and _width(current) <= tol:
            return ComplexNewtonResult("unique_root", current, iteration, True)
    return ComplexNewtonResult(
        "unique_root" if unique else "max_iter",
        current,
        max_iter,
        unique,
    )


__all__ = [
    "ComplexFn",
    "ComplexNewtonResult",
    "ComplexNewtonStatus",
    "parametric_complex_interval_newton",
]
