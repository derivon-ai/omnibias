# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The shared containment assertion for every rigorous-enclosure test.

An enclosure is a **logical claim**: every attainable value lies inside
``[lo, hi]``. It is not an estimate of the range, so the two directions of error
are not comparable. A too-wide interval is merely loose and still true. A
too-narrow one is false, and the size of the miss is irrelevant -- ``1e-16``
outside the interval falsifies the claim exactly as thoroughly as ``1e3``
outside it.

That asymmetry is why these helpers take **no tolerance**. The repo's enclosure
tests used to assert ``lo - 1e-9 <= y <= hi + 1e-9``, which quietly converts the
claim into "approximately encloses" and cannot distinguish "sound and tight"
from "unsound by less than a nanounit". Two genuinely unsound certificate paths
(the branch-and-bound frontier and the submodular marginal bound) passed a fully
green suite for exactly that reason.

Where a *computed reference* carries its own floating-point error, widen the
reference into an interval and use :func:`assert_encloses_interval`; do not add
slack to the enclosure under test. The distinction matters: the first says "the
truth is somewhere in here and the enclosure covers all of it", the second says
"the enclosure is wrong but not by much".
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Protocol


class _Bounded(Protocol):
    lo: float
    hi: float


def bounds(enclosure: Any) -> tuple[float, float]:
    """``(lo, hi)`` from an ``Interval``-like object or a plain 2-tuple."""
    if hasattr(enclosure, "lo") and hasattr(enclosure, "hi"):
        return float(enclosure.lo), float(enclosure.hi)
    lo, hi = enclosure
    return float(lo), float(hi)


def assert_encloses(enclosure: Any, value: float, *, what: str = "value") -> None:
    """Assert ``value`` lies in ``enclosure``. Exact -- no tolerance, by design.

    ``what`` labels the quantity in the failure message (e.g. ``"f(0.5)"``).
    """
    lo, hi = bounds(enclosure)
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise AssertionError(f"enclosure [{lo!r}, {hi!r}] is not finite")
    if lo > hi:
        raise AssertionError(f"malformed enclosure [{lo!r}, {hi!r}]: lo > hi")
    if value < lo:
        raise AssertionError(
            f"UNSOUND: {what} = {value!r} is below the enclosure [{lo!r}, {hi!r}] "
            f"by {lo - value!r}. An enclosure must contain every attainable value; "
            f"being close is not the same as being correct."
        )
    if value > hi:
        raise AssertionError(
            f"UNSOUND: {what} = {value!r} is above the enclosure [{lo!r}, {hi!r}] "
            f"by {value - hi!r}. An enclosure must contain every attainable value; "
            f"being close is not the same as being correct."
        )


def assert_encloses_all(enclosure: Any, values: Iterable[float], *, what: str = "value") -> None:
    """Apply :func:`assert_encloses` to every sampled value."""
    for index, value in enumerate(values):
        assert_encloses(enclosure, float(value), what=f"{what}[{index}]")


def assert_encloses_interval(outer: Any, inner: Any, *, what: str = "reference") -> None:
    """Assert ``outer`` contains all of ``inner``.

    Use when the reference is itself computed and carries rounding error: widen the
    *reference* into an interval rather than adding slack to the enclosure under test.
    """
    lo, hi = bounds(outer)
    ref_lo, ref_hi = bounds(inner)
    if ref_lo < lo or ref_hi > hi:
        raise AssertionError(
            f"UNSOUND: {what} interval [{ref_lo!r}, {ref_hi!r}] is not contained in "
            f"[{lo!r}, {hi!r}] (undershoot {max(lo - ref_lo, 0.0)!r}, "
            f"overshoot {max(ref_hi - hi, 0.0)!r})"
        )


def assert_lower_bound(bound: float, true_value: float, *, what: str = "quantity") -> None:
    """Assert ``bound <= true_value``. Exact -- a lower bound that is too high is false."""
    if bound > true_value:
        raise AssertionError(
            f"UNSOUND: certified lower bound {bound!r} exceeds the true {what} "
            f"{true_value!r} by {bound - true_value!r}"
        )


def assert_upper_bound(bound: float, true_value: float, *, what: str = "quantity") -> None:
    """Assert ``bound >= true_value``. Exact -- an upper bound that is too low is false."""
    if bound < true_value:
        raise AssertionError(
            f"UNSOUND: certified upper bound {bound!r} is below the true {what} "
            f"{true_value!r} by {true_value - bound!r}"
        )


def assert_sandwich(
    lower: float, true_value: float, upper: float, *, what: str = "optimum"
) -> None:
    """Assert ``lower <= true_value <= upper`` against an oracle. Exact.

    This is the claim a gap certificate actually makes. Checking only
    ``lower <= upper`` is *internal consistency* and stays true when both sides
    sit on the same wrong side of the truth.
    """
    assert_lower_bound(lower, true_value, what=what)
    assert_upper_bound(upper, true_value, what=what)


__all__ = [
    "assert_encloses",
    "assert_encloses_all",
    "assert_encloses_interval",
    "assert_lower_bound",
    "assert_sandwich",
    "assert_upper_bound",
    "bounds",
]
