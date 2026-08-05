# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Branch-and-bound over the input box: refine a sound enclosure to any tolerance.

A single Taylor-model pass is sound but can be loose when the box is large or an
activation is very nonlinear / unstable across it.  Splitting the input box along
its widest axis and re-enclosing each sub-box shrinks the over-approximation
(the dependency/wrapping error scales with the box radius), and the **hull** of
the sub-box enclosures is still a rigorous enclosure of the whole.  This module
drives that refinement for a scalar read-out ``c . net(x) + d`` -- the quantity
behind robustness margins, Lipschitz sweeps and reachable-set faces.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike, hull
from omnibias.verify._core.network import Network
from omnibias.verify._core.propagate import interval_propagate
from omnibias.verify._core.taylor import linear_image, taylor_propagate

Box = tuple[Interval, ...]


def _readout(
    net: Network, box: Box, weights: Sequence[float], bias: float, order: int
) -> Interval:
    """Sound enclosure of ``weights . net(box) + bias``, intersected with IBP."""
    models = taylor_propagate(net, box, order=order)
    tm = linear_image(weights, models).bound() + Interval.point(bias)
    ibp_out = interval_propagate(net, box).output
    ibp = Interval.point(bias)
    for w, oi in zip(weights, ibp_out, strict=True):
        ibp = ibp + Interval.point(w) * oi
    return Interval(max(tm.lo, ibp.lo), min(tm.hi, ibp.hi))


def _widest_axis(box: Box) -> int:
    return max(range(len(box)), key=lambda i: box[i].width)


def _split(box: Box, axis: int) -> tuple[Box, Box]:
    mid = box[axis].mid
    left = tuple(Interval(iv.lo, mid) if i == axis else iv for i, iv in enumerate(box))
    right = tuple(Interval(mid, iv.hi) if i == axis else iv for i, iv in enumerate(box))
    return left, right


@dataclass(frozen=True)
class RangeResult:
    """A rigorous enclosure of the scalar read-out and the work spent to get it."""

    enclosure: Interval
    boxes_explored: int
    refined: bool


def scalar_readout_range(
    net: Network,
    input_box: Sequence[IntervalLike],
    weights: Sequence[float],
    *,
    bias: float = 0.0,
    order: int = 2,
    max_boxes: int = 256,
    tol: float = 1e-6,
) -> RangeResult:
    r"""Rigorous enclosure of ``weights . net(x) + bias`` over the input box.

    Branch-and-bound on both faces: the priority queues expand whichever sub-box
    currently owns the global minimum (resp. maximum) candidate, so effort is
    spent only where it tightens the answer.  The returned :attr:`enclosure`
    contains the read-out for *every* ``x`` in the box, regardless of the budget.

    ``max_boxes`` and ``tol`` are **search** budgets: spending less of either
    yields a wider enclosure, never an invalid one.  In particular ``tol`` is not
    a slack on the reported bound -- it only decides when refining has stopped
    paying off.
    """
    box0: Box = tuple(Interval.from_value(v) for v in input_box)
    full = _readout(net, box0, weights, bias, order)

    lower = _bound_face(net, box0, weights, bias, order, max_boxes, tol, maximize=False)
    upper = _bound_face(net, box0, weights, bias, order, max_boxes, tol, maximize=True)
    enclosure = Interval(lower[0], upper[0])
    boxes = 1 + lower[1] + upper[1]
    # Never report looser than the single-pass enclosure (BaB only tightens).
    enclosure = Interval(max(enclosure.lo, full.lo), min(enclosure.hi, full.hi))
    return RangeResult(enclosure=enclosure, boxes_explored=boxes, refined=boxes > 1)


def _bound_face(
    net: Network,
    box0: Box,
    weights: Sequence[float],
    bias: float,
    order: int,
    max_boxes: int,
    tol: float,
    *,
    maximize: bool,
) -> tuple[float, int]:
    """Return ``(certified_extreme, boxes_explored)`` for min (or max) of the read-out.

    Sign trick: maximisation runs the minimisation loop on the negated read-out.

    The boxes on the heap always **partition** ``box0`` (a split replaces a box by
    two children that cover it), so the smallest lower bound on the heap is a sound
    lower bound on the read-out over the whole input box. That frontier minimum is
    the only thing this function certifies.

    Sampling the box centre gives an *incumbent* -- a feasible value, hence an
    **upper** bound on the minimum. It steers the search and decides when to stop;
    it must never be mixed into the returned bound, and ``tol`` must never shrink
    it. Both are soundness traps: an incumbent is on the wrong side of the true
    minimum, so ``min(frontier, incumbent)`` can certify a value the read-out never
    attains, and a ``tol``-sized concession is a ``tol``-sized false claim.
    ``tol`` and ``max_boxes`` buy tightness only; the result is sound at any budget.
    """
    sign = -1.0 if maximize else 1.0
    w = [sign * x for x in weights]
    b = sign * bias

    def readout(box: Box) -> Interval:
        return _readout(net, box, w, b, order)

    counter = itertools.count()
    # heap entries: (enclosure.lo, tie, box) -- pop the box that owns the global min.
    root = readout(box0)
    heap: list[tuple[float, int, Box]] = [(root.lo, next(counter), box0)]
    best_feasible = _sample_center(net, box0, w, b, order)
    explored = 1
    while heap and explored < max_boxes:
        entry = heapq.heappop(heap)
        cand_lo, _, box = entry
        if cand_lo >= best_feasible - tol:
            # Refining further cannot improve on the incumbent by more than `tol`,
            # so stop -- but put the box back first. It still owns the frontier
            # minimum, and dropping it would leave `box0` uncovered and the
            # remaining minimum too high (i.e. not a lower bound at all).
            heapq.heappush(heap, entry)
            break
        left, right = _split(box, _widest_axis(box))
        for child in (left, right):
            enc = readout(child)
            explored += 1
            best_feasible = min(best_feasible, _sample_center(net, child, w, b, order))
            heapq.heappush(heap, (enc.lo, next(counter), child))
    # The frontier minimum, and nothing else. `root.lo` is the (unreachable) fallback
    # for an empty heap and is itself a sound bound over the whole box.
    certified_min = heap[0][0] if heap else root.lo
    # Translate back: min of (sign * f) is sign * (extreme of f).
    return sign * certified_min, explored


def _sample_center(
    net: Network, box: Box, weights: Sequence[float], bias: float, order: int
) -> float:
    """A feasible (hence bounding) read-out value: evaluate at the box centre."""
    center = tuple(Interval.point(iv.mid) for iv in box)
    models = taylor_propagate(net, center, order=order)
    return (linear_image(weights, models).bound() + Interval.point(bias)).hi


def output_range(
    net: Network,
    input_box: Sequence[IntervalLike],
    out_index: int,
    *,
    order: int = 2,
    max_boxes: int = 256,
    tol: float = 1e-6,
) -> RangeResult:
    """Branch-and-bound enclosure of a single output coordinate over the input box."""
    models = taylor_propagate(net, input_box, order=order)
    n_out = len(models)
    if not 0 <= out_index < n_out:
        raise ValueError(f"out_index must be in 0..{n_out - 1}")
    weights = [1.0 if i == out_index else 0.0 for i in range(n_out)]
    return scalar_readout_range(
        net, input_box, weights, order=order, max_boxes=max_boxes, tol=tol
    )


def hull_of_boxes(enclosures: Sequence[Interval]) -> Interval:
    """Rigorous hull of several enclosures (helper for manual subdivisions)."""
    return hull(list(enclosures))


__all__ = [
    "RangeResult",
    "hull_of_boxes",
    "output_range",
    "scalar_readout_range",
]
