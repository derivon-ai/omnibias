# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded global minimization with caller-supplied interval enclosures.

The objective must be continuous on the finite closed input box. The value
callback must enclose its value at EVERY point of each supplied box. If a
gradient callback is supplied, the objective must be continuously differentiable
there and each returned interval must enclose the corresponding partial.

Under these assumptions, every result encloses the global minimum and its
candidate boxes cover ALL global minimizers, including on the boundary. The
algorithm cannot establish the semantic soundness of arbitrary Python callbacks;
it does not imply a Lean proof, a particular transcendental backend, uniqueness,
or finite convergence to an arbitrary requested tolerance.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from operator import index as integer_index
from typing import Literal

from omnibias.core.verified.interval import Interval

Box = tuple[Interval, ...]
ValueEnclosure = Callable[[Box], Interval]
GradientEnclosure = Callable[[Box], Sequence[Interval]]
MinimumStatus = Literal["value_tolerance", "box_budget", "resolution_limit"]


def _gap(lower: float, upper: float) -> float:
    if lower == upper:
        return 0.0
    return math.nextafter(upper - lower, math.inf)


@dataclass(frozen=True)
class BoxMinimum:
    """Conditional global bounds and a cover of every possible minimizer.

    ``point`` is a feasible sampled point and ``point_value`` encloses its
    objective. ``candidates`` covers minimizers, not necessarily the whole
    domain after strict bound pruning or monotonic face reduction. Small value
    uncertainty does not imply small candidate boxes or a unique minimizer.
    """

    domain: Box
    minimum: Interval
    point: tuple[float, ...]
    point_value: Interval
    candidates: tuple[Box, ...]
    status: MinimumStatus
    evaluated_boxes: int
    value_evaluations: int
    gradient_evaluations: int

    @property
    def gap_upper(self) -> float:
        """Outward-rounded upper bound on the incumbent's suboptimality."""
        return _gap(self.minimum.lo, self.minimum.hi)


@dataclass(frozen=True)
class _Candidate:
    box: Box
    lower: float
    number: int


def _midpoint(interval: Interval) -> float:
    # Unlike (lo+hi)/2, this cannot overflow for finite endpoints of one sign.
    value = 0.5 * interval.lo + 0.5 * interval.hi
    return min(interval.hi, max(interval.lo, value))


def _split(box: Box) -> tuple[Box, Box] | None:
    axis: int | None = None
    cut = 0.0
    largest = -1.0
    for i, interval in enumerate(box):
        midpoint = _midpoint(interval)
        if not interval.lo < midpoint < interval.hi:
            midpoint = math.nextafter(interval.lo, interval.hi)
        if not interval.lo < midpoint < interval.hi:
            continue
        width = interval.hi - interval.lo  # Infinity is safe as a priority only.
        if width > largest:
            axis, cut, largest = i, midpoint, width
    if axis is None:
        return None
    interval = box[axis]
    left, right = list(box), list(box)
    left[axis] = Interval(interval.lo, cut)
    right[axis] = Interval(cut, interval.hi)
    return tuple(left), tuple(right)


def _enclosure(value: object, label: str) -> Interval:
    if not isinstance(value, Interval):
        raise TypeError(f"{label} callback must return Interval enclosures")
    value = _binary64_interval(value, label)
    # Infinite bounds can be sound but [inf,inf] and [-inf,-inf] cannot
    # contain a finite real objective or partial derivative.
    if not value.lo < math.inf or not value.hi > -math.inf:
        raise ValueError(f"{label} enclosure must contain a finite real value")
    return value


def _binary64_interval(interval: Interval, label: str) -> Interval:
    endpoints: list[float] = []
    for endpoint in (interval.lo, interval.hi):
        if isinstance(endpoint, bool) or not isinstance(endpoint, (int, float)):
            raise ValueError(f"{label} endpoints must be binary64 floats or exactly representable integers")
        try:
            converted = float(endpoint)
        except OverflowError as exc:
            raise ValueError(f"{label} endpoint is not exactly representable in binary64") from exc
        if converted != endpoint:
            raise ValueError(f"{label} endpoint is not exactly representable in binary64")
        endpoints.append(converted)
    return Interval(*endpoints)


def _centered(
    box: Box, point: tuple[float, ...], point_value: Interval,
    gradient: tuple[Interval, ...],
) -> Interval | None:
    """Optional mean-value tightening; arithmetic overflow only loses precision."""
    if not all(math.isfinite(v) for interval in (point_value, *gradient)
               for v in (interval.lo, interval.hi)):
        return None
    try:
        result = point_value
        for interval, center, derivative in zip(box, point, gradient, strict=True):
            if interval.lo == interval.hi or derivative.lo == derivative.hi == 0.0:
                continue
            offset = interval - Interval.point(center)
            if not math.isfinite(offset.lo) or not math.isfinite(offset.hi):
                return None
            result = result + derivative * offset
            if not math.isfinite(result.lo) or not math.isfinite(result.hi):
                return None
        return result
    except (OverflowError, ValueError):
        # This scope contains interval arithmetic only, never a callback.
        return None


def minimize_box(
    value: ValueEnclosure, domain: Sequence[Interval], *,
    gradient: GradientEnclosure | None = None, value_tolerance: float = 1e-6,
    max_boxes: int = 4095,
) -> BoxMinimum:
    """Enclose a global minimum over any finite-dimensional compact box.

    ``max_boxes`` counts all evaluated boxes, including the root and discarded
    boxes. A split is performed only when both children fit that budget. Each
    evaluated box uses at most two value calls and one gradient call. Strict
    derivative signs reduce coordinates to the minimizing face BEFORE value
    sampling. Finite gradient bounds can also tighten the natural enclosure by
    the centered mean-value form. Parent lower bounds are inherited.

    Only ``lower > incumbent_upper`` prunes a box, preserving tied minimizers.
    Budget and representable-resolution exits retain every surviving candidate
    and valid global bounds. ``value_tolerance`` certifies the value gap only.
    Unbounded callback intervals are allowed, but may prevent a finite gap.
    Callback exceptions propagate; visibly contradictory enclosures are refused.
    Interval endpoints must be binary64 floats or exactly representable integers;
    use ``Interval.from_rational`` to enclose other exact scalar values.
    """
    if not callable(value) or (gradient is not None and not callable(gradient)):
        raise TypeError("value and optional gradient must be callable")
    if isinstance(max_boxes, bool):
        raise ValueError("max_boxes must be a positive integer")
    try:
        budget = integer_index(max_boxes)
    except TypeError as exc:
        raise ValueError("max_boxes must be a positive integer") from exc
    if budget < 1:
        raise ValueError("max_boxes must be a positive integer")
    if isinstance(value_tolerance, bool):
        raise ValueError("value_tolerance must be finite and nonnegative")
    try:
        tolerance = float(value_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("value_tolerance must be finite and nonnegative") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("value_tolerance must be finite and nonnegative")
    initial = tuple(domain)
    if not initial or any(not isinstance(interval, Interval) for interval in initial):
        raise ValueError("domain must be a nonempty sequence of Interval objects")
    initial = tuple(_binary64_interval(interval, "domain") for interval in initial)
    if not all(math.isfinite(endpoint) for interval in initial for endpoint in (interval.lo, interval.hi)):
        raise ValueError("domain endpoints must be finite")
    evaluated_boxes = value_evaluations = gradient_evaluations = 0
    best_point: tuple[float, ...] | None = None
    best_value: Interval | None = None

    def evaluate(box: Box, inherited_lower: float = -math.inf) -> _Candidate:
        nonlocal evaluated_boxes, value_evaluations, gradient_evaluations, best_point, best_value
        evaluated_boxes += 1
        derivative: tuple[Interval, ...] | None = None
        if gradient is not None:
            gradient_evaluations += 1
            raw = gradient(box)
            if not isinstance(raw, Sequence) or len(raw) != len(box):
                raise ValueError("gradient callback must return one Interval per coordinate")
            derivative = tuple(_enclosure(entry, "gradient") for entry in raw)
            box = tuple(Interval.point(interval.lo) if bound.lo > 0.0 else
                        Interval.point(interval.hi) if bound.hi < 0.0 else interval
                        for interval, bound in zip(box, derivative, strict=True))
        value_evaluations += 1
        natural = _enclosure(value(box), "value")
        point = tuple(_midpoint(interval) for interval in box)
        point_box = tuple(Interval.point(coordinate) for coordinate in point)
        if point_box == box:
            point_value = natural
        else:
            value_evaluations += 1
            point_value = _enclosure(value(point_box), "value")
        if max(natural.lo, point_value.lo) > min(natural.hi, point_value.hi):
            raise ValueError("box and point value enclosures contradict each other")
        bound = natural
        if derivative is not None:
            centered = _centered(box, point, point_value, derivative)
            if centered is not None:
                try:
                    bound = bound.intersect(centered)
                except ValueError as exc:
                    raise ValueError("value and gradient enclosures contradict each other") from exc
        lower = max(inherited_lower, bound.lo)
        if lower > bound.hi or lower > point_value.hi:
            raise ValueError("value enclosures contradict an inherited or centered lower bound")
        if best_value is None or point_value.hi < best_value.hi:
            best_point, best_value = point, point_value
        return _Candidate(box, lower, evaluated_boxes)

    root = evaluate(initial)
    active: list[tuple[float, int, _Candidate]] = [(root.lower, root.number, root)]
    terminal: list[_Candidate] = []
    status: MinimumStatus
    while True:
        assert best_value is not None and best_point is not None
        upper = best_value.hi
        # Once the least lower bound exceeds U, every queued box can be pruned.
        if active and active[0][0] > upper:
            active.clear()
        terminal = [node for node in terminal if node.lower <= upper]
        lowers = ([active[0][0]] if active else []) + [node.lower for node in terminal]
        if not lowers:
            raise ValueError("oracle bounds exclude every candidate for the global minimum")
        lower = min(lowers)
        if _gap(lower, upper) <= tolerance:
            status = "value_tolerance"
            break
        if not active:
            status = "resolution_limit"
            break
        candidate = active[0][2]
        children = _split(candidate.box)
        if children is None:
            heapq.heappop(active)
            terminal.append(candidate)
            continue
        if evaluated_boxes + 2 > budget:
            status = "box_budget"
            break
        # Keep the parent alive until BOTH child enclosures have been obtained.
        left = evaluate(children[0], candidate.lower)
        right = evaluate(children[1], candidate.lower)
        heapq.heappop(active)
        assert best_value is not None
        for node in (left, right):
            if node.lower <= best_value.hi:
                heapq.heappush(active, (node.lower, node.number, node))

    survivors = [item[2] for item in active if item[0] <= best_value.hi] + terminal
    survivors.sort(key=lambda node: (node.lower, node.number))
    return BoxMinimum(initial, Interval(lower, best_value.hi), best_point, best_value,
                      tuple(node.box for node in survivors), status, evaluated_boxes,
                      value_evaluations, gradient_evaluations)


__all__ = ["Box", "BoxMinimum", "GradientEnclosure", "MinimumStatus", "ValueEnclosure", "minimize_box"]
