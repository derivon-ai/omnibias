# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global bounds, all-minimizer coverage, and finite-budget failure modes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import cast

import pytest
from omnibias.core.verified.global_min import (
    Box,
    BoxMinimum,
    GradientEnclosure,
    ValueEnclosure,
    minimize_box,
)
from omnibias.core.verified.interval import Interval


def _covered(result: BoxMinimum, point: tuple[float, ...]) -> bool:
    return any(all(axis.contains(x) for axis, x in zip(box, point, strict=True))
               for box in result.candidates)


def _constant(_: Box) -> Interval:
    return Interval.point(0.0)


def _loose_constant(box: Box) -> Interval:
    return Interval.point(0.0) if all(x.lo == x.hi for x in box) else Interval(-1.0, 0.0)


def test_interior_quadratic_encloses_exact_minimum_and_witness() -> None:
    result = minimize_box(lambda box: (box[0] - 0.3) ** 2,
                          (Interval(-1.0, 2.0),), value_tolerance=1e-8)
    exact_witness = (Fraction(result.point[0]) - Fraction(0.3)) ** 2
    assert result.minimum.contains(0.0)
    assert result.point_value.lo <= exact_witness <= result.point_value.hi
    assert exact_witness <= Fraction(result.gap_upper)
    assert result.status == "value_tolerance"
    assert result.gap_upper <= 1e-8
    assert _covered(result, (0.3,))
    assert result.gradient_evaluations == 0
    assert result.evaluated_boxes <= 4095
    assert result.value_evaluations <= 2 * result.evaluated_boxes


def test_strict_signs_reduce_to_correct_corner_before_sampling() -> None:
    domain = (Interval(-2.0, 1.0), Interval(-1.0, 3.0))
    value_boxes: list[Box] = []
    gradient_boxes: list[Box] = []

    def value(box: Box) -> Interval:
        value_boxes.append(box)
        return box[0] - 2 * box[1]

    def gradient(box: Box) -> tuple[Interval, ...]:
        gradient_boxes.append(box)
        return Interval.point(1.0), Interval.point(-2.0)

    result = minimize_box(value, domain, gradient=gradient, value_tolerance=1e-12)
    corner = (Interval.point(-2.0), Interval.point(3.0))
    assert result.point == (-2.0, 3.0)
    assert result.candidates == (corner,)
    assert result.minimum.contains(-8.0)
    assert gradient_boxes == [domain]
    assert value_boxes == [corner]
    assert (result.evaluated_boxes, result.value_evaluations, result.gradient_evaluations) == (1, 1, 1)


@pytest.mark.parametrize("derivative", [Interval(0.0, 1.0), Interval(-1.0, 0.0), Interval.point(0.0)])
def test_nonstrict_derivative_sign_preserves_flat_minimizers(derivative: Interval) -> None:
    domain = (Interval(-1.0, 1.0),)
    result = minimize_box(_constant, domain, gradient=lambda _: (derivative,), value_tolerance=0.0)
    assert result.candidates == (domain,)
    assert result.minimum == Interval.point(0.0)
    assert result.status == "value_tolerance"


def test_centered_form_tightens_loose_constant_bound() -> None:
    result = minimize_box(_loose_constant, (Interval(-2.0, 2.0),),
                          gradient=lambda _: (Interval.point(0.0),), value_tolerance=0.0)
    assert result.minimum == Interval.point(0.0)
    assert result.evaluated_boxes == 1
    assert result.candidates == ((Interval(-2.0, 2.0),),)


def test_equal_disconnected_minima_are_not_pruned() -> None:
    def two_wells(box: Box) -> Interval:
        x = box[0]
        if x.lo == x.hi and abs(x.lo) == 1.0:
            return Interval.point(0.0)
        raw = (x ** 2 - 1.0) ** 2
        return Interval(max(0.0, raw.lo), raw.hi)

    result = minimize_box(two_wells, (Interval(-2.0, 2.0),), value_tolerance=0.0)
    assert result.status == "value_tolerance"
    assert result.minimum == Interval.point(0.0)
    assert result.evaluated_boxes == 3
    assert _covered(result, (-1.0,))
    assert _covered(result, (1.0,))
    assert len(result.candidates) == 2


def test_coupled_quadratic_cover_contains_known_minimizer() -> None:
    def quadratic(box: Box) -> Interval:
        x, y = box
        return (x + y - 0.25) ** 2 + (x - y - 0.75) ** 2

    def gradient(box: Box) -> tuple[Interval, ...]:
        x, y = box
        return 4 * x - 2, 4 * y + 1

    result = minimize_box(quadratic, (Interval(-1.0, 1.0),) * 2,
                          gradient=gradient, value_tolerance=1e-7)
    assert result.status == "value_tolerance"
    assert result.minimum.contains(0.0)
    assert _covered(result, (0.5, -0.25))
    assert result.gradient_evaluations == result.evaluated_boxes


@pytest.mark.parametrize("dimension", [1, 3, 257])
def test_flat_objective_preserves_entire_box_in_any_dimension(dimension: int) -> None:
    domain = (Interval(-1.0, 2.0),) * dimension
    result = minimize_box(_constant, domain, value_tolerance=0.0)
    assert result.domain == domain
    assert result.candidates == (domain,)
    assert result.status == "value_tolerance"
    assert result.evaluated_boxes == 1
    assert result.gap_upper == 0.0
    with pytest.raises(FrozenInstanceError):
        result.status = "box_budget"  # type: ignore[misc]


@pytest.mark.parametrize(("budget", "expected"), [(1, 1), (2, 1), (3, 3), (4, 3), (5, 5)])
def test_split_budget_is_atomic_and_counts_every_evaluated_box(budget: int, expected: int) -> None:
    result = minimize_box(_loose_constant, (Interval(-1.0, 1.0),),
                          value_tolerance=0.0, max_boxes=budget)
    assert result.status == "box_budget"
    assert result.evaluated_boxes == expected
    assert result.value_evaluations == 2 * expected
    assert result.minimum == Interval(-1.0, 0.0)
    for x in (-1.0, -0.7, 0.0, 0.7, 1.0):
        assert _covered(result, (x,))


def test_boundary_minimum_survives_without_gradient() -> None:
    result = minimize_box(lambda box: box[0], (Interval(0.0, 1.0),),
                          value_tolerance=1e-5)
    assert result.status == "value_tolerance"
    assert result.minimum.contains(0.0)
    assert _covered(result, (0.0,))
    assert result.evaluated_boxes > len(result.candidates)


def test_outward_gap_cannot_round_down_to_requested_tolerance() -> None:
    point_value = 2.0 ** -53

    def value(box: Box) -> Interval:
        return Interval.point(point_value) if box[0].lo == box[0].hi else Interval(-1.0, point_value)

    result = minimize_box(value, (Interval(-1.0, 1.0),), value_tolerance=1.0, max_boxes=1)
    assert result.status == "box_budget"
    assert result.gap_upper > 1.0
    assert Fraction(result.gap_upper) >= Fraction(1) + Fraction(point_value)


def test_incumbent_uses_point_upper_bound() -> None:
    result = minimize_box(lambda _: Interval(-100.0, 3.0), (Interval(-1.0, 1.0),), max_boxes=1)
    assert result.minimum.hi == 3.0
    assert result.point_value == Interval(-100.0, 3.0)
    assert result.gap_upper >= 103.0


def test_adjacent_endpoint_resolution_preserves_bounds() -> None:
    lower = 1.0
    upper = math.nextafter(lower, math.inf)
    domain = (Interval(lower, upper),)
    result = minimize_box(_loose_constant, domain, value_tolerance=0.0)
    assert result.status == "resolution_limit"
    assert result.evaluated_boxes == 1
    assert result.minimum == Interval(-1.0, 0.0)
    assert result.candidates == (domain,)
    assert domain[0].contains(result.point[0])


def test_terminal_lower_bounds_remain_in_global_bound_at_budget_exit() -> None:
    tiny = math.nextafter(0.0, 1.0)

    def value(box: Box) -> Interval:
        x = box[0]
        if x.lo == x.hi:
            return Interval.point(0.0)
        return Interval(-0.5 if x.lo >= 2 * tiny else -1.0, 0.0)

    result = minimize_box(value, (Interval(0.0, 5 * tiny),), max_boxes=5, value_tolerance=0.0)
    assert result.status == "box_budget"
    assert result.minimum.lo == -1.0
    assert result.evaluated_boxes == 5
    assert len(result.candidates) == 3
    for i in range(6):
        assert _covered(result, (i * tiny,))


def test_fixed_coordinate_does_not_prevent_other_axis_split() -> None:
    domain = (Interval.point(3.0), Interval(0.0, 1.0))
    result = minimize_box(_loose_constant, domain, max_boxes=3, value_tolerance=0.0)
    assert result.evaluated_boxes == 3
    assert len(result.candidates) == 2
    assert all(box[0] == domain[0] for box in result.candidates)
    assert _covered(result, (3.0, 0.0))
    assert _covered(result, (3.0, 1.0))


@pytest.mark.parametrize("axis", [Interval(1e308, 1.7e308), Interval(-1.7e308, -1e308),
                                  Interval(-1.7e308, 1.7e308)])
def test_extreme_finite_endpoints_have_feasible_midpoints_and_splits(axis: Interval) -> None:
    result = minimize_box(_loose_constant, (axis,), max_boxes=3, value_tolerance=0.0)
    assert result.evaluated_boxes == 3
    assert math.isfinite(result.point[0])
    assert axis.contains(result.point[0])
    assert len(result.candidates) == 2
    assert result.candidates[0][0].hi == result.candidates[1][0].lo
    assert _covered(result, (axis.lo,)) and _covered(result, (axis.hi,))


@pytest.mark.parametrize("derivative", [Interval(-math.inf, math.inf),
                                        Interval(-1e308, 1e308), Interval(-1.0, 1.0)])
def test_unsafe_centered_form_falls_back_to_natural_bound(derivative: Interval) -> None:
    result = minimize_box(_loose_constant, (Interval(-1.7e308, 1.7e308),),
                          gradient=lambda _: (derivative,), max_boxes=1)
    assert result.minimum == Interval(-1.0, 0.0)
    assert result.status == "box_budget"


def test_infinite_strict_gradient_can_still_reduce_to_face() -> None:
    result = minimize_box(lambda box: box[0], (Interval(-1.0, 1.0),),
                          gradient=lambda _: (Interval(1.0, math.inf),), value_tolerance=0.0)
    assert result.point == (-1.0,)
    assert result.minimum == Interval.point(-1.0)
    assert result.status == "value_tolerance"


@pytest.mark.parametrize("bound", [Interval(-math.inf, math.inf), Interval(-math.inf, 0.0),
                                  Interval(0.0, math.inf)])
def test_unbounded_value_enclosure_remains_inconclusive(bound: Interval) -> None:
    result = minimize_box(lambda _: bound, (Interval(-1.0, 1.0),), max_boxes=1)
    assert result.minimum == bound
    assert result.gap_upper == math.inf
    assert result.status == "box_budget"


def test_point_domain_with_uncertain_value_reports_resolution_limit() -> None:
    result = minimize_box(lambda _: Interval(-1.0, 1.0), (Interval.point(3.0),), max_boxes=1)
    assert result.status == "resolution_limit"
    assert result.value_evaluations == 1
    assert result.candidates == ((Interval.point(3.0),),)


def test_parent_lower_bound_is_inherited_when_children_are_looser() -> None:
    def value(box: Box) -> Interval:
        x = box[0]
        if x.lo == x.hi:
            return Interval.point(0.0)
        return Interval(-1.0 if x.lo == -1.0 and x.hi == 1.0 else -100.0, 0.0)

    result = minimize_box(value, (Interval(-1.0, 1.0),), max_boxes=3, value_tolerance=0.0)
    assert result.minimum.lo == -1.0


def test_inconsistent_box_and_point_values_are_refused() -> None:
    def value(box: Box) -> Interval:
        return Interval.point(1.0) if box[0].lo == box[0].hi else Interval.point(0.0)

    with pytest.raises(ValueError, match="contradict"):
        minimize_box(value, (Interval(-1.0, 1.0),))


def test_inconsistent_inherited_lower_bound_is_refused() -> None:
    calls = 0

    def value(_: Box) -> Interval:
        nonlocal calls
        calls += 1
        return Interval(0.0, 1.0) if calls <= 2 else Interval.point(-1.0)

    with pytest.raises(ValueError, match="inherited"):
        minimize_box(value, (Interval(-1.0, 1.0),), max_boxes=3, value_tolerance=0.0)


@pytest.mark.parametrize("exception", [ValueError("callback failure"), OverflowError("callback failure")])
@pytest.mark.parametrize("location", ["value", "gradient"])
def test_callback_exceptions_are_not_masked(exception: Exception, location: str) -> None:
    def fail(_: Box) -> Interval:
        raise exception

    with pytest.raises(type(exception), match="callback failure"):
        minimize_box(fail if location == "value" else _constant, (Interval(-1.0, 1.0),),
                     gradient=cast(GradientEnclosure, fail) if location == "gradient" else None)


@pytest.mark.parametrize("budget", [0, -1, True, False, 1.5, "3", None])
def test_invalid_budget_is_refused(budget: object) -> None:
    with pytest.raises(ValueError, match="max_boxes"):
        minimize_box(_constant, (Interval(-1.0, 1.0),), max_boxes=cast(int, budget))


@pytest.mark.parametrize("tolerance", [-1.0, math.inf, -math.inf, math.nan, True, None, "bad"])
def test_invalid_tolerance_is_refused(tolerance: object) -> None:
    with pytest.raises(ValueError, match="value_tolerance"):
        minimize_box(_constant, (Interval(-1.0, 1.0),), value_tolerance=cast(float, tolerance))


@pytest.mark.parametrize("domain", [(), (1.0,), (Interval(-math.inf, 1.0),),
                                  (Interval(0.0, math.inf),)])
def test_invalid_domain_is_refused(domain: object) -> None:
    with pytest.raises(ValueError, match="domain"):
        minimize_box(_constant, cast(Sequence[Interval], domain))


@pytest.mark.parametrize("endpoint", [2**53 + 1, Fraction(1, 10), True])
@pytest.mark.parametrize("location", ["domain", "value", "gradient"])
def test_nonbinary64_endpoints_are_refused_without_changing_problem(endpoint: object, location: str) -> None:
    malformed = Interval(cast(float, endpoint), cast(float, endpoint))
    domain = (malformed,) if location == "domain" else (Interval(-1.0, 1.0),)
    value = (lambda _: malformed) if location == "value" else _constant
    gradient = (lambda _: (malformed,)) if location == "gradient" else None
    with pytest.raises(ValueError, match="binary64"):
        minimize_box(value, domain, gradient=gradient)


def test_exact_integer_endpoints_are_accepted_and_normalized() -> None:
    result = minimize_box(lambda _: Interval(0, 0), (Interval(1, 2),), value_tolerance=0.0)
    assert result.domain == (Interval(1.0, 2.0),)
    assert type(result.domain[0].lo) is float
    assert type(result.point_value.lo) is float


@pytest.mark.parametrize("bound", [Interval(math.inf, math.inf), Interval(-math.inf, -math.inf)])
@pytest.mark.parametrize("location", ["value", "gradient"])
def test_enclosure_without_any_finite_real_value_is_refused(bound: Interval, location: str) -> None:
    with pytest.raises(ValueError, match="finite real"):
        minimize_box((lambda _: bound) if location == "value" else _constant,
                     (Interval(-1.0, 1.0),),
                     gradient=(lambda _: (bound,)) if location == "gradient" else None)


@pytest.mark.parametrize("raw", [(), (Interval.point(0.0),) * 2, 1.0, None])
def test_malformed_gradient_shape_is_refused(raw: object) -> None:
    with pytest.raises(ValueError, match="one Interval"):
        minimize_box(_constant, (Interval(-1.0, 1.0),),
                     gradient=lambda _: cast(Sequence[Interval], raw))


@pytest.mark.parametrize("location", ["value", "gradient"])
def test_noninterval_callback_output_is_refused(location: str) -> None:
    with pytest.raises(TypeError, match="Interval"):
        minimize_box((lambda _: cast(Interval, 0.0)) if location == "value" else _constant,
                     (Interval(-1.0, 1.0),),
                     gradient=(lambda _: (cast(Interval, 0.0),)) if location == "gradient" else None)


def test_noncallable_oracles_are_refused() -> None:
    with pytest.raises(TypeError, match="callable"):
        minimize_box(cast(ValueEnclosure, None), (Interval(-1.0, 1.0),))
    with pytest.raises(TypeError, match="callable"):
        minimize_box(_constant, (Interval(-1.0, 1.0),), gradient=cast(GradientEnclosure, 1))
