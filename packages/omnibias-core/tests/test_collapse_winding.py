# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Winding collapse: unique integer in Δarg/2π; 0 on the contour blocks."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from omnibias.core.collapse import (
    WINDING_SPEC,
    are_distinct,
    get_collapse,
    reset_collapse_registry,
    winding_collapse,
)
from omnibias.core.collapse.winding import (
    arg_iv,
    contains_origin,
    integers_in,
    winding_enclosure,
)
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_winding_is_registered_and_distinct() -> None:
    assert get_collapse("winding") == WINDING_SPEC
    for name in ("bias", "temperature", "enclosure", "verdict", "identity"):
        assert are_distinct(WINDING_SPEC, get_collapse(name)).distinct


def test_arg_of_one_and_i() -> None:
    one = arg_iv(ComplexInterval.one())
    assert one is not None
    assert one.contains(0.0)
    imag = arg_iv(ComplexInterval.imag_unit())
    assert imag is not None
    assert imag.contains(math.pi / 2)
    assert arg_iv(ComplexInterval.zero()) is None
    assert contains_origin(ComplexInterval.zero())


def test_integers_in_isolates_a_unique_count() -> None:
    assert integers_in(Interval(-0.1, 0.1)) == (0,)
    assert integers_in(Interval(0.9, 1.1)) == (1,)
    assert integers_in(Interval(0.9, 2.1)) == (1, 2)


def test_unit_circle_winding_of_z() -> None:
    verdict = winding_collapse((0, 1), expected=1)
    assert verdict.proved
    assert verdict.outcome.surviving == 1
    assert verdict.outcome.residual is not None
    assert verdict.outcome.residual.contains(1.0)
    assert verdict.outcome.honesty["winding_collapse"] is True
    assert verdict.outcome.honesty["founding_bias_collapse"] is False


def test_unit_circle_winding_of_z_squared() -> None:
    verdict = winding_collapse((0, 0, 1), expected=2)
    assert verdict.proved
    assert verdict.outcome.surviving == 2


def test_root_outside_the_circle_has_winding_zero() -> None:
    verdict = winding_collapse((-2, 1), expected=0)
    assert verdict.proved
    assert verdict.outcome.surviving == 0


def test_wrong_expected_winding_is_disproved() -> None:
    verdict = winding_collapse((0, 1), expected=0)
    assert verdict.disproved
    assert verdict.outcome.surviving == 1


def test_constant_has_winding_zero() -> None:
    assert winding_collapse((1,), expected=0).proved


def test_zero_polynomial_is_blocked() -> None:
    assert winding_collapse((0, 0), expected=0).blocked


def test_enclosure_contains_the_true_winding_and_avoids_zeros() -> None:
    enc = winding_enclosure((0, 1), 0j, 1.0, segments=64)
    assert enc is not None
    assert enc.contains(1.0)
    rng = random.Random(0)
    for _ in range(32):
        theta = rng.random() * 2.0 * math.pi
        value = complex(math.cos(theta), math.sin(theta))
        assert abs(value) > 0.0
    grid = [2.0 * math.pi * k / 20.0 for k in range(20)]
    for theta in grid:
        value = complex(math.cos(theta), math.sin(theta))
        assert abs(value) > 0.5


def test_rectangle_contour_encloses_winding_and_validates_shape() -> None:
    enc = winding_enclosure(
        (0, 1),
        0j,
        1.0,
        contour="rectangle",
        half_width=2.0,
        half_height=1.0,
        segments=64,
    )
    assert enc is not None
    assert enc.contains(1.0)
    verdict = winding_collapse(
        (0, 1),
        expected=1,
        contour="rectangle",
        half_width=2.0,
        half_height=1.0,
    )
    assert verdict.proved
    # A deterministic grid and random perimeter samples stay away from z=0.
    grid = [
        complex(-2.0 + 4.0 * t, -1.0) for t in np.linspace(0.0, 1.0, 17)
    ]
    grid += [
        complex(2.0, -1.0 + 2.0 * t) for t in np.linspace(0.0, 1.0, 17)
    ]
    grid += [
        complex(2.0 - 4.0 * t, 1.0) for t in np.linspace(0.0, 1.0, 17)
    ]
    grid += [
        complex(-2.0, 1.0 - 2.0 * t) for t in np.linspace(0.0, 1.0, 17)
    ]
    assert all(abs(value) >= 1.0 for value in grid)
    rng = random.Random(20260827)
    for _ in range(64):
        edge = rng.randrange(4)
        fraction = rng.random()
        point = (
            complex(-2.0 + 4.0 * fraction, -1.0)
            if edge == 0
            else complex(2.0, -1.0 + 2.0 * fraction)
            if edge == 1
            else complex(2.0 - 4.0 * fraction, 1.0)
            if edge == 2
            else complex(-2.0, 1.0 - 2.0 * fraction)
        )
        assert abs(point) >= 1.0
    with pytest.raises(ValueError, match="divisible by 4"):
        winding_enclosure((0, 1), 0j, 1.0, contour="rectangle", segments=10)
