# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validated complex continuation of scalar D-finite equations."""

from __future__ import annotations

import cmath
import math
import random

import pytest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.dfinite_continuation import continue_dfinite
from omnibias.core.verified.interval import Interval


def test_complex_interval_division_refuses_origin_box() -> None:
    denominator = ComplexInterval.from_parts(Interval(-1.0, 1.0), Interval(-1.0, 1.0))
    with pytest.raises(ZeroDivisionError, match="denominator box excluding zero"):
        _ = ComplexInterval.one() / denominator


def test_complex_interval_division_accepts_box_separated_by_imaginary_part() -> None:
    denominator = ComplexInterval.from_parts(Interval(-1.0, 1.0), Interval(2.0, 3.0))
    quotient = ComplexInterval.one() / denominator
    for point in (-1.0 + 2.0j, 3.0j, 1.0 + 2.0j):
        assert quotient.contains(1.0 / point)


def test_continue_exponential_to_real_point() -> None:
    result = continue_dfinite(
        ((-1,), (1,)),
        0.0,
        (1.0,),
        0.1,
        n_steps=16,
        order=14,
    )

    assert result.value.contains(math.exp(0.1))


def test_continue_sine_to_complex_point() -> None:
    target = 0.2 + 0.1j
    result = continue_dfinite(
        ((1,), (), (1,)),
        0.0,
        (0.0, 1.0),
        target,
        n_steps=32,
        order=16,
    )

    assert result.value.contains(cmath.sin(target))
    assert result.jet[1].contains(cmath.cos(target))


def test_continue_exponential_over_complex_box_contains_corners() -> None:
    target = ComplexInterval.from_parts(Interval(0.09, 0.11), Interval(-0.01, 0.01))
    result = continue_dfinite(
        ((-1,), (1,)),
        0.0,
        (1.0,),
        target,
        n_steps=32,
        order=16,
    )

    for x in (target.re.lo, target.re.hi):
        for y in (target.im.lo, target.im.hi):
            assert result.value.contains(cmath.exp(complex(x, y)))
    generator = random.Random(20260914)
    for _ in range(16):
        point = complex(
            generator.uniform(target.re.lo, target.re.hi),
            generator.uniform(target.im.lo, target.im.hi),
        )
        assert result.value.contains(cmath.exp(point))


def test_continuation_refuses_leading_coefficient_zero() -> None:
    # h f' - f = 0, continued from h=1 across a box containing h=0.
    with pytest.raises(ZeroDivisionError, match="leading coefficient"):
        continue_dfinite(
            ((-1,), (0, 1)),
            1.0,
            (1.0,),
            Interval(-0.1, 0.1),
            n_steps=8,
        )
