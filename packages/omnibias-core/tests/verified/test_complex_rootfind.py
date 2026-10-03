# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parametric complex interval-Newton tests."""

from __future__ import annotations

import cmath
import random

import pytest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.complex_rootfind import (
    parametric_complex_interval_newton,
)
from omnibias.core.verified.interval import Interval


def _samples(box: ComplexInterval, *, seed: int) -> list[complex]:
    values = [
        complex(real, imag)
        for real in (box.re.lo, box.re.mid, box.re.hi)
        for imag in (box.im.lo, box.im.mid, box.im.hi)
    ]
    rng = random.Random(seed)
    values.extend(
        complex(
            rng.uniform(box.re.lo, box.re.hi),
            rng.uniform(box.im.lo, box.im.hi),
        )
        for _ in range(32)
    )
    return values


def test_parametric_linear_root_branch_is_certified() -> None:
    parameter = ComplexInterval(Interval(0.9, 1.1), Interval(-0.1, 0.1))
    domain = ComplexInterval(Interval(0.5, 1.5), Interval(-0.5, 0.5))
    result = parametric_complex_interval_newton(
        lambda value: value - parameter,
        lambda _value: ComplexInterval.one(),
        domain,
    )
    assert result.status == "unique_root"
    assert result.unique_for_every_parameter
    for truth in _samples(parameter, seed=18001):
        assert result.enclosure.contains(truth)


def test_parametric_square_root_branch_contains_dense_and_random_truth() -> None:
    parameter = ComplexInterval(Interval(3.99, 4.01), Interval(-0.01, 0.01))
    domain = ComplexInterval(Interval(1.9, 2.1), Interval(-0.1, 0.1))
    result = parametric_complex_interval_newton(
        lambda value: value * value - parameter,
        lambda value: 2 * value,
        domain,
    )
    assert result.status == "unique_root"
    assert result.unique_for_every_parameter
    for value in _samples(parameter, seed=18002):
        assert result.enclosure.contains(cmath.sqrt(value))
        assert not result.enclosure.contains(-cmath.sqrt(value))


def test_complex_newton_refuses_singular_derivative_and_invalid_arguments() -> None:
    domain = ComplexInterval(
        Interval(-0.1, 0.1),
        Interval(-0.1, 0.1),
    )
    result = parametric_complex_interval_newton(
        lambda value: value * value,
        lambda value: 2 * value,
        domain,
    )
    assert result.status == "derivative_contains_zero"
    assert not result.unique_for_every_parameter
    with pytest.raises(ValueError, match="positive integer"):
        parametric_complex_interval_newton(
            lambda value: value,
            lambda _value: ComplexInterval.one(),
            domain,
            max_iter=0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        parametric_complex_interval_newton(
            lambda value: value,
            lambda _value: ComplexInterval.one(),
            domain,
            tol=0.0,
        )
