# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Containment tests for validated complexified ODE flow."""

from __future__ import annotations

import cmath
import random

import pytest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.complex_ode import (
    ComplexTaylorSeries,
    integrate_complex_ivp,
)
from omnibias.core.verified.interval import Interval


def _exponential(
    series: list[ComplexTaylorSeries],
) -> list[ComplexTaylorSeries]:
    return [series[0]]


def _riccati(
    series: list[ComplexTaylorSeries],
) -> list[ComplexTaylorSeries]:
    return [series[0] * series[0]]


def _harmonic(
    series: list[ComplexTaylorSeries],
) -> list[ComplexTaylorSeries]:
    return [series[1], -series[0]]


def _complex_rectangle_samples(
    box: ComplexInterval,
    *,
    seed: int,
    random_count: int = 64,
) -> list[complex]:
    real = (box.re.lo, box.re.mid, box.re.hi)
    imag = (box.im.lo, box.im.mid, box.im.hi)
    points = [complex(x, y) for x in real for y in imag]
    rng = random.Random(seed)
    points.extend(
        complex(
            rng.uniform(box.re.lo, box.re.hi),
            rng.uniform(box.im.lo, box.im.hi),
        )
        for _ in range(random_count)
    )
    return points


def test_complex_exponential_encloses_dense_and_random_initial_values() -> None:
    initial = ComplexInterval(Interval(0.99, 1.01), Interval(-0.01, 0.01))
    (result,) = integrate_complex_ivp(
        _exponential,
        [initial],
        0.0,
        0.5,
        order=14,
        n_steps=8,
    )
    multiplier = cmath.exp(0.5)
    for value in _complex_rectangle_samples(initial, seed=17001):
        assert result.contains(value * multiplier)


def test_complex_riccati_encloses_dense_and_random_initial_values() -> None:
    initial = ComplexInterval(Interval(0.45, 0.55), Interval(-0.05, 0.05))
    final_time = 0.25
    (result,) = integrate_complex_ivp(
        _riccati,
        [initial],
        0.0,
        final_time,
        order=16,
        n_steps=12,
    )
    for value in _complex_rectangle_samples(initial, seed=17002):
        truth = value / (1.0 - final_time * value)
        assert result.contains(truth)


def test_complex_harmonic_flow_contains_exact_solution() -> None:
    first, second = integrate_complex_ivp(
        _harmonic,
        [1 + 0.1j, -0.2 + 0.3j],
        0.0,
        0.75,
        order=14,
        n_steps=8,
    )
    c = cmath.cos(0.75)
    s = cmath.sin(0.75)
    assert first.contains((1 + 0.1j) * c + (-0.2 + 0.3j) * s)
    assert second.contains(-(1 + 0.1j) * s + (-0.2 + 0.3j) * c)


def test_complex_taylor_series_algebra() -> None:
    first = ComplexTaylorSeries(
        [
            ComplexInterval.point(1 + 1j),
            ComplexInterval.point(2 - 1j),
            ComplexInterval.zero(),
        ]
    )
    second = ComplexTaylorSeries(
        [
            ComplexInterval.point(1 - 1j),
            ComplexInterval.one(),
            ComplexInterval.zero(),
        ]
    )
    product = first * second
    assert product.coeffs[0].contains((1 + 1j) * (1 - 1j))
    assert product.coeffs[1].contains((1 + 1j) + (2 - 1j) * (1 - 1j))
    assert (2 - first).coeffs[0].contains(1 - 1j)


def test_complex_integrator_refuses_invalid_arguments_and_dimensions() -> None:
    with pytest.raises(ValueError, match="t1 >= t0"):
        integrate_complex_ivp(_exponential, [1], 1.0, 0.0)
    with pytest.raises(ValueError, match="positive integer"):
        integrate_complex_ivp(_exponential, [1], 0.0, 1.0, n_steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        integrate_complex_ivp(_exponential, [1], 0.0, 1.0, order=0)

    def wrong_dimension(
        series: list[ComplexTaylorSeries],
    ) -> list[ComplexTaylorSeries]:
        return [series[0], series[0]]

    with pytest.raises(ValueError, match="wrong dimension"):
        integrate_complex_ivp(wrong_dimension, [1], 0.0, 0.1)
