# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Delta integral: fundamental theorem on discrete scales, quadrature on R."""

from __future__ import annotations

import math

import pytest
from omnibias.timescale import (
    delta_derivative,
    delta_integral,
    finite,
    h_integers,
    quantum,
    reals,
)


def test_fundamental_theorem_hZ() -> None:
    H = h_integers(0.5)
    f = lambda x: x**3 - 2 * x  # noqa: E731
    fD = lambda x: delta_derivative(f, x, H)  # noqa: E731
    got = delta_integral(fD, 0.0, 3.0, H)
    assert got == pytest.approx(f(3.0) - f(0.0))


def test_fundamental_theorem_quantum() -> None:
    Q = quantum(2.0)
    f = lambda x: x**2  # noqa: E731
    fD = lambda x: delta_derivative(f, x, Q) if x != 0 else 0.0  # noqa: E731
    got = delta_integral(fD, 1.0, 8.0, Q)
    assert got == pytest.approx(f(8.0) - f(1.0))


def test_fundamental_theorem_finite() -> None:
    T = finite((0.0, 1.0, 2.5, 4.0, 7.0))
    f = lambda x: math.sin(x) + x  # noqa: E731
    fD = lambda x: delta_derivative(f, x, T)  # noqa: E731
    got = delta_integral(fD, 0.0, 7.0, T)
    assert got == pytest.approx(f(7.0) - f(0.0))


def test_delta_integral_is_graininess_weighted_sum() -> None:
    H = h_integers(1.0)
    # sum_{t=0}^{2} 1 * f(t), half-open [0, 3): f(0)+f(1)+f(2).
    f = lambda x: x + 1.0  # noqa: E731
    assert delta_integral(f, 0.0, 3.0, H) == pytest.approx(1.0 + 2.0 + 3.0)


def test_reals_quadrature() -> None:
    R = reals()
    got = delta_integral(math.sin, 0.0, math.pi, R)
    assert got == pytest.approx(2.0, rel=1e-6)  # int_0^pi sin = 2


def test_integral_errors_and_degenerate() -> None:
    H = h_integers(0.5)
    assert delta_integral(lambda x: x, 1.0, 1.0, H) == 0.0
    with pytest.raises(ValueError):
        delta_integral(lambda x: x, 2.0, 1.0, H)
