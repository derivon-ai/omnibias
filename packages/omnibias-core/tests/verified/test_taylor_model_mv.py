# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Containment tests for multivariate Taylor models over a box in R^n."""

from __future__ import annotations

import itertools

import pytest
from omnibias.core.multi_index import num_multi_indices
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.taylor_model_mv import TaylorModelMV


def _grid(center: tuple[float, ...], radius: tuple[float, ...], n: int = 5) -> list[tuple[float, ...]]:
    axes = [
        [c - r + 2 * r * i / (n - 1) for i in range(n)] if r > 0 else [c]
        for c, r in zip(center, radius, strict=True)
    ]
    return list(itertools.product(*axes))


def test_constant_model_is_tight() -> None:
    tm = TaylorModelMV.constant(3.5, center=(1.0, -2.0), radius=(2.0, 0.5), order=3)
    b = tm.bound()
    assert b.lo <= 3.5 <= b.hi
    assert b.width < 1e-12


def test_coordinate_bound_is_box_axis() -> None:
    tm = TaylorModelMV.coordinate(1, center=(0.5, 0.25), radius=(0.1, 0.25), order=4)
    b = tm.bound()
    assert b.lo <= 0.0  # 0.25 - 0.25
    assert b.hi >= 0.5  # 0.25 + 0.25


def test_coordinate_constructor_validation() -> None:
    with pytest.raises(ValueError):
        TaylorModelMV.coordinate(2, center=(0.0, 0.0), radius=(1.0, 1.0), order=3)
    with pytest.raises(ValueError):
        TaylorModelMV.coordinate(0, center=(0.0,), radius=(1.0,), order=0)


def test_coefficient_count_must_match() -> None:
    with pytest.raises(ValueError):
        TaylorModelMV((0.0, 0.0), (1.0, 1.0), 2, [Interval.point(0.0)], Interval.point(0.0))


def test_polynomial_product_encloses_pointwise_2d() -> None:
    center, radius = (0.3, -0.2), (0.4, 0.3)
    order = 6
    x = TaylorModelMV.coordinate(0, center, radius, order)
    y = TaylorModelMV.coordinate(1, center, radius, order)
    # f(x, y) = (x^2 + 1) * (y - 2) + x*y
    f = (x.pow_int(2) + 1.0) * (y - 2.0) + x * y
    b = f.bound()
    for xi, yi in _grid(center, radius):
        val = (xi * xi + 1.0) * (yi - 2.0) + xi * yi
        assert b.lo <= val <= b.hi


def test_eval_at_point_is_exact_for_polynomial() -> None:
    center, radius = (0.0, 0.0), (0.5, 0.5)
    order = 4
    x = TaylorModelMV.coordinate(0, center, radius, order)
    y = TaylorModelMV.coordinate(1, center, radius, order)
    f = x * x - y * y + 3.0  # degree 2, exactly representable -> ~zero remainder
    # Outward rounding inflates a [0,0] remainder by at most ~1 ulp of zero.
    assert f.remainder.width < 1e-300
    for xi, yi in _grid(center, radius):
        enc = f.eval((xi - center[0], yi - center[1]))
        true = xi * xi - yi * yi + 3.0
        assert enc.lo <= true <= enc.hi
        assert enc.width < 1e-9  # point evaluation of an exact polynomial is tight


def test_remainder_absorbs_high_degree_2d() -> None:
    center, radius = (0.0, 0.0), (0.5, 0.5)
    order = 2
    x = TaylorModelMV.coordinate(0, center, radius, order)
    y = TaylorModelMV.coordinate(1, center, radius, order)
    # (x*y)^2 has total degree 4 > order 2 -> must be pushed into the remainder.
    f = (x * y).pow_int(2)
    assert f.remainder.width > 0.0
    b = f.bound()
    for xi, yi in _grid(center, radius):
        assert b.lo <= (xi * yi) ** 2 <= b.hi


def test_product_remainder_is_rigorous_three_vars() -> None:
    center, radius = (0.1, -0.1, 0.2), (0.3, 0.2, 0.25)
    order = 3
    x = TaylorModelMV.coordinate(0, center, radius, order)
    y = TaylorModelMV.coordinate(1, center, radius, order)
    z = TaylorModelMV.coordinate(2, center, radius, order)
    f = (x + y + z).pow_int(3) - x * y * z
    b = f.bound()
    for xi, yi, zi in _grid(center, radius, n=4):
        val = (xi + yi + zi) ** 3 - xi * yi * zi
        assert b.lo <= val <= b.hi


def test_distributivity_addition_then_bound() -> None:
    center, radius = (0.2, 0.3), (0.4, 0.4)
    order = 4
    x = TaylorModelMV.coordinate(0, center, radius, order)
    y = TaylorModelMV.coordinate(1, center, radius, order)
    lhs = (x + y) * (x - y)
    rhs = x.pow_int(2) - y.pow_int(2)
    # Same function: enclosures must both contain every sample.
    for xi, yi in _grid(center, radius):
        val = xi * xi - yi * yi
        assert lhs.bound().lo <= val <= lhs.bound().hi
        assert rhs.eval((xi - center[0], yi - center[1])).lo <= val
        assert val <= rhs.eval((xi - center[0], yi - center[1])).hi


def test_scalar_and_reverse_ops() -> None:
    center, radius = (0.0, 0.0), (0.5, 0.5)
    order = 3
    x = TaylorModelMV.coordinate(0, center, radius, order)
    g = 2.0 * x + 1.0
    h = 1.0 - x  # exercises __rsub__
    for xi, _yi in _grid(center, radius):
        assert g.eval((xi, 0.0)).lo <= 2.0 * xi + 1.0 <= g.eval((xi, 0.0)).hi
        assert h.eval((xi, 0.0)).lo <= 1.0 - xi <= h.eval((xi, 0.0)).hi


def test_coefficient_layout_matches_multi_index() -> None:
    order = 5
    tm = TaylorModelMV.constant(0.0, (0.0, 0.0), (1.0, 1.0), order)
    assert len(tm.coeffs) == num_multi_indices(2, order)
