# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent analytic minima for verified deep-network optimization."""

from __future__ import annotations

import pytest
from omnibias.core.verified.global_min import BoxMinimum, minimize_box
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_mv import mlp_jet_mv
from omnibias.core.verified.mlp_objective import IntervalMLP
from omnibias.core.verified.transcend import certificate_mode

mp = pytest.importorskip("mpmath")


def _contains_minimizer(result: BoxMinimum, point: tuple[object, ...]) -> bool:
    return any(
        all(mp.mpf(axis.lo) <= coordinate <= mp.mpf(axis.hi)
            for axis, coordinate in zip(box, point, strict=True))
        for box in result.candidates
    )


def test_monotone_user_chain_has_endpoint_global_minimum() -> None:
    objective = IntervalMLP.from_layers([
        ([[1.0]], [0.0], "sigmoid"),
        ([[1.0]], [0.0], "tanh"),
        ([[1.0]], [0.0], "sin"),
    ])
    result = minimize_box(objective.value, (Interval(-4.0, 4.0),),
                          gradient=objective.gradient, value_tolerance=1e-12)
    with mp.workdps(80):
        exact = mp.sin(mp.tanh(1 / (1 + mp.exp(4))))
        assert mp.mpf(result.minimum.lo) <= exact <= mp.mpf(result.minimum.hi)
    assert result.status == "value_tolerance"
    assert result.point == (-4.0,)
    assert result.candidates == ((Interval.point(-4.0),),)
    assert result.evaluated_boxes == 1
    assert result.gap_upper <= 1e-12


def test_nonconvex_chain_keeps_analytic_interior_minimizer() -> None:
    objective = IntervalMLP.from_layers([
        ([[1.0]], [0.0], "tanh"),
        ([[3.0]], [0.0], "sin"),
    ])
    result = minimize_box(objective.value, (Interval(-2.0, 2.0),),
                          gradient=objective.gradient, value_tolerance=1e-7,
                          max_boxes=4095)
    with mp.workdps(80):
        minimizer = -mp.atanh(mp.pi / 6)
        assert _contains_minimizer(result, (minimizer,))
        witness = mp.sin(3 * mp.tanh(mp.mpf(result.point[0])))
        assert mp.mpf(result.point_value.lo) <= witness <= mp.mpf(result.point_value.hi)
        assert witness + 1 <= mp.mpf(result.gap_upper)
    assert result.minimum.lo <= -1.0 <= result.minimum.hi
    assert result.status == "value_tolerance"
    assert result.gap_upper <= 1e-7
    print({"nonconvex_1d_bounds": (result.minimum.lo, result.minimum.hi),
           "evaluated_boxes": result.evaluated_boxes})


def test_coupled_two_dimensional_network_global_minimum() -> None:
    # f(x,y) = sin(3 tanh(x+y)) + c sin(2 tanh(x-y)).
    # Each sine is >= -1, and both attain -1 at an interior point of this box.
    coefficient = 0.4
    objective = IntervalMLP.from_layers([
        ([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0], "tanh"),
        ([[3.0, 0.0], [0.0, 2.0]], [0.0, 0.0], "sin"),
        ([[1.0, coefficient]], [0.0], None),
    ])
    result = minimize_box(objective.value, (Interval(-1.0, 1.0),) * 2,
                          gradient=objective.gradient, value_tolerance=1e-5,
                          max_boxes=8191)
    with mp.workdps(80):
        a, b = -mp.atanh(mp.pi / 6), -mp.atanh(mp.pi / 4)
        minimizer = ((a + b) / 2, (a - b) / 2)
        exact = -1 - mp.mpf(coefficient)
        assert _contains_minimizer(result, minimizer)
        assert mp.mpf(result.minimum.lo) <= exact <= mp.mpf(result.minimum.hi)
        x, y = map(mp.mpf, result.point)
        witness = mp.sin(3 * mp.tanh(x + y)) + mp.mpf(coefficient) * mp.sin(2 * mp.tanh(x - y))
        assert mp.mpf(result.point_value.lo) <= witness <= mp.mpf(result.point_value.hi)
        assert witness - exact <= mp.mpf(result.gap_upper)
    assert result.status == "value_tolerance"
    assert result.gap_upper <= 1e-5
    print({"nonconvex_2d_bounds": (result.minimum.lo, result.minimum.hi),
           "evaluated_boxes": result.evaluated_boxes})


def test_budget_exit_still_contains_nonconvex_global_minimum() -> None:
    objective = IntervalMLP.from_layers([
        ([[1.0]], [0.0], "tanh"),
        ([[3.0]], [0.0], "sin"),
    ])
    result = minimize_box(objective.value, (Interval(-2.0, 2.0),),
                          gradient=objective.gradient, value_tolerance=1e-12,
                          max_boxes=2)
    with mp.workdps(80):
        assert _contains_minimizer(result, (-mp.atanh(mp.pi / 6),))
    assert result.minimum.lo <= -1.0 <= result.minimum.hi
    assert result.status == "box_budget"
    assert result.evaluated_boxes == 1
    assert result.gap_upper > 1e-12


def test_value_only_weight_training_retains_both_zero_loss_parameters() -> None:
    # A parameter-box forward pass is a separate oracle from the fixed MLP
    # input-gradient adapter. No input derivative is reused as a weight gradient.
    def loss(parameter_box: tuple[Interval, ...]) -> Interval:
        with certificate_mode():
            prediction = mlp_jet_mv(
                (Interval.point(1.0),),
                [([[parameter_box[0]]], [0.0], "sin")],
                0,
            )[0][0]
            return (prediction - 0.5)**2

    result = minimize_box(loss, (Interval(-3.0, 3.0),),
                          value_tolerance=1e-8, max_boxes=4095)
    with mp.workdps(80):
        for weight in (mp.pi / 6, 5 * mp.pi / 6):
            assert _contains_minimizer(result, (weight,))
        witness_loss = (mp.sin(mp.mpf(result.point[0])) - mp.mpf("0.5"))**2
        assert mp.mpf(result.point_value.lo) <= witness_loss <= mp.mpf(result.point_value.hi)
    assert result.minimum.lo <= 0.0 <= result.minimum.hi
    assert result.status == "value_tolerance"
    assert result.gradient_evaluations == 0
    assert result.gap_upper <= 1e-8
