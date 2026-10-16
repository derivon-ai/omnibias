# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Snapshot and independent high-precision enclosure checks for MLP objectives."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from math import inf, nan

import pytest
from omnibias.core.verified import mlp_objective as mlp_module
from omnibias.core.verified import transcend
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mlp_objective import IntervalMLP


@pytest.mark.parametrize("use_factory", [True, False])
def test_source_mutation_does_not_change_objective(use_factory: bool) -> None:
    weights = [[2.0, -3.0]]
    biases = [0.5]
    layers = [(weights, biases, None)]
    objective = IntervalMLP.from_layers(layers) if use_factory else IntervalMLP(layers)
    box = (Interval(-1, 2), Interval(1, 3))
    before = objective.value(box), objective.gradient(box)
    weights[0][0] = 100.0
    weights.append([7.0, 8.0])
    biases[0] = -50.0
    layers.clear()
    assert (objective.value(box), objective.gradient(box)) == before
    assert objective.input_dim == 2
    assert objective.layers == ((((2.0, -3.0),), (0.5,), None),)
    with pytest.raises(FrozenInstanceError):
        objective.layers = ()
    assert before[0].contains(-10.5) and before[0].contains(1.5)
    assert before[1][0].contains(2) and before[1][1].contains(-3)


def test_affine_only_works_without_transcendental_backend(monkeypatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    objective = IntervalMLP.from_layers([([[2]], [1], None), ([[3]], None, None)])
    box = (Interval(0, 1),)
    value, gradient = objective.value(box), objective.gradient(box)
    assert value.contains(3) and value.contains(9)
    assert gradient[0].contains(6)
    smooth = IntervalMLP.from_layers([([[1]], None, "sin")])
    with pytest.raises(RuntimeError, match="rigorous transcendental backend"):
        smooth.value(box)
    with pytest.raises(RuntimeError, match="rigorous transcendental backend"):
        smooth.gradient(box)


@pytest.mark.parametrize("previous_strict", [False, True])
@pytest.mark.parametrize("raise_during_evaluation", [False, True])
def test_strict_evaluation_preserves_fallback_provenance_and_outer_state(
    monkeypatch, previous_strict, raise_during_evaluation
) -> None:
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", previous_strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    objective = IntervalMLP.from_layers([([[1]], None, "sin")])
    if raise_during_evaluation:
        def fail(*args):
            assert transcend.strict_backend()
            raise RuntimeError("test interval evaluation failure")
        monkeypatch.setattr(mlp_module, "mlp_jet_mv", fail)
    for method in (objective.value, objective.gradient):
        if raise_during_evaluation:
            with pytest.raises(RuntimeError, match="test interval evaluation failure"):
                method((Interval(0.1, 0.2),))
        else:
            method((Interval(0.1, 0.2),))
        assert transcend.libm_fallback_used()
        assert transcend.strict_backend() is previous_strict


@pytest.mark.parametrize("coefficient", [True, 1j, Fraction(1, 2), Interval.point(1.0), "1"])
def test_nonfloat_coefficients_are_rejected(coefficient) -> None:
    with pytest.raises(TypeError, match="Python int or float"):
        IntervalMLP.from_layers([([[coefficient]], None, None)])
    with pytest.raises(TypeError, match="Python int or float"):
        IntervalMLP.from_layers([([[1]], [coefficient], None)])


@pytest.mark.parametrize("coefficient", [nan, inf, -inf, 10**400, 2**53 + 1])
def test_nonfinite_and_inexact_integer_coefficients_are_rejected(coefficient) -> None:
    with pytest.raises(ValueError, match="binary64"):
        IntervalMLP.from_layers([([[coefficient]], None, None)])


@pytest.mark.parametrize("layers,match", [
    ([], "at least one"),
    ([([], None, None)], "positive"),
    ([([[]], None, None)], "positive"),
    ([([[1], [1, 2]], None, None)], "rectangular"),
    ([([[1]], [0, 1], None)], "bias width"),
    ([([[1], [2]], None, None)], "scalar output"),
    ([([[1, 2]], None, None), ([[1, 2]], None, None)], "preceding"),
    ([([[1]], None, "relu")], "activation"),
    ([([[1]], None, "softplus")], "activation"),
    ([([[1]], None, [])], "activation"),
    ([([[1]], None)], "each MLP layer"),
])
def test_invalid_layer_shapes_and_activations(layers, match) -> None:
    with pytest.raises(ValueError, match=match):
        IntervalMLP.from_layers(layers)


def test_invalid_and_degenerate_boxes() -> None:
    objective = IntervalMLP.from_layers([([[2]], [1], None)])
    for method in (objective.value, objective.gradient):
        with pytest.raises(ValueError, match="dimension"):
            method(())
        with pytest.raises(ValueError, match="finite endpoints"):
            method((Interval(-inf, 1),))
        with pytest.raises(TypeError, match="Interval objects"):
            method(((0, 1),))
    assert objective.value((Interval.point(0.5),)).contains(2)
    assert objective.gradient((Interval.point(0.5),))[0].contains(2)


@pytest.mark.parametrize("endpoint", [Fraction(1, 10), True, 2**53 + 1])
def test_box_endpoints_cannot_be_silently_rounded(endpoint) -> None:
    objective = IntervalMLP.from_layers([([[1]], None, None)])
    for method in (objective.value, objective.gradient):
        with pytest.raises((TypeError, ValueError), match="endpoints"):
            method((Interval(endpoint, endpoint),))


def test_exact_integer_box_endpoints_match_binary64_boxes() -> None:
    objective = IntervalMLP.from_layers([([[1]], None, None)])
    for method in (objective.value, objective.gradient):
        assert method((Interval(2**53, 2**53 + 2),)) == method((Interval(float(2**53), float(2**53 + 2)),))


def test_monotone_user_chain_high_precision_enclosures() -> None:
    mp = pytest.importorskip("mpmath")
    objective = IntervalMLP.from_layers([
        ([[1]], [0], "sigmoid"), ([[1]], [0], "tanh"), ([[1]], [0], "sin")
    ])
    box = (Interval(-4, 4),)
    value, gradient = objective.value(box), objective.gradient(box)[0]
    assert gradient.lo > 0  # sufficient to select the left boundary minimum
    with mp.workdps(80):
        for i in range(81):
            x = mp.mpf(-4) + mp.mpf(i) / 10
            sigmoid = 1 / (1 + mp.exp(-x))
            tanh = mp.tanh(sigmoid)
            expected = mp.sin(tanh)
            derivative = mp.cos(tanh) * (1 - tanh**2) * sigmoid * (1 - sigmoid)
            assert mp.mpf(value.lo) <= expected <= mp.mpf(value.hi)
            assert mp.mpf(gradient.lo) <= derivative <= mp.mpf(gradient.hi)


def test_nonconvex_profile_enclosure_tightens_near_minimum() -> None:
    mp = pytest.importorskip("mpmath")
    objective = IntervalMLP.from_layers([([[1]], None, "tanh"), ([[3]], None, "sin")])
    with mp.workdps(80):
        center = float(-mp.atanh(mp.pi / 6))
        wide = objective.value((Interval(center - 0.1, center + 0.1),))
        narrow_box = (Interval(center - 0.001, center + 0.001),)
        narrow = objective.value(narrow_box)
        gradient = objective.gradient(narrow_box)[0]
        assert narrow.contains(-1) and wide.contains(-1)
        assert gradient.contains_zero()
        assert narrow.width < wide.width / 1000
        for i in range(21):
            x = mp.mpf(center) - mp.mpf(0.001) + mp.mpf(0.002) * i / 20
            if not mp.mpf(narrow_box[0].lo) <= x <= mp.mpf(narrow_box[0].hi):
                continue
            t = mp.tanh(x)
            f, derivative = mp.sin(3 * t), 3 * mp.cos(3 * t) * (1 - t**2)
            assert mp.mpf(narrow.lo) <= f <= mp.mpf(narrow.hi)
            assert mp.mpf(gradient.lo) <= derivative <= mp.mpf(gradient.hi)


def test_two_dimensional_mixture_values_and_gradient() -> None:
    mp = pytest.importorskip("mpmath")
    objective = IntervalMLP.from_layers([
        ([[1, 1], [1, -1]], [0, 0], "tanh"),
        ([[3, 0], [0, 2]], None, "sin"),
        ([[1, 0.4]], None, None),
    ])
    box = (Interval(-1, 1), Interval(-1, 1))
    value, gradient = objective.value(box), objective.gradient(box)
    with mp.workdps(80):
        for i in range(11):
            for j in range(11):
                x, y = mp.mpf(-1) + mp.mpf(i) / 5, mp.mpf(-1) + mp.mpf(j) / 5
                a, b = mp.tanh(x + y), mp.tanh(x - y)
                first = 3 * mp.cos(3 * a) * (1 - a**2)
                second = 2 * mp.mpf(0.4) * mp.cos(2 * b) * (1 - b**2)
                expected = (mp.sin(3 * a) + mp.mpf(0.4) * mp.sin(2 * b), first + second, first - second)
                for real, interval in zip(expected, (value, *gradient), strict=True):
                    assert mp.mpf(interval.lo) <= real <= mp.mpf(interval.hi)


def test_cosine_and_affine_bias_readout() -> None:
    mp = pytest.importorskip("mpmath")
    objective = IntervalMLP.from_layers([([[2]], [0.5], "cos"), ([[-3]], [2], None)])
    point = 0.2
    value, gradient = objective.value((Interval.point(point),)), objective.gradient((Interval.point(point),))
    with mp.workdps(80):
        argument = 2 * mp.mpf(point) + mp.mpf(0.5)
        expected, derivative = 2 - 3 * mp.cos(argument), 6 * mp.sin(argument)
        assert mp.mpf(value.lo) <= expected <= mp.mpf(value.hi)
        assert mp.mpf(gradient[0].lo) <= derivative <= mp.mpf(gradient[0].hi)


@pytest.mark.parametrize("first,second,point", [
    (1e308, 1.0, 1e308),
    (1e308, 1e-308, 2.0),
    (1e-308, 1e308, 1e-308),
    (5e-324, 1e308, 5e-324),
    (1e308, 1e308, 0.0),
])
def test_extreme_affine_values_enclose_exact_rationals(first, second, point) -> None:
    objective = IntervalMLP.from_layers([([[first]], None, None), ([[second]], None, None)])
    box = (Interval.point(point),)
    value, gradient = objective.value(box), objective.gradient(box)[0]
    derivative = Fraction.from_float(first) * Fraction.from_float(second)
    expected = derivative * Fraction.from_float(point)
    # Direct rational comparisons also handle infinite interval endpoints.
    assert value.lo <= expected <= value.hi
    assert gradient.lo <= derivative <= gradient.hi


def test_overflow_cancellation_remains_an_enclosure() -> None:
    objective = IntervalMLP.from_layers([
        ([[1e308], [-1e308]], None, None), ([[1, 1]], None, None)
    ])
    box = (Interval.point(2.0),)
    assert objective.value(box).contains(0)
    assert objective.gradient(box)[0].contains(0)


def test_underflowing_nonlinear_values_do_not_claim_exact_zero() -> None:
    mp = pytest.importorskip("mpmath")
    objective = IntervalMLP.from_layers([([[1e-308]], None, "sin")])
    box = (Interval.point(1e-308),)
    value, gradient = objective.value(box), objective.gradient(box)[0]
    with mp.workdps(80):
        argument = mp.mpf(1e-308)**2
        expected = mp.sin(argument)
        derivative = mp.mpf(1e-308) * mp.cos(argument)
        assert expected > 0 and value.hi > 0
        assert mp.mpf(value.lo) <= expected <= mp.mpf(value.hi)
        assert mp.mpf(gradient.lo) <= derivative <= mp.mpf(gradient.hi)


def test_overflowing_sine_argument_uses_safe_global_range() -> None:
    pytest.importorskip("mpmath")
    objective = IntervalMLP.from_layers([([[1e308]], None, "sin")])
    box = (Interval.point(1e308),)
    value, gradient = objective.value(box), objective.gradient(box)[0]
    assert value.lo <= -1 and value.hi >= 1
    assert gradient.lo <= -1e308 and gradient.hi >= 1e308
