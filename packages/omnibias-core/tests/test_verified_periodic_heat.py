# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Continuum heat bounds, exact initial derivatives and refusal contracts."""

from __future__ import annotations

import importlib
import math
from fractions import Fraction
from typing import cast

import pytest
from omnibias.core.exp_polynomial import ExponentialBudget
from omnibias.core.periodic_heat import PeriodicHeatPlan
from omnibias.core.verified import transcend
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.periodic_heat import enclose_burgers_jet, enclose_heat_derivative


def _point(value: float = 0.0) -> tuple[Interval, ...]:
    return (Interval.point(value),)


def _cosine(*, degree: int = 16, budget: ExponentialBudget | None = None) -> PeriodicHeatPlan:
    return PeriodicHeatPlan.from_log_potential(1, {(1,): Fraction(1, 2)}, degree=degree, budget=budget)


def test_zero_potential_has_zero_remainder_and_constant_velocity_jet() -> None:
    pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_log_potential(2, {}, drift=(Fraction(1, 3), -2))
    point = (Interval(-2.0, 3.0), Interval(-0.5, 0.5))
    heat = enclose_heat_derivative(plan, point, Interval(0.0, 10.0))
    assert heat.value.contains(1.0) and heat.remainder_bound == 0.0
    assert enclose_heat_derivative(plan, point, Interval(0.0, 10.0), (2, 1)).value == Interval.point(0.0)
    jet = enclose_burgers_jet(plan, point, Interval(0.0, 10.0), order=4)
    assert jet.value[0].contains(Fraction(1, 3))
    assert jet.value[1].contains(-2.0)
    assert jet.heat_lower_bound == 1.0
    assert len(jet.multi_indices) == math.comb(6, 2)
    for alpha in jet.multi_indices[1:]:
        assert jet.derivative(alpha) == (Interval.point(0.0),) * 2


def test_low_degree_positivity_overrides_negative_finite_approximation() -> None:
    mp = pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_log_potential(1, {(1,): 2}, degree=1)
    heat = enclose_heat_derivative(plan, _point(3.0), Interval.point(0.0))
    assert heat.finite_part.hi < 0.0
    assert heat.value.lo > 0.0
    with mp.workdps(80):
        expected = mp.exp(2 * mp.cos(3))
        assert mp.mpf(heat.value.lo) <= expected <= mp.mpf(heat.value.hi)
        assert mp.mpf(heat.value.lo) <= mp.exp(-2)
    jet = enclose_burgers_jet(plan, _point(3.0), Interval.point(0.0))
    assert jet.heat_lower_bound > 0.0
    assert jet.value[0].contains(4.0 * math.sin(3.0))


def test_global_remainder_keeps_omitted_zero_mode_at_late_time() -> None:
    mp = pytest.importorskip("mpmath")
    plan = _cosine(degree=0)
    heat = enclose_heat_derivative(plan, _point(), Interval.point(100.0))
    assert heat.finite_part.contains(1.0)
    assert heat.remainder_bound > 0.0
    # exp(cos(x)/2) has conserved average I_0(1/2)>1. An error treated
    # incorrectly as a high-frequency tail would vanish as time grows.
    with mp.workdps(80):
        average = mp.besseli(0, mp.mpf("0.5"))
        assert average > mp.mpf(heat.finite_part.hi)
        assert mp.mpf(heat.value.lo) <= average <= mp.mpf(heat.value.hi)


def test_omitted_derivative_has_its_own_bound_at_initial_time() -> None:
    mp = pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_log_potential(1, {(7,): Fraction(1, 2)}, degree=0)
    with mp.workdps(80):
        x = mp.mpf("0.125")
        zeroth = enclose_heat_derivative(plan, _point(float(x)), Interval.point(0.0))
        fourth = enclose_heat_derivative(plan, _point(float(x)), Interval.point(0.0), (4,))
        expected = mp.diff(lambda z: mp.exp(mp.cos(7*z)/2), x, 4)
        assert fourth.finite_part == Interval.point(0.0)
        assert fourth.remainder_bound > zeroth.remainder_bound * 7**4
        assert mp.mpf(fourth.value.lo) <= expected <= mp.mpf(fourth.value.hi)


def test_mixed_modes_initial_raw_derivatives_and_getter_validation() -> None:
    mp = pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_velocity({1: Fraction(1, 4), 2: Fraction(-1, 5)},
                                         {3: Fraction(1, 8)}, mean=Fraction(2, 3), degree=22)
    x = 0.25
    jet = enclose_burgers_jet(plan, _point(x), Interval.point(0.0), order=5)
    with mp.workdps(80):
        for order in range(6):
            angle = order * mp.pi / 2
            expected = mp.cos(mp.mpf(x) + angle)/4 - 2**order * mp.cos(2*mp.mpf(x) + angle)/5
            expected += 3**order * mp.sin(3*mp.mpf(x) + angle)/8
            if order == 0:
                expected += mp.mpf(2)/3
            actual = jet.derivative((order,))[0]
            assert mp.mpf(actual.lo) <= expected <= mp.mpf(actual.hi)
    assert jet.value == jet.derivative((0,))
    with pytest.raises(ValueError, match="not stored"):
        jet.derivative((6,))
    with pytest.raises(ValueError, match="dimension"):
        jet.derivative((0, 0))
    with pytest.raises(TypeError, match="integer"):
        jet.derivative((True,))


def test_interval_space_time_enclosure_contains_independent_heat_values() -> None:
    mp = pytest.importorskip("mpmath")
    plan = _cosine()
    enclosure = enclose_heat_derivative(plan, (Interval(0.25, 0.5),), Interval(0.125, 0.25), (2,))
    with mp.workdps(80):
        for x in (mp.mpf("0.25"), mp.mpf("0.375"), mp.mpf("0.5")):
            for time in (mp.mpf("0.125"), mp.mpf("0.25")):
                # Independent Fourier-Bessel expansion of exp(cos(x)/2).
                expected = sum(-2*n*n*mp.besseli(n, mp.mpf("0.5"))*mp.exp(-n*n*time)*mp.cos(n*x)
                               for n in range(1, 60))
                assert mp.mpf(enclosure.value.lo) <= expected <= mp.mpf(enclosure.value.hi)


def test_pure_heat_ignores_drift_while_burgers_transports() -> None:
    pytest.importorskip("mpmath")
    stationary = _cosine()
    moving = PeriodicHeatPlan.from_log_potential(1, {(1,): Fraction(1, 2)}, drift=(2,))
    x, time = _point(0.5), Interval.point(0.25)
    assert enclose_heat_derivative(stationary, x, time) == enclose_heat_derivative(moving, x, time)
    jet = enclose_burgers_jet(moving, x, time)
    assert jet.value[0].contains(2.0)  # transported argument is exactly zero
    assert jet.value[0].hi - jet.value[0].lo < 1e-8


@pytest.mark.parametrize("endpoint", [True, Fraction(1, 10), 2**53 + 1, float("inf"), 10**400])
def test_invalid_interval_endpoint_is_not_silently_changed(endpoint: object) -> None:
    raw = cast(float, endpoint)
    # Interval itself accepts exact rational and large-int endpoints.
    if endpoint == 10**400:
        # Its own isnan conversion refuses this magnitude first.
        with pytest.raises(OverflowError):
            Interval(raw, raw)
        return
    interval = Interval(raw, raw)
    with pytest.raises((TypeError, ValueError), match="endpoints"):
        enclose_heat_derivative(_cosine(), (interval,), Interval.point(0.0))
    with pytest.raises((TypeError, ValueError), match="endpoints"):
        enclose_burgers_jet(_cosine(), _point(), interval)


@pytest.mark.parametrize("alpha", [(True,), (1.0,), (-1,), (0, 1)])
def test_bad_multi_index_is_rejected(alpha: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        enclose_heat_derivative(_cosine(), _point(), Interval.point(0.0), cast(tuple[int, ...], alpha))


@pytest.mark.parametrize("order", [True, 1.0, -1])
def test_bad_jet_order_is_rejected(order: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        enclose_burgers_jet(_cosine(), _point(), Interval.point(0.0), cast(int, order))


def test_shape_time_and_type_validation() -> None:
    plan = _cosine()
    with pytest.raises(ValueError, match="dimension"):
        enclose_heat_derivative(plan, (), Interval.point(0.0))
    with pytest.raises(ValueError, match="nonnegative time"):
        enclose_burgers_jet(plan, _point(), Interval(-0.25, 0.25))
    with pytest.raises(TypeError, match="Interval"):
        enclose_heat_derivative(plan, cast(tuple[Interval, ...], (0.0,)), Interval.point(0.0))
    with pytest.raises(TypeError, match="PeriodicHeatPlan"):
        enclose_heat_derivative(cast(PeriodicHeatPlan, object()), _point(), Interval.point(0.0))


def test_derivative_budget_includes_heat_order_above_velocity_order() -> None:
    plan = _cosine(budget=ExponentialBudget(max_derivative_order=2))
    with pytest.raises(ValueError, match="derivative order"):
        enclose_heat_derivative(plan, _point(), Interval.point(0.0), (3,))
    with pytest.raises(ValueError, match="derivative order"):
        enclose_burgers_jet(plan, _point(), Interval.point(0.0), order=2)


def test_combinatorial_jet_storage_is_refused_before_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("omnibias.core.verified.periodic_heat")
    plan = PeriodicHeatPlan.from_log_potential(5, {}, budget=ExponentialBudget(max_terms=10))

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("multi-index storage should be checked before materialization")

    monkeypatch.setattr(module, "multi_indices", unexpected)
    with pytest.raises(ValueError, match="coefficient count"):
        enclose_burgers_jet(plan, (Interval.point(0.0),) * 5, Interval.point(0.0), order=3)


def test_work_budget_is_shared_across_all_heat_derivatives() -> None:
    pytest.importorskip("mpmath")
    plan = _cosine(degree=1, budget=ExponentialBudget(max_work=700))
    for order in range(5):
        enclose_heat_derivative(plan, _point(), Interval.point(0.0), (order,))
    with pytest.raises(ValueError, match="work"):
        enclose_burgers_jet(plan, _point(), Interval.point(0.0), order=4)


def test_exact_derivative_multipliers_obey_bit_budget() -> None:
    pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_log_potential(1, {(7,): 1}, degree=1,
                                              budget=ExponentialBudget(max_coefficient_bits=8))
    with pytest.raises(ValueError, match="bit budget"):
        enclose_heat_derivative(plan, _point(), Interval.point(0.0), (4,))


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_strict_mode_and_sticky_history_survive_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, sticky: bool, strict: bool, failure: bool,
) -> None:
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    if failure:
        plan = PeriodicHeatPlan.from_log_potential(1, {(1,): 1000}, degree=0)
        with pytest.raises(ArithmeticError):
            enclose_burgers_jet(plan, _point(), Interval.point(0.0))
    else:
        enclose_burgers_jet(_cosine(), _point(), Interval.point(0.0), order=1)
    assert transcend.strict_backend() is strict
    assert transcend.libm_fallback_used() is sticky


def test_missing_backend_is_explicit_and_preserves_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    with pytest.raises(RuntimeError, match="rigorous transcendental backend"):
        enclose_heat_derivative(_cosine(), _point(), Interval.point(0.0))
    assert transcend.libm_fallback_used() and not transcend.strict_backend()


def test_overflowed_phases_and_decay_are_inconclusive() -> None:
    pytest.importorskip("mpmath")
    oscillatory = PeriodicHeatPlan.from_log_potential(1, {(10,): 1}, degree=1)
    with pytest.raises(ArithmeticError, match="nonfinite"):
        enclose_heat_derivative(oscillatory, _point(1e308), Interval.point(0.0))
    fast = PeriodicHeatPlan.from_log_potential(1, {(1,): 1}, degree=1, viscosity=10**308)
    with pytest.raises(ArithmeticError, match="nonfinite"):
        enclose_heat_derivative(fast, _point(), Interval.point(10.0))


def test_nonrepresentable_drift_does_not_taint_pure_heat() -> None:
    pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_log_potential(1, {}, drift=(10**400,))
    assert enclose_heat_derivative(plan, _point(), Interval.point(1.0)).value.contains(1.0)
    with pytest.raises(ArithmeticError):
        enclose_burgers_jet(plan, _point(), Interval.point(1.0))
