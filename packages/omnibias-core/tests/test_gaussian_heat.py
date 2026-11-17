# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent Gaussian formulas, exact geometry and numerical refusals."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import product
from typing import Any, cast

import omnibias.core.verified.mean_map as mean_map_module
import omnibias.core.verified.transcend as transcend
import pytest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval


@pytest.fixture
def mp() -> Any:
    return pytest.importorskip("mpmath")


def _symmetric(*, time: int | Fraction = 1) -> GaussianHeatMap:
    return GaussianHeatMap(((-1,), (1,)), (0, 0), time=time)


def test_metadata_snapshot_exact_geometry_and_immutability() -> None:
    centers = [[0, 0], [3, 4], [1, 2]]
    biases = [0, 1, -1]
    model = GaussianHeatMap(centers, biases, viscosity=Fraction(3, 2), time=Fraction(5, 3))
    centers[1][0], biases[1] = 20, 50
    assert model.centers == ((0, 0), (3, 4), (1, 2))
    assert model.biases == (0, 1, -1)
    assert model.dimension == 2
    assert model.mode_count == 3
    assert model.diameter_squared == 25
    assert model.contraction_bound == Fraction(5, 4)
    assert model.viscosity == Fraction(3, 2)
    assert model.time == Fraction(5, 3)
    assert all(isinstance(value, Fraction) for center in model.centers for value in center)
    assert hash(model) == hash(GaussianHeatMap(((0, 0), (3, 4), (1, 2)), (0, 1, -1),
                                              viscosity=Fraction(3, 2), time=Fraction(5, 3)))
    with pytest.raises(FrozenInstanceError):
        model.time = Fraction(0)  # type: ignore[misc]


def test_symmetric_gaussian_has_known_tanh_velocity_and_covariance(mp: Any) -> None:
    model = _symmetric()
    assert model.diameter_squared == 4
    assert model.contraction_bound == Fraction(1, 2)
    with mp.workdps(80):
        for x in (-3.0, 0.0, 0.7, 2.0):
            box = (Interval.point(x),)
            expected_mean = mp.tanh(mp.mpf(x) / 2)
            velocity = model.velocity(box)[0]
            fixed = model.scaled_fixed_point(box, (Fraction(1, 3),))[0]
            derivative = model.scaled_jacobian(box)[0][0]
            assert velocity.lo <= x - expected_mean <= velocity.hi
            assert fixed.lo <= mp.mpf(1) / 3 + expected_mean <= fixed.hi
            assert derivative.lo <= 1 - 1 / (2 * mp.cosh(mp.mpf(x) / 2) ** 2) <= derivative.hi
            assert model.float_scaled_fixed_point((x,), (Fraction(1, 3),))[0] == pytest.approx(float(mp.mpf(1) / 3 + expected_mean))


def test_scaled_coordinates_and_physical_velocity_have_correct_time_factors(mp: Any) -> None:
    model = _symmetric(time=2)
    assert model.contraction_bound == Fraction(1, 4)
    with mp.workdps(70):
        y = mp.mpf("0.5")
        expected = y - mp.tanh(y / 2) / 2
        physical = model.velocity((Interval.point(1.0),))[0]
        scaled = model.scaled_velocity((Interval.point(0.5),))[0]
        assert physical.lo <= expected <= physical.hi
        assert scaled.lo <= expected <= scaled.hi
        jacobian = model.scaled_jacobian((Interval.point(0.5),))[0][0]
        derivative = mp.diff(lambda v: v - mp.tanh(v / 2) / 2, y)
        assert jacobian.lo <= derivative <= jacobian.hi
        fixed = model.scaled_fixed_point((Interval.point(0.5),), (Fraction(1, 4),))[0]
        assert fixed.lo <= mp.mpf(1) / 4 + mp.tanh(y / 2) / 2 <= fixed.hi


def test_multidimensional_boxes_against_independent_original_gaussian_formula(mp: Any) -> None:
    model = GaussianHeatMap(((0, 0), (1, -2), (-2, 1), (Fraction(2, 3), Fraction(1, 5))),
                            (0, Fraction(1, 3), -1, 2), viscosity=Fraction(3, 2), time=2)
    physical_box = (Interval(-0.4, 0.6), Interval(0.8, 1.2))
    scaled_box = (Interval(-0.2, 0.3), Interval(0.4, 0.6))
    velocity, scaled = model.velocity(physical_box), model.scaled_velocity(scaled_box)
    jacobian = model.scaled_jacobian(scaled_box)
    fixed = model.scaled_fixed_point(scaled_box, (Fraction(1, 7), Fraction(-2, 9)))

    def exact(value: Fraction) -> Any:
        return mp.mpf(value.numerator) / value.denominator

    with mp.workdps(80):
        for y in product((-0.2, 0.0, 0.3), (0.4, 0.5, 0.6)):
            x = [2 * mp.mpf(value) for value in y]
            unnormalized = [mp.exp(exact(bias) - mp.fsum((axis - exact(center)) ** 2
                                                       for axis, center in zip(x, point, strict=True)) / 12)
                            for point, bias in zip(model.centers, model.biases, strict=True)]
            weights = [value / mp.fsum(unnormalized) for value in unnormalized]
            mean = [mp.fsum(weight * exact(point[i]) for weight, point in zip(weights, model.centers, strict=True))
                    for i in range(2)]
            for i in range(2):
                expected = (x[i] - mean[i]) / 2
                assert velocity[i].lo <= expected <= velocity[i].hi
                assert scaled[i].lo <= expected <= scaled[i].hi
                target = mp.mpf(1) / 7 if i == 0 else -mp.mpf(2) / 9
                assert fixed[i].lo <= target + mean[i] / 2 <= fixed[i].hi
                for j in range(2):
                    covariance = mp.fsum(weight * (exact(point[i]) - mean[i]) * (exact(point[j]) - mean[j])
                                          for weight, point in zip(weights, model.centers, strict=True))
                    expected_derivative = int(i == j) - covariance / 6
                    assert jacobian[i][j].lo <= expected_derivative <= jacobian[i][j].hi
                    assert jacobian[i][j] == jacobian[j][i]


def test_huge_common_bias_is_cancelled_before_every_numeric_boundary(mp: Any) -> None:
    common = 10**1000
    base = GaussianHeatMap(((-1,), (1,)), (0, 1))
    shifted = GaussianHeatMap(((-1,), (1,)), (common, common + 1))
    box = (Interval(-0.5, 0.2),)
    assert shifted.velocity(box) == base.velocity(box)
    assert shifted.scaled_velocity(box) == base.scaled_velocity(box)
    assert shifted.scaled_fixed_point(box, (0,)) == base.scaled_fixed_point(box, (0,))
    assert shifted.scaled_jacobian(box) == base.scaled_jacobian(box)
    assert shifted.float_scaled_fixed_point((0.1,), (Fraction(1, 3),)) == base.float_scaled_fixed_point((0.1,), (Fraction(1, 3),))


def test_exact_geometry_avoids_squaring_huge_common_center() -> None:
    # Squaring the common center would exceed this exact bit budget.
    common = 10**600
    model = GaussianHeatMap(((common,), (common + 1,)), (0, 0),
                            budget=replace(RidgeBudget(), max_coefficient_bits=2500))
    assert model.diameter_squared == 1
    assert model.contraction_bound == Fraction(1, 8)


def test_single_and_duplicate_centers_have_zero_contraction(mp: Any) -> None:
    for model in (GaussianHeatMap(((2, -3),), (8,), time=2),
                  GaussianHeatMap(((2, -3), (2, -3)), (100, -50), time=2)):
        assert model.diameter_squared == model.contraction_bound == 0
        physical = model.velocity((Interval.point(4.0), Interval.point(1.0)))
        assert physical[0].contains(1.0) and physical[1].contains(2.0)
        fixed = model.scaled_fixed_point((Interval(-4.0, 7.0), Interval(-8.0, 9.0)), (1, 2))
        assert fixed[0].contains(2.0) and fixed[1].contains(0.5)
        jacobian = model.scaled_jacobian((Interval(-4.0, 7.0), Interval(-8.0, 9.0)))
        assert all(value.contains(float(i == j)) for i, row in enumerate(jacobian) for j, value in enumerate(row))


def test_noncontractive_family_really_can_have_multiple_velocity_inverses(mp: Any) -> None:
    model = GaussianHeatMap(((-2,), (2,)), (0, 0))
    assert model.contraction_bound == 2
    with mp.workdps(60):
        positive = mp.findroot(lambda x: x - 2 * mp.tanh(x), 1.8)
        assert positive > 1.9
        for root in (-positive, mp.mpf(0), positive):
            midpoint = float(root)
            box = (Interval(math.nextafter(midpoint, -math.inf), math.nextafter(midpoint, math.inf)),)
            assert model.velocity(box)[0].contains_zero()
    assert model.scaled_jacobian((Interval.point(0.0),))[0][0].hi < 0


@pytest.mark.parametrize("bad", [True, 1.0, float("inf"), float("nan"), complex(1), "1"])
def test_metadata_and_targets_require_exact_rationals(bad: Any) -> None:
    with pytest.raises(TypeError):
        GaussianHeatMap(((bad,),), (0,))
    with pytest.raises(TypeError):
        GaussianHeatMap(((0,),), (bad,))
    with pytest.raises(TypeError):
        GaussianHeatMap(((0,),), (0,), viscosity=bad)
    with pytest.raises(TypeError):
        GaussianHeatMap(((0,),), (0,), time=bad)
    model = _symmetric()
    with pytest.raises(TypeError):
        model.scaled_fixed_point((Interval.point(0.0),), (bad,))
    with pytest.raises(TypeError):
        model.float_scaled_fixed_point((0.0,), (bad,))


@pytest.mark.parametrize("value", [0, -1, Fraction(-1, 3)])
def test_time_and_viscosity_must_be_positive(value: int | Fraction) -> None:
    with pytest.raises(ValueError, match="positive"):
        GaussianHeatMap(((0,),), (0,), viscosity=value)
    with pytest.raises(ValueError, match="positive"):
        GaussianHeatMap(((0,),), (0,), time=value)


def test_shapes_container_types_and_invalid_budget() -> None:
    for centers, biases in (((), ()), (((),), (0,)), (((0,), (1, 2)), (0, 0)), (((0,),), ())):
        with pytest.raises(ValueError):
            GaussianHeatMap(centers, biases)
    with pytest.raises(TypeError):
        GaussianHeatMap(cast(Any, iter(((0,),))), (0,))
    with pytest.raises(TypeError):
        GaussianHeatMap(((0,),), (0,), budget=cast(Any, 1))
    model = _symmetric()
    with pytest.raises(ValueError):
        model.velocity(())
    with pytest.raises(ValueError):
        model.scaled_fixed_point((Interval.point(0.0),), ())
    with pytest.raises(ValueError):
        model.float_scaled_fixed_point((), (0,))
    with pytest.raises(ValueError):
        model.float_scaled_fixed_point((0.0,), ())
    with pytest.raises(TypeError):
        model.velocity(cast(Any, (0.0,)))


@pytest.mark.parametrize("endpoint", [True, Fraction(1, 10), float("inf"), 2**53 + 1])
def test_box_and_float_point_endpoint_policy(endpoint: Any) -> None:
    model = _symmetric()
    interval = Interval(endpoint, endpoint)
    for method in (model.velocity, model.scaled_velocity, model.scaled_jacobian):
        with pytest.raises((TypeError, ValueError)):
            method((interval,))
    with pytest.raises((TypeError, ValueError)):
        model.float_scaled_fixed_point((endpoint,), (0,))


def test_constructor_resource_limits_and_intermediate_bit_growth() -> None:
    for budget in (replace(RidgeBudget(), max_matrix_entries=2), replace(RidgeBudget(), max_elimination_work=2)):
        with pytest.raises(ValueError, match="budget"):
            GaussianHeatMap(((-1,), (1,)), (0, 0), budget=budget)
    with pytest.raises(ValueError, match="coefficient"):
        GaussianHeatMap(((0,), (16,)), (0, 0), budget=replace(RidgeBudget(), max_coefficient_bits=4))
    with pytest.raises(ValueError, match="coefficient"):
        GaussianHeatMap(((0,), (8,)), (0, 0), budget=replace(RidgeBudget(), max_coefficient_bits=4))


def test_numerical_overflow_refuses_before_infinity_can_cancel(mp: Any) -> None:
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=Fraction(1, 10**400))
    with pytest.raises(ArithmeticError):
        model.velocity((Interval.point(0.0),))
    with pytest.raises(ArithmeticError):
        model.float_scaled_fixed_point((0.0,), (0,))


def test_missing_backend_does_not_prevent_exact_geometry_or_float_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    model = _symmetric()
    assert model.contraction_bound == Fraction(1, 2)
    assert model.float_scaled_fixed_point((0.0,), (0,)) == (0.0,)
    with pytest.raises(RuntimeError, match="rigorous"):
        model.scaled_fixed_point((Interval.point(0.0),), (0,))
    assert transcend.libm_fallback_used()
    assert not transcend.strict_backend()


@pytest.mark.parametrize("previous", [False, True])
def test_strict_state_and_sticky_provenance_survive_success_and_error(mp: Any, monkeypatch: pytest.MonkeyPatch,
                                                                   previous: bool) -> None:
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", previous)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    model = _symmetric()
    model.velocity((Interval.point(0.0),))
    model.scaled_jacobian((Interval.point(0.0),))
    assert transcend.strict_backend() is previous
    assert transcend.libm_fallback_used()

    def fail(argument: Interval) -> Interval:
        assert transcend.strict_backend()
        raise ArithmeticError("deliberate strict evaluation failure")

    monkeypatch.setattr(mean_map_module, "exp_iv", fail)
    with pytest.raises(ArithmeticError, match="deliberate"):
        model.scaled_velocity((Interval.point(0.0),))
    assert transcend.strict_backend() is previous
    assert transcend.libm_fallback_used()
