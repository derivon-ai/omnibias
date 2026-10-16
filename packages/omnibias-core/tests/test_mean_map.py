# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent numerical oracles, exact witnesses and refusal contracts."""

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
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mean_map import MeanMap


@pytest.fixture
def mp() -> Any:
    return pytest.importorskip("mpmath")


def _simplex() -> MeanMap:
    return MeanMap(((0, 0), (1, 0), (0, 1)), (0, -1, 1), (Fraction(1, 3), Fraction(1, 3)))


def test_exact_input_snapshot_and_direct_constructor_are_frozen() -> None:
    modes = [[0, 0], [1, 0], [0, 1]]
    biases = [0, 0, 0]
    target = [Fraction(1, 3), Fraction(1, 3)]
    model = MeanMap(modes, biases, target)
    modes[1][0], biases[1], target[0] = 7, 3, Fraction(9)
    assert model.modes == ((0, 0), (1, 0), (0, 1))
    assert model.biases == (0, 0, 0)
    assert model.target == (Fraction(1, 3),) * 2
    assert model.dimension == 2
    assert model.mode_count == 3
    assert model.affine_rank == 2
    assert all(isinstance(value, Fraction) for row in model.modes for value in row)
    assert hash(model) == hash(MeanMap(((0, 0), (1, 0), (0, 1)), (0, 0, 0), (Fraction(1, 3),) * 2))
    with pytest.raises(FrozenInstanceError):
        model.target = (Fraction(0), Fraction(0))  # type: ignore[misc]


def test_exact_rank_handles_duplicates_and_tiny_rational_independence() -> None:
    assert MeanMap(((1, 2),), (0,), (1, 2)).affine_rank == 0
    assert MeanMap(((1, 2), (1, 2), (1, 2)), (0, 1, -1), (1, 2)).affine_rank == 0
    assert MeanMap(((0, 0), (1, 2), (2, 4)), (0, 0, 0), (0, 0)).affine_rank == 1
    tiny = Fraction(1, 10**100)
    assert MeanMap(((0, 0), (1, 2), (2, 4 + tiny)), (0, 0, 0), (0, 0)).affine_rank == 2
    assert MeanMap(((0,), (0,), (1,)), (0, 0, 0), (Fraction(1, 3),)).affine_rank == 1


def test_stdlib_proposal_matches_independent_simplex_values() -> None:
    result = _simplex().float_evaluate((1.0, -1.0))
    assert result.objective == pytest.approx(math.log(3), abs=1e-15)
    assert result.residual == (0.0, 0.0)
    assert result.covariance[0] == pytest.approx((2 / 9, -1 / 9), abs=1e-15)
    assert result.covariance[1] == pytest.approx((-1 / 9, 2 / 9), abs=1e-15)
    with pytest.raises(FrozenInstanceError):
        result.objective = 4  # type: ignore[misc]


def test_scalar_logistic_derivative_and_objective(mp: Any) -> None:
    model = MeanMap(((0,), (1,)), (0, 0), (Fraction(1, 4),))
    with mp.workdps(70):
        x = 0.7
        probability = 1 / (1 + mp.exp(-mp.mpf(x)))
        box = (Interval.point(x),)
        residual = model.residual(box)[0]
        covariance = model.jacobian(box)[0][0]
        objective = model.objective(box)
        assert residual.lo <= probability - mp.mpf(1) / 4 <= residual.hi
        assert covariance.lo <= probability * (1 - probability) <= covariance.hi
        assert objective.lo <= mp.log(1 + mp.exp(x)) - mp.mpf(x) / 4 <= objective.hi


def test_rigorous_boxes_contain_independent_high_precision_means_and_covariances(mp: Any) -> None:
    model = MeanMap(((0, 0), (1, -2), (-1, 1), (Fraction(2, 3), Fraction(1, 5))),
                    (0, Fraction(1, 3), -1, 2), (Fraction(1, 7), Fraction(-2, 9)))
    box = (Interval(-0.2, 0.3), Interval(0.4, 0.6))
    residual, covariance, objective = model.residual(box), model.jacobian(box), model.objective(box)

    def exact(value: Fraction) -> Any:
        return mp.mpf(value.numerator) / value.denominator

    with mp.workdps(80):
        for point in product((-0.2, 0.0, 0.3), (0.4, 0.5, 0.6)):
            logits = [mp.fsum(exact(k) * mp.mpf(x) for k, x in zip(mode, point, strict=True)) + exact(bias)
                      for mode, bias in zip(model.modes, model.biases, strict=True)]
            weights = [mp.exp(value) for value in logits]
            partition = mp.fsum(weights)
            probabilities = [value / partition for value in weights]
            mean = [mp.fsum(p * exact(mode[axis]) for p, mode in zip(probabilities, model.modes, strict=True))
                    for axis in range(2)]
            for axis in range(2):
                assert residual[axis].lo <= mean[axis] - exact(model.target[axis]) <= residual[axis].hi
                for other in range(2):
                    value = mp.fsum(p * (exact(mode[axis]) - mean[axis]) * (exact(mode[other]) - mean[other])
                                    for p, mode in zip(probabilities, model.modes, strict=True))
                    assert covariance[axis][other].lo <= value <= covariance[axis][other].hi
                    assert covariance[axis][other] == covariance[other][axis]
            value = mp.log(partition) - mp.fsum(exact(t) * mp.mpf(x) for t, x in zip(model.target, point, strict=True))
            assert objective.lo <= value <= objective.hi


def test_huge_common_exact_modes_and_biases_cancel_before_binary_conversion(mp: Any) -> None:
    common_mode, common_bias = 10**400, 10**200
    base = MeanMap(((0,), (1,)), (0, 1), (Fraction(1, 2),))
    shifted = MeanMap(((common_mode,), (common_mode + 1,)), (common_bias, common_bias + 1),
                      (common_mode + Fraction(1, 2),))
    box = (Interval(-0.3, 0.2),)
    assert shifted.affine_rank == 1
    assert shifted.residual(box) == base.residual(box)
    assert shifted.jacobian(box) == base.jacobian(box)
    assert shifted.float_evaluate((0.1,)).residual == base.float_evaluate((0.1,)).residual
    assert shifted.float_evaluate((0.1,)).covariance == base.float_evaluate((0.1,)).covariance
    assert shifted.float_evaluate((0.1,)).reduced_objective == base.float_evaluate((0.1,)).reduced_objective
    assert shifted.float_evaluate((0.1,)).objective == shifted.float_evaluate((0.0,)).objective
    assert shifted.float_evaluate((0.1,)).reduced_objective != shifted.float_evaluate((0.0,)).reduced_objective
    with mp.workdps(250):
        value = mp.mpf(common_bias) + mp.log(1 + mp.exp(1))
        enclosure = shifted.objective((Interval.point(0.0),))
        assert enclosure.lo <= value <= enclosure.hi


def test_unrepresentable_common_bias_does_not_break_residual_or_covariance(mp: Any) -> None:
    model = MeanMap(((0,), (1,)), (10**400, 10**400), (Fraction(1, 2),))
    box = (Interval.point(0.0),)
    assert model.residual(box)[0].contains(0.0)
    assert model.jacobian(box)[0][0].contains(0.25)
    with pytest.raises(ArithmeticError):
        model.objective(box)
    with pytest.raises(ArithmeticError):
        model.float_evaluate((0.0,))


def test_wide_boxes_allow_probability_lower_underflow_without_false_division(mp: Any) -> None:
    model = MeanMap(((0,), (1,)), (0, 0), (Fraction(1, 2),))
    box = (Interval(-1000.0, 1000.0),)
    result = model.residual(box)[0]
    assert result.lo <= -0.5 and result.hi >= 0.5
    covariance = model.jacobian(box)[0][0]
    assert covariance.lo == 0.0
    assert covariance.contains(0.25)
    objective = model.objective(box)
    assert objective.contains(math.log(2))
    assert all(math.isfinite(value) for value in (result.lo, result.hi, covariance.hi, objective.lo, objective.hi))


def test_single_mode_duplicate_modes_and_boundary_witnesses(mp: Any) -> None:
    single = MeanMap(((2, -3),), (1,), (2, -3))
    box = (Interval(-2.0, 3.0), Interval(-5.0, 1.0))
    assert all(value.contains(0.0) for value in single.residual(box))
    assert all(value.contains(0.0) for row in single.jacobian(box) for value in row)
    assert single.objective(box).contains(1.0)
    assert not single.target_interior((1,))
    assert single.dual_bound((1,)).contains(1.0)
    duplicate = MeanMap(((1,), (1,), (1,)), (0, 0, 0), (1,))
    assert duplicate.objective((Interval(-1.0, 1.0),)).contains(math.log(3))
    boundary = MeanMap(((0,), (1,)), (2, 7), (0,))
    assert not boundary.target_interior((1, 0))
    assert boundary.dual_bound((1, 0)).contains(2.0)


def test_exact_positive_witness_establishes_interior_and_dual_lower_bound(mp: Any) -> None:
    model = _simplex()
    weights = (Fraction(1, 3),) * 3
    assert model.target_interior(weights)
    bound = model.dual_bound(weights)
    with mp.workdps(70):
        assert bound.lo <= mp.log(3) <= bound.hi
    for point in ((1.0, -1.0), (-3.0, 4.0), (0.0, 0.0)):
        assert bound.lo <= model.float_evaluate(point).objective


def test_dual_entropy_handles_positive_rational_weights_below_binary64_range(mp: Any) -> None:
    tiny = Fraction(1, 10**400)
    model = MeanMap(((0,), (1,)), (0, 0), (tiny,))
    assert model.target_interior((1 - tiny, tiny))
    bound = model.dual_bound((1 - tiny, tiny))
    with mp.workdps(450):
        q = mp.mpf(1) / mp.mpf(10) ** 400
        entropy = -(1 - q) * mp.log(1 - q) - q * mp.log(q)
        assert bound.lo <= entropy <= bound.hi


@pytest.mark.parametrize("weights", [(Fraction(1, 2), Fraction(1, 2), 0), (1, 0, 0), (-1, 1, 1), (1, 1, 1)])
def test_invalid_mathematical_witness_is_false_and_dual_bound_refuses(weights: tuple[Fraction | int, ...]) -> None:
    model = _simplex()
    assert not model.target_interior(weights)
    with pytest.raises(ValueError, match="witness"):
        model.dual_bound(weights)


@pytest.mark.parametrize("value", [True, 1.0, 0.5, float("nan"), float("inf"), complex(1), "1"])
def test_metadata_rejects_nonexact_values(value: Any) -> None:
    with pytest.raises(TypeError):
        MeanMap(((value,), (1,)), (0, 0), (0,))
    with pytest.raises(TypeError):
        MeanMap(((0,), (1,)), (value, 0), (0,))
    with pytest.raises(TypeError):
        MeanMap(((0,), (1,)), (0, 0), (value,))
    with pytest.raises(TypeError):
        _simplex().target_interior((value, 0, 1))


@pytest.mark.parametrize("modes,biases,target", [((), (), (1,)), (((),), (0,), ()),
                                                  (((0,), (1, 2)), (0, 0), (1,)),
                                                  (((0,),), (), (0,)), (((0,),), (0,), (0, 1))])
def test_metadata_shape_errors(modes: Any, biases: Any, target: Any) -> None:
    with pytest.raises(ValueError):
        MeanMap(modes, biases, target)


def test_exact_witness_shape_errors_and_invalid_budget() -> None:
    with pytest.raises(ValueError):
        _simplex().target_interior((1,))
    with pytest.raises(ValueError):
        _simplex().dual_bound((1,))
    with pytest.raises(TypeError):
        MeanMap(((0,),), (0,), (0,), budget=cast(RidgeBudget, 1))


def test_constructor_rejects_nonsequence_data_and_preflights_before_row_scan() -> None:
    with pytest.raises(TypeError, match="sequence"):
        MeanMap(cast(Any, iter(((0,),))), (0,), (0,))
    with pytest.raises(TypeError, match="sequence"):
        MeanMap(cast(Any, (1,)), (0,), (0,))

    class UnreadableRow:
        def __len__(self) -> int:
            raise AssertionError("row scanned before resource preflight")

    with pytest.raises(ValueError, match="budget"):
        MeanMap(cast(Any, (UnreadableRow(),)), (0,), (0,), budget=replace(RidgeBudget(), max_matrix_entries=1))


@pytest.mark.parametrize("endpoint", [True, Fraction(1, 10), float("inf"), 2**53 + 1])
def test_box_rejects_ambiguous_or_nonfinite_endpoints(endpoint: Any) -> None:
    interval = Interval(endpoint, endpoint)
    for method in (_simplex().residual, _simplex().jacobian, _simplex().objective):
        with pytest.raises((TypeError, ValueError)):
            method((interval, Interval.point(0.0)))


def test_box_and_float_point_shapes_and_types() -> None:
    model = _simplex()
    for method in (model.residual, model.jacobian, model.objective):
        with pytest.raises(ValueError):
            method((Interval.point(0.0),))
        with pytest.raises(TypeError):
            method((cast(Interval, 0.0), Interval.point(0.0)))
    with pytest.raises(ValueError):
        model.float_evaluate((0.0,))
    for point in ((float("nan"), 0.0), (float("inf"), 0.0), (True, 0.0), (Fraction(1, 2), 0.0)):
        with pytest.raises((ValueError, TypeError)):
            model.float_evaluate(cast(tuple[float, ...], point))


def test_nonfinite_interval_and_float_intermediates_are_refused(mp: Any) -> None:
    model = MeanMap(((0,), (10**300,)), (0, 0), (0,))
    with pytest.raises(ArithmeticError):
        model.residual((Interval.point(1e300),))
    with pytest.raises(ArithmeticError):
        model.jacobian((Interval.point(0.0),))
    with pytest.raises(ArithmeticError):
        model.float_evaluate((1e300,))


def test_constructor_preflights_storage_work_and_coefficient_growth() -> None:
    for limits in (replace(RidgeBudget(), max_matrix_entries=2), replace(RidgeBudget(), max_elimination_work=2)):
        with pytest.raises(ValueError, match="budget"):
            MeanMap(((0,), (1,)), (0, 0), (0,), budget=limits)
    with pytest.raises(ValueError, match="coefficient"):
        MeanMap(((0,), (16,)), (0, 0), (0,), budget=replace(RidgeBudget(), max_coefficient_bits=4))
    # Input coefficients fit, but the exact reference difference does not.
    with pytest.raises(ValueError, match="coefficient"):
        MeanMap(((-15,), (15,)), (0, 0), (0,), budget=replace(RidgeBudget(), max_coefficient_bits=4))


def test_missing_rigorous_backend_is_explicit_without_changing_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    model = _simplex()
    box = (Interval.point(0.0), Interval.point(0.0))
    for method in (model.residual, model.jacobian, model.objective):
        with pytest.raises(RuntimeError, match="rigorous"):
            method(box)
    with pytest.raises(RuntimeError, match="rigorous"):
        model.dual_bound((Fraction(1, 3),) * 3)
    assert not transcend.strict_backend()
    assert transcend.libm_fallback_used()
    assert model.float_evaluate((0.0, 0.0)).objective > 0


@pytest.mark.parametrize("previous", [False, True])
def test_strict_state_and_sticky_history_survive_success_and_failure(mp: Any, monkeypatch: pytest.MonkeyPatch,
                                                                  previous: bool) -> None:
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", previous)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    model = _simplex()
    model.jacobian((Interval.point(0.0), Interval.point(0.0)))
    model.objective((Interval.point(0.0), Interval.point(0.0)))
    model.dual_bound((Fraction(1, 3),) * 3)
    assert transcend.strict_backend() is previous
    assert transcend.libm_fallback_used()

    def fail(argument: Interval) -> Interval:
        assert transcend.strict_backend()
        raise ArithmeticError("deliberate transcendental failure")

    monkeypatch.setattr(mean_map_module, "exp_iv", fail)
    with pytest.raises(ArithmeticError, match="deliberate"):
        model.residual((Interval.point(0.0), Interval.point(0.0)))
    assert transcend.strict_backend() is previous
    assert transcend.libm_fallback_used()
