# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contracts for certified constrained forward formulas and feasible points."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import (
    GaussianAffineAging,
    certify_gaussian_affine_aging,
    certify_gaussian_affine_predictor,
    transcend,
)
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_aging import certify_gaussian_inverse_aging
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_predictor import certify_gaussian_inverse_predictor
from omnibias.core.verified.interval import Interval

FACTORIES = (certify_gaussian_affine_predictor, certify_gaussian_affine_aging)
ZERO_BOX = (Interval.point(0),)*2


def _restriction(time=16, budget=None, biases=(0, 0)):
    source = GaussianHeatMap(((-1, -2), (1, 2)), biases, time=time, budget=budget)
    result = restrict_gaussian_heat(source, ((0, 1),), (1,))
    assert result is not None
    return result


@pytest.mark.parametrize("order", (0, 1, 4, 8))
def test_normal_cost_series_matches_independent_tanh_lagrange_coefficients(order):
    # h=tanh(t*(h+2)/32). Direct Lagrange inversion gives these exact
    # ordinary coefficients, including a negative third coefficient.
    expected = (0, F(1, 16), F(1, 512), F(-1, 49152), F(-13, 1572864))
    result = certify_gaussian_affine_aging(_restriction(), ZERO_BOX, order=order)
    assert result is not None and result.aging_parameter == F(3, 32)
    assert result.mode == "series" and result.order == order
    for index, value in enumerate(expected[:order+1]):
        interval = result.correction_coefficients[index][0]
        assert F(interval.lo) <= value <= F(interval.hi)
    assert all(row[1] == Interval.point(0) for row in result.correction_coefficients)
    assert result.predictor_enclosure[1] == result.enclosure[1] == Interval.point(1)
    assert result.feasible_point[1] == 1


@pytest.mark.parametrize("factory", FACTORIES)
def test_normal_target_variation_has_no_effect(factory):
    restriction = _restriction()
    a = factory(restriction, (Interval(-0.25, 0.25), Interval.point(0)), order=1)
    b = factory(restriction, (Interval(-0.25, 0.25), Interval(-1e100, 1e100)), order=1)
    assert a is not None and b is not None
    assert a.target_box != b.target_box
    for name in ("relative_mean", "projected_target_box", "correction_coefficients", "predictor_enclosure",
                 "approximation_error_bound", "enclosure", "feasible_point", "feasible_point_error_bound"):
        assert getattr(a, name) == getattr(b, name)


@pytest.mark.parametrize("factory", FACTORIES)
def test_formula_error_and_actual_point_gap_have_distinct_scopes(factory):
    source = GaussianHeatMap(((0, 0), (1, 2), (3, -1)), (0, 0, 0), time=16)
    restriction = restrict_gaussian_heat(source, ((1, 2),), (F(3, 4),))
    result = factory(restriction, (Interval(-0.25, 0.25), Interval(-0.125, 0.125)), order=1)
    assert result is not None
    assert result.feasible_point[0]+2*result.feasible_point[1] == F(3, 4)
    assert result.feasible_point_error_bound > result.approximation_error_bound
    assert result.objective_gap_bound == result.approximation_error_bound**2/32
    assert result.feasible_point_objective_gap_bound == result.feasible_point_error_bound**2/32
    assert result.feasible_point_objective_gap_bound > result.objective_gap_bound


@pytest.mark.parametrize("order", (0, 1, 8))
def test_empty_constraints_recover_unconstrained_aging(order):
    source = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=16)
    restriction = restrict_gaussian_heat(source, (), ())
    box = (Interval.point(0.25),)
    constrained = certify_gaussian_affine_aging(restriction, box, order=order)
    original = certify_gaussian_inverse_aging(source, box, order=order)
    assert constrained is not None and original is not None
    assert constrained.correction_coefficients == original.correction_coefficients
    assert constrained.approximation_error_bound == original.approximation_error_bound
    assert constrained.enclosure == original.enclosure
    if order < 2:
        predictor = certify_gaussian_affine_predictor(restriction, box, order=order)
        reference = certify_gaussian_inverse_predictor(source, box, order=order)
        assert predictor is not None and reference is not None
        assert predictor.approximation_error_bound == reference.approximation_error_bound
        assert predictor.predictor_enclosure[0].intersect(reference.predictor_enclosure[0])


@pytest.mark.parametrize("factory", FACTORIES)
def test_affine_huge_cost_cancellation_skips_transcendentals(factory, monkeypatch):
    huge = 10**400
    source = GaussianHeatMap(((-huge, huge), (-huge, huge+1)), (0, huge), time=huge)
    restriction = restrict_gaussian_heat(source, ((0, 1),), (0,))
    assert restriction is not None and restriction.model.diameter_squared == 0
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = factory(restriction, (Interval.point(1), Interval.point(0)), order=1)
    assert result is not None
    assert result.feasible_point == (F(0), F(0))
    assert result.enclosure == result.predictor_enclosure == ZERO_BOX
    assert result.approximation_error_bound == result.feasible_point_error_bound == 0
    assert result.objective_gap_bound == result.feasible_point_objective_gap_bound == 0
    assert result.certificate["meta"]["transcend_backend"] == "not_used"


@pytest.mark.parametrize("factory", FACTORIES)
def test_full_rank_constraint_is_singleton_even_with_nonzero_targets(factory):
    source = GaussianHeatMap(((-1, -2), (1, 2)), (0, 0))
    restriction = restrict_gaussian_heat(source, ((1, 0), (0, 1)), (F(1, 3), F(-1, 7)))
    result = factory(restriction, (Interval(-1e100, 1e100),)*2, order=1)
    assert result is not None and result.approximation_error_bound == 0
    assert result.feasible_point == (F(1, 3), F(-1, 7))
    # Coordinate conversion has width, although the exact projected point is
    # the singleton. A positive point bound is safe and must not become zero
    # just because the analytic series tail vanishes.
    assert result.feasible_point_error_bound > 0
    assert result.feasible_point_objective_gap_bound > 0


def test_distinct_predictor_and_series_convergence_regimes():
    restriction = _restriction(time=1)
    assert restriction.model.contraction_bound == F(1, 2)
    for order in (0, 1):
        result = certify_gaussian_affine_predictor(restriction, ZERO_BOX, order=order)
        assert result is not None and result.aging_parameter == F(3, 2)
        assert result.approximation_error_bound == (F(3, 2) if order == 0 else F(3))
        assert certify_gaussian_affine_aging(restriction, ZERO_BOX, order=order) is None
    for factory in FACTORIES:
        assert factory(_restriction(time=F(1, 2)), ZERO_BOX, order=1) is None
    # Q=1/3 is excluded even though the constrained optimizer is unique.
    assert certify_gaussian_affine_aging(_restriction(time=F(9, 2)), ZERO_BOX, order=0) is None


def test_forward_path_needs_no_inverse_iteration_covariance_or_factorization(monkeypatch):
    module = import_module("omnibias.core.verified.gaussian_affine_aging")
    original = module._posterior_moments
    restriction = _restriction()

    def moments(posterior, converted, *, covariance):
        assert covariance is False
        return original(posterior, converted, covariance=covariance)

    def fail(*args, **kwargs):
        pytest.fail("forward approximation must not solve or factorize an inverse")

    monkeypatch.setattr(module, "_posterior_moments", moments)
    monkeypatch.setattr(import_module("omnibias.core.verified.inverse_gaussian_heat"), "inverse_gaussian_heat", fail)
    monkeypatch.setattr(import_module("omnibias.core._rational_linear"), "factor_positive_definite", fail)
    for factory in FACTORIES:
        assert factory(restriction, ZERO_BOX, order=1) is not None


@pytest.mark.parametrize("factory", FACTORIES)
def test_frozen_factory_bound_digest_and_snapshot(factory):
    with pytest.raises(TypeError):
        GaussianAffineAging()
    restriction = _restriction()
    result = factory(restriction, ZERO_BOX, order=1)
    assert result is not None and verify_certificate_digest(result.certificate)
    with pytest.raises(FrozenInstanceError):
        result.order = 8
    original = deepcopy(result.certificate)
    restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == original and verify_certificate_digest(result.certificate)
    assert factory(restriction, ZERO_BOX, order=1) is None
    result.certificate["payload"]["feasible_point"][0] = [99, 1]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("factory", FACTORIES)
def test_missing_backend_and_sticky_history(factory, monkeypatch):
    restriction = _restriction()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = factory(restriction, ZERO_BOX, order=1)
    assert result is not None and transcend.libm_fallback_used()
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError):
        factory(restriction, ZERO_BOX, order=1)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("order,error", [(True, TypeError), (1.0, TypeError), (-1, ValueError)])
def test_invalid_orders_raise(factory, order, error):
    with pytest.raises(error):
        factory(_restriction(), ZERO_BOX, order=order)


def test_predictor_rejects_higher_order_instead_of_changing_formula():
    with pytest.raises(ValueError):
        certify_gaussian_affine_predictor(_restriction(), ZERO_BOX, order=2)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("box,error", [(None, TypeError), ((0.0, 0.0), TypeError), ((), ValueError),
                                       ((Interval.point(0),), ValueError),
                                       ((Interval(0, float("inf")), Interval.point(0)), ValueError)])
def test_invalid_box_contract(factory, box, error):
    with pytest.raises(error):
        factory(_restriction(), box, order=1)


@pytest.mark.parametrize("factory", FACTORIES)
def test_invalid_restriction_and_nonfinite_evaluation(factory):
    with pytest.raises(TypeError):
        factory(None, ZERO_BOX, order=1)
    assert factory(_restriction(), (Interval.point(1e308), Interval.point(0)), order=1) is None


@pytest.mark.parametrize("limits,target,order", [
    (RidgeBudget(max_order=2), 0, 3),
    (RidgeBudget(max_matrix_entries=1500), 0, 8),
    (RidgeBudget(max_elimination_work=10000), 0, 8),
    (RidgeBudget(max_coefficient_bits=64), 1e100, 1),
])
def test_resource_failure_is_inconclusive(limits, target, order):
    # Empty constraints let the exact input compile within the deliberately
    # small budgets; the forward query must meter its own extra work.
    source = GaussianHeatMap(((0,), (0,)), (0, 1), budget=limits)
    restriction = restrict_gaussian_heat(source, (), ())
    assert restriction is not None
    assert certify_gaussian_affine_aging(restriction, (Interval.point(target),), order=order) is None
