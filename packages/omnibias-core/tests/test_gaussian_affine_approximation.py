# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed-age normal-cost resummation and its rigorous refusal contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import mpmath as mp
import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import (
    GaussianAffineApproximation,
    certify_gaussian_affine_approximation,
    transcend,
)
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_aging import certify_gaussian_affine_aging
from omnibias.core.verified.gaussian_aging import certify_gaussian_inverse_aging
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval

ZERO_BOX = (Interval.point(0),)*2


def _restriction(level=4, time=4, **kwargs):
    source = GaussianHeatMap(((-1, -1), (1, 1)), (0, 0), time=time, **kwargs)
    result = restrict_gaussian_heat(source, ((0, 1),), (level,))
    assert result is not None
    return result


def _mp(value):
    exact = F(value)
    return mp.mpf(exact.numerator)/exact.denominator


@pytest.mark.parametrize("order", (0, 1, 2, 4, 8, 12))
@pytest.mark.parametrize("level", (0, 4, 1000000))
def test_normal_cost_range_does_not_control_fixed_age_convergence(order, level):
    restriction = _restriction(level)
    result = certify_gaussian_affine_approximation(restriction, ZERO_BOX, order=order)
    assert result is not None and result.base_approximation is not None
    q = restriction.model.contraction_bound
    assert q == F(1, 8)
    expected = (q if order == 0 else 2*q*q) if order < 2 else (3*q)**(order+1)/(1-3*q)
    assert result.approximation_error_bound == expected
    assert result.objective_gap_bound == expected**2/8
    assert result.feasible_point[1] == level
    if level:
        assert certify_gaussian_affine_aging(restriction, ZERO_BOX, order=order) is None
    with mp.workdps(90):
        mean = mp.tanh(mp.mpf(level)/8)
        root = mp.findroot(lambda x: x-mp.tanh(x/8+mp.mpf(level)/8), mean, tol=mp.mpf("1e-80"))
        assert _mp(result.enclosure[0].lo) <= root <= _mp(result.enclosure[0].hi)
        assert abs(_mp(result.feasible_point[0])-root) <= _mp(result.feasible_point_error_bound)
        if order:
            first = mean*(1-mean*mean)/8
            interval = result.correction_coefficients[1][0]
            assert _mp(interval.lo) <= first <= _mp(interval.hi)


@pytest.mark.parametrize("time", (F(1, 2), F(1), F(3, 2)))
def test_real_predictors_and_geometric_series_have_distinct_regimes(time):
    restriction = _restriction(time=time)
    for order in (0, 1):
        result = certify_gaussian_affine_approximation(restriction, ZERO_BOX, order=order)
        assert (result is not None) == (time > F(1, 2))
    assert certify_gaussian_affine_approximation(restriction, ZERO_BOX, order=2) is None


def test_oblique_target_box_has_exact_feasible_candidate_and_separate_point_error():
    model = GaussianHeatMap(((0, 0), (1, 2), (3, -1)), (0, 0, 0), time=16)
    restriction = restrict_gaussian_heat(model, ((1, 2), (2, 4)), (F(3, 4), F(3, 2)))
    result = certify_gaussian_affine_approximation(restriction, (Interval(-0.25, 0.25), Interval(-0.125, 0.125)), order=8)
    assert result is not None
    assert result.feasible_point[0]+2*result.feasible_point[1] == F(3, 4)
    assert result.feasible_point_error_bound > result.approximation_error_bound
    assert result.feasible_point_objective_gap_bound == result.feasible_point_error_bound**2/32
    assert result.objective_gap_bound == result.approximation_error_bound**2/32


@pytest.mark.parametrize("order", (0, 1, 8))
def test_normal_target_variation_cancels_before_base_evaluation(order):
    restriction = _restriction()
    a = certify_gaussian_affine_approximation(restriction, (Interval(-0.25, 0.25), Interval.point(0)), order=order)
    b = certify_gaussian_affine_approximation(restriction, (Interval(-0.25, 0.25), Interval(-1e308, 1e308)), order=order)
    assert a is not None and b is not None
    assert a.target_box != b.target_box
    for name in ("projected_target_box", "correction_coefficients", "predictor_enclosure", "enclosure",
                 "approximation_error_bound", "feasible_point", "feasible_point_error_bound"):
        assert getattr(a, name) == getattr(b, name)


def test_no_constraints_matches_existing_series():
    source = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=16)
    restriction = restrict_gaussian_heat(source, (), ())
    box = (Interval(-0.125, 0.25),)
    result = certify_gaussian_affine_approximation(restriction, box, order=8)
    reference = certify_gaussian_inverse_aging(source, box, order=8)
    assert result is not None and reference is not None
    assert result.correction_coefficients == reference.correction_coefficients
    assert result.predictor_enclosure == reference.predictor_enclosure
    assert result.approximation_error_bound == reference.approximation_error_bound


@pytest.mark.parametrize("order", (0, 1, 8, 128))
def test_affine_case_preserves_huge_cancellation_without_backend(order, monkeypatch):
    huge = 10**400
    source = GaussianHeatMap(((-huge, huge), (-huge, huge+1)), (0, huge), time=huge)
    restriction = restrict_gaussian_heat(source, ((0, 1),), (0,))
    assert restriction is not None
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_affine_approximation(restriction, (Interval.point(1), Interval.point(0)), order=order)
    assert result is not None and result.base_approximation is None
    assert result.feasible_point == (0, 0)
    assert result.correction_coefficients == (ZERO_BOX,)*(order+1)
    assert result.predictor_enclosure == result.enclosure == ZERO_BOX
    assert result.approximation_error_bound == result.feasible_point_error_bound == 0
    assert result.objective_gap_bound == result.feasible_point_objective_gap_bound == 0
    assert result.certificate["meta"]["transcend_backend"] == "not_used"


def test_singleton_constraint_and_nonbinary_anchor():
    source = GaussianHeatMap(((-1, -2), (1, 2)), (0, 0))
    restriction = restrict_gaussian_heat(source, ((1, 0), (0, 1)), (F(1, 3), F(-1, 7)))
    result = certify_gaussian_affine_approximation(restriction, (Interval(-1e100, 1e100),)*2)
    assert result is not None and result.approximation_error_bound == 0
    assert result.feasible_point == (F(1, 3), F(-1, 7))
    assert result.feasible_point_error_bound > 0  # Includes coordinate interval width.


def test_forward_evaluation_uses_no_root_solver_or_factorization(monkeypatch):
    restriction = _restriction()

    def fail(*args, **kwargs):
        pytest.fail("fixed-age approximation must not solve an inverse")

    monkeypatch.setattr(import_module("omnibias.core.verified.inverse_gaussian_heat"), "inverse_gaussian_heat", fail)
    monkeypatch.setattr(import_module("omnibias.core._rational_linear"), "factor_positive_definite", fail)
    for order in (0, 1, 8):
        assert certify_gaussian_affine_approximation(restriction, ZERO_BOX, order=order) is not None


def test_frozen_factory_and_dependency_snapshots():
    with pytest.raises(TypeError):
        GaussianAffineApproximation()
    restriction = _restriction()
    result = certify_gaussian_affine_approximation(restriction, ZERO_BOX)
    assert result is not None and verify_certificate_digest(result.certificate)
    with pytest.raises(FrozenInstanceError):
        result.order = 0
    saved = deepcopy(result.certificate)
    assert result.base_approximation is not None
    result.base_approximation.certificate["payload"]["damaged"] = object()
    restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert certify_gaussian_affine_approximation(restriction, ZERO_BOX) is None
    result.certificate["payload"]["feasible_point"][0] = [99, 1]
    assert not verify_certificate_digest(result.certificate)


def test_damaged_base_certificate_is_refused(monkeypatch):
    restriction = _restriction()
    module = import_module("omnibias.core.verified.gaussian_affine_approximation")
    base = certify_gaussian_inverse_aging(restriction.model, ZERO_BOX)
    assert base is not None
    base.certificate["payload"]["damaged"] = object()
    monkeypatch.setattr(module, "certify_gaussian_inverse_aging", lambda *args, **kwargs: base)
    assert certify_gaussian_affine_approximation(restriction, ZERO_BOX) is None


def test_missing_backend_raises_and_sticky_history_is_preserved(monkeypatch):
    restriction = _restriction()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_affine_approximation(restriction, ZERO_BOX)
    assert result is not None and transcend.libm_fallback_used()
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError):
        certify_gaussian_affine_approximation(restriction, ZERO_BOX)


@pytest.mark.parametrize("order,error", [(True, TypeError), (1.0, TypeError), (-1, ValueError)])
def test_invalid_orders_raise(order, error):
    with pytest.raises(error):
        certify_gaussian_affine_approximation(_restriction(), ZERO_BOX, order=order)


@pytest.mark.parametrize("box,error", [(None, TypeError), ((0.0, 0.0), TypeError), ((), ValueError),
                                       ((Interval.point(0),), ValueError),
                                       ((Interval(0, float("inf")), Interval.point(0)), ValueError)])
def test_invalid_box_contract(box, error):
    with pytest.raises(error):
        certify_gaussian_affine_approximation(_restriction(), box)


def test_invalid_restriction_and_nonfinite_output():
    with pytest.raises(TypeError):
        certify_gaussian_affine_approximation(None, ZERO_BOX)
    assert certify_gaussian_affine_approximation(_restriction(), (Interval.point(1e308), Interval.point(0))) is None


@pytest.mark.parametrize("limits,target,order", [
    (RidgeBudget(max_order=2), 0, 3),
    (RidgeBudget(max_matrix_entries=1500), 0, 8),
    (RidgeBudget(max_elimination_work=6000), 0, 8),
    (RidgeBudget(max_coefficient_bits=64), 1e100, 1),
])
def test_resource_failure_is_inconclusive(limits, target, order):
    source = GaussianHeatMap(((0,), (0,)), (0, 1), budget=limits)
    restriction = restrict_gaussian_heat(source, (), ())
    assert restriction is not None
    assert certify_gaussian_affine_approximation(restriction, (Interval.point(target),), order=order) is None
