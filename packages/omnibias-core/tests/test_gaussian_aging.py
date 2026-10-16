# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified inverse aging coefficients, analytic tails and refusal contracts."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import mpmath as mp
import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_aging import (
    GaussianInverseAging,
    certify_gaussian_inverse_aging,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_predictor import certify_gaussian_inverse_predictor
from omnibias.core.verified.interval import Interval


def _mp(value):
    rational = F(value)
    return mp.mpf(rational.numerator)/rational.denominator


def _model(time=16, **kwargs):
    return GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=time, **kwargs)


@pytest.mark.parametrize("order", (0, 1, 2, 4, 8, 12))
@pytest.mark.parametrize("age", (F(2), F(16), F(1024)))
def test_scalar_enclosures_contain_independent_inverse(order, age):
    model = _model(age)
    with mp.workdps(90):
        for target in (F(-2), F(-1, 8), F(0), F(1, 4), F(3)):
            result = certify_gaussian_inverse_aging(model, (Interval.point(float(target)),), order=order)
            assert result is not None
            mu = mp.tanh(_mp(target)/2+mp.mpf(1)/6)
            root = mp.findroot(lambda x, target=target: x-_mp(age)*_mp(target)-mp.tanh(x/(2*_mp(age))+mp.mpf(1)/6),
                               _mp(age)*_mp(target)+mu, tol=mp.mpf("1e-80"))
            assert _mp(result.enclosure[0].lo) <= root <= _mp(result.enclosure[0].hi)
            assert len(result.correction_coefficients) == order+1
            assert result.correction_coefficients[0] == (Interval.point(0),)
            error = F(3*model.contraction_bound)**(order+1)/(1-3*model.contraction_bound)
            if order == 0:
                error = min(error, model.contraction_bound)
            if order == 1:
                error = min(error, 2*model.contraction_bound**2)
            assert result.approximation_error_bound == error


def test_first_coefficient_is_scaled_third_moment_and_matches_predictor():
    model = _model()
    box = (Interval.point(0.25),)
    result = certify_gaussian_inverse_aging(model, box, order=1)
    first = certify_gaussian_inverse_predictor(model, box)
    assert result is not None and first is not None
    with mp.workdps(90):
        mean = mp.tanh(mp.mpf(1)/8+mp.mpf(1)/6)
        exact_coefficient = mean*(1-mean*mean)/32
        coefficient = result.correction_coefficients[1][0]
        assert _mp(coefficient.lo) <= exact_coefficient <= _mp(coefficient.hi)
    assert result.approximation_error_bound == first.approximation_error_bound
    assert result.predictor_enclosure[0].intersect(first.predictor_enclosure[0])


def test_inverse_age_scaling_and_cauchy_bounds_are_bound_to_each_coefficient():
    a = certify_gaussian_inverse_aging(_model(8), (Interval.point(0.25),), order=10)
    b = certify_gaussian_inverse_aging(_model(16), (Interval.point(0.25),), order=10)
    assert a is not None and b is not None
    for n, (left, right) in enumerate(zip(a.correction_coefficients, b.correction_coefficients, strict=True)):
        scaled = right[0]*Interval.point(2**n)
        assert scaled.intersect(left[0])
        bound = F(3, 16)**n
        assert F(left[0].lo) >= -bound and F(left[0].hi) <= bound


def test_target_box_contains_point_coefficients_and_inverse_enclosures():
    model = _model()
    result = certify_gaussian_inverse_aging(model, (Interval(-0.25, 0.25),), order=6)
    assert result is not None
    for target in (-0.25, -0.125, 0, 0.125, 0.25):
        point = certify_gaussian_inverse_aging(model, (Interval.point(target),), order=6)
        assert point is not None
        for box_coefficient, point_coefficient in zip(result.correction_coefficients, point.correction_coefficients, strict=True):
            assert box_coefficient[0].lo <= point_coefficient[0].lo <= point_coefficient[0].hi <= box_coefficient[0].hi
        assert result.enclosure[0].lo <= point.enclosure[0].lo <= point.enclosure[0].hi <= result.enclosure[0].hi


@pytest.mark.parametrize("order", (0, 1, 8, 128))
def test_affine_case_is_exact_without_transcendentals_and_preserves_large_cancellation(monkeypatch, order):
    model = GaussianHeatMap(((-10**400,), (-10**400,)), (0, 10**400), time=10**400)
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_inverse_aging(model, (Interval.point(1),), order=order)
    assert result is not None
    assert result.predictor_enclosure == result.enclosure == result.relative_mean == (Interval.point(0),)
    assert result.correction_coefficients == ((Interval.point(0),),)*(order+1)
    assert result.approximation_error_bound == 0
    assert result.certificate["meta"]["transcend_backend"] == "not_used"


def test_center_and_common_bias_gauges_preserve_coefficients():
    shifted = GaussianHeatMap(((10**100-1,), (10**100+1,)), (10**400, 10**400+F(1, 3)), time=16)
    a = certify_gaussian_inverse_aging(_model(), (Interval.point(0.25),), order=8)
    b = certify_gaussian_inverse_aging(shifted, (Interval.point(0.25),), order=8)
    assert a is not None and b is not None
    assert a.relative_mean == b.relative_mean
    assert a.correction_coefficients == b.correction_coefficients
    assert a.approximation_error_bound == b.approximation_error_bound


def test_no_inverse_solve_factorization_or_covariance_is_used(monkeypatch):
    import omnibias.core.verified.gaussian_aging as module
    original = module._posterior_moments

    def checked_moments(posterior, converted, *, covariance):
        assert covariance is False
        return original(posterior, converted, covariance=covariance)

    def fail(*args, **kwargs):
        pytest.fail("aging coefficients must not solve or factorize an inverse")

    monkeypatch.setattr(module, "_posterior_moments", checked_moments)
    monkeypatch.setattr(import_module("omnibias.core.verified.inverse_gaussian_heat"), "inverse_gaussian_heat", fail)
    monkeypatch.setattr(import_module("omnibias.core._rational_linear"), "factor_positive_definite", fail)
    assert certify_gaussian_inverse_aging(_model(), (Interval.point(0.25),), order=8) is not None


def test_missing_backend_raises_and_fresh_certificate_preserves_sticky_history(monkeypatch):
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_inverse_aging(_model(), (Interval.point(0),))
    assert result is not None and transcend.libm_fallback_used()
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert verify_certificate_digest(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError):
        certify_gaussian_inverse_aging(_model(), (Interval.point(0),))


@pytest.mark.parametrize("order,error", [(True, TypeError), (0.0, TypeError), (-1, ValueError)])
def test_invalid_orders_raise(order, error):
    with pytest.raises(error):
        certify_gaussian_inverse_aging(_model(), (Interval.point(0),), order=order)


@pytest.mark.parametrize("box,error", [(None, TypeError), ((0.0,), TypeError), ((), ValueError),
                                       ((Interval(0, 1), Interval(0, 1)), ValueError),
                                       ((Interval(0, float("inf")),), ValueError)])
def test_invalid_boxes_raise(box, error):
    with pytest.raises(error):
        certify_gaussian_inverse_aging(_model(), box)


@pytest.mark.parametrize("age", (F(3, 2), F(1), F(1, 2)))
def test_conservative_convergence_range_is_enforced_at_every_order(age):
    for order in (0, 1, 8):
        assert certify_gaussian_inverse_aging(_model(age), (Interval.point(0),), order=order) is None


def test_nonfinite_arithmetic_is_refused():
    assert certify_gaussian_inverse_aging(_model(), (Interval.point(1e308),)) is None


def test_factory_frozen_record_and_bound_certificate():
    with pytest.raises(TypeError):
        GaussianInverseAging()
    with pytest.raises(TypeError):
        certify_gaussian_inverse_aging(None, (Interval.point(0),))
    result = certify_gaussian_inverse_aging(_model(), (Interval(-0.125, 0.25),), order=4)
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.order = 0
    assert result.certificate["payload"]["order"] == 4
    assert result.certificate["payload"]["time"] == [16, 1]
    result.certificate["payload"]["correction_coefficients"][1][0]["lo"] = "invalid"
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("limits,target,order", [
    (RidgeBudget(max_order=2), 0, 3),
    (RidgeBudget(max_matrix_entries=200), 0, 8),
    (RidgeBudget(max_elimination_work=1500), 0, 8),
    (RidgeBudget(max_coefficient_bits=64), 1e100, 8),
])
def test_resource_failure_is_inconclusive(limits, target, order):
    model = GaussianHeatMap(((0,), (0,)), (0, 1), budget=limits)
    assert certify_gaussian_inverse_aging(model, (Interval.point(target),), order=order) is None
