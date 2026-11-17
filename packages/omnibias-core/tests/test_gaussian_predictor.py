# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""One-pass predictor enclosures against independent high-precision inverses."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from itertools import product

import mpmath as mp
import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_predictor import (
    GaussianInversePredictor,
    certify_gaussian_inverse_predictor,
)
from omnibias.core.verified.interval import Interval


def _mp(value):
    exact = F(value)
    return mp.mpf(exact.numerator)/exact.denominator


def _scalar_model(time=16, **kwargs):
    return GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=time, **kwargs)


def _references(model, target, order):
    centers = [tuple(_mp(x) for x in row) for row in model.centers]
    nu, age = _mp(model.viscosity), _mp(model.time)
    wanted = tuple(_mp(x) for x in target)

    def moments(logits):
        weights = [mp.exp(x-max(logits)) for x in logits]
        probabilities = [w/sum(weights) for w in weights]
        mean = tuple(sum(p*a[i] for p, a in zip(probabilities, centers, strict=True)) for i in range(model.dimension))
        centered = [tuple(a-m for a, m in zip(row, mean, strict=True)) for row in centers]
        third = tuple(sum(p*row[i]*sum(x*x for x in row) for p, row in zip(probabilities, centered, strict=True))
                      for i in range(model.dimension))
        return mean, third

    logits = [_mp(b)+sum(a*v for a, v in zip(row, wanted, strict=True))/(2*nu)
              for b, row in zip(model.biases, centers, strict=True)]
    mean, third = moments(logits)
    prediction = tuple(age*v+m-(t/(4*nu*age) if order else 0)
                       for v, m, t in zip(wanted, mean, third, strict=True))

    def residual(*point):
        original_logits = [_mp(b)-sum((x-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
                           for b, row in zip(model.biases, centers, strict=True)]
        actual_mean, _ = moments(original_logits)
        return tuple((x-m)/age-v for x, m, v in zip(point, actual_mean, wanted, strict=True))

    root = tuple(mp.findroot(residual, prediction, tol=mp.mpf("1e-80")))
    return prediction, root


@pytest.mark.parametrize("order", (0, 1))
@pytest.mark.parametrize("time", (F(3, 5), F(2), F(16), F(1024)))
def test_scalar_predictor_and_inverse_containment_at_multiple_heat_ages(order, time):
    model = _scalar_model(time)
    with mp.workdps(90):
        for target in (F(-3), F(-1, 8), F(0), F(1, 4), F(7)):
            result = certify_gaussian_inverse_predictor(model, (Interval.point(float(target)),), order=order)
            assert result is not None
            nominal, actual = _references(model, (target,), order)
            assert _mp(result.predictor_enclosure[0].lo) <= nominal[0] <= _mp(result.predictor_enclosure[0].hi)
            assert _mp(result.enclosure[0].lo) <= actual[0] <= _mp(result.enclosure[0].hi)
            assert abs(actual[0]-nominal[0]) <= _mp(result.approximation_error_bound)
            q = model.contraction_bound
            assert result.approximation_error_bound == (q if order == 0 else 2*q*q)


@pytest.mark.parametrize("order", (0, 1))
def test_coupled_box_encloses_predictors_and_inverses_at_corners_and_interior(order):
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 2), (1, -1)), (0, F(1, 3), F(-1, 4), F(1, 7)), time=8)
    box = (Interval(-0.125, 0.25), Interval(-0.25, 0.125))
    result = certify_gaussian_inverse_predictor(model, box, order=order)
    assert result is not None
    with mp.workdps(90):
        for target in product((F(-1, 8), F(1, 32), F(1, 4)), (F(-1, 4), F(-1, 16), F(1, 8))):
            nominal, actual = _references(model, target, order)
            for p, x, pbox, xbox in zip(nominal, actual, result.predictor_enclosure, result.enclosure, strict=True):
                assert _mp(pbox.lo) <= p <= _mp(pbox.hi)
                assert _mp(xbox.lo) <= x <= _mp(xbox.hi)
            assert mp.norm(tuple(x-p for x, p in zip(actual, nominal, strict=True))) <= _mp(result.approximation_error_bound)


def test_first_order_bound_improves_only_below_half_contraction():
    for age, improves in ((F(4), True), (F(1), False), (F(3, 4), False)):
        zero = certify_gaussian_inverse_predictor(_scalar_model(age), (Interval.point(0),), order=0)
        first = certify_gaussian_inverse_predictor(_scalar_model(age), (Interval.point(0),), order=1)
        assert zero is not None and first is not None
        assert (first.approximation_error_bound < zero.approximation_error_bound) == improves


@pytest.mark.parametrize("order", (0, 1))
def test_affine_case_combines_exact_large_terms_before_binary_boundary(monkeypatch, order):
    model = GaussianHeatMap(((-10**400,), (-10**400,)), (0, 10**400), time=10**400)
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_inverse_predictor(model, (Interval.point(1),), order=order)
    assert result is not None
    assert result.predictor_enclosure == result.enclosure == (Interval.point(0),)
    assert result.relative_mean == result.third_trace == (Interval.point(0),)
    assert result.approximation_error_bound == 0
    assert result.certificate["meta"]["transcend_backend"] == "not_used"


def test_exact_common_bias_and_center_gauges_preserve_limiting_moments():
    reference = _scalar_model()
    shifted = GaussianHeatMap(((10**100-1,), (10**100+1,)), (10**400, 10**400+F(1, 3)), time=16)
    a = certify_gaussian_inverse_predictor(reference, (Interval.point(0.25),))
    b = certify_gaussian_inverse_predictor(shifted, (Interval.point(0.25),))
    assert a is not None and b is not None
    assert a.relative_mean == b.relative_mean and a.third_trace == b.third_trace
    assert a.approximation_error_bound == b.approximation_error_bound


def test_core_predictor_does_not_call_an_inverse_or_build_a_covariance(monkeypatch):
    import omnibias.core.verified.gaussian_predictor as module
    inverse_module = import_module("omnibias.core.verified.inverse_gaussian_heat")
    original = module._posterior_moments

    def checked_moments(posterior, converted, *, covariance):
        assert covariance is False
        return original(posterior, converted, covariance=covariance)

    monkeypatch.setattr(module, "_posterior_moments", checked_moments)
    monkeypatch.setattr(inverse_module, "inverse_gaussian_heat", lambda *a, **k: pytest.fail("one-pass path must not iterate"))
    assert certify_gaussian_inverse_predictor(_scalar_model(), (Interval.point(0.25),)) is not None


def test_missing_backend_raises_and_fresh_proof_preserves_sticky_history(monkeypatch):
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_inverse_predictor(_scalar_model(), (Interval.point(0),))
    assert result is not None and transcend.libm_fallback_used()
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert verify_certificate_digest(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError):
        certify_gaussian_inverse_predictor(_scalar_model(), (Interval.point(0),))


@pytest.mark.parametrize("order,error", [(True, TypeError), (0.0, TypeError), (-1, ValueError), (2, ValueError)])
def test_invalid_orders_raise(order, error):
    with pytest.raises(error):
        certify_gaussian_inverse_predictor(_scalar_model(), (Interval.point(0),), order=order)


@pytest.mark.parametrize("box,error", [(None, TypeError), ((0.0,), TypeError), ((), ValueError),
                                       ((Interval(0, 1), Interval(0, 1)), ValueError),
                                       ((Interval(0, float("inf")),), ValueError)])
def test_invalid_boxes_raise(box, error):
    with pytest.raises(error):
        certify_gaussian_inverse_predictor(_scalar_model(), box)


def test_noncontractive_and_nonfinite_problems_are_refused():
    assert certify_gaussian_inverse_predictor(_scalar_model(F(1, 2)), (Interval.point(0),)) is None
    assert certify_gaussian_inverse_predictor(_scalar_model(F(1, 4)), (Interval.point(0),)) is None
    model = _scalar_model(100000, viscosity=F(1, 100))
    assert certify_gaussian_inverse_predictor(model, (Interval.point(1e308),)) is None


def test_factory_record_and_certificate_binding():
    with pytest.raises(TypeError):
        GaussianInversePredictor()
    with pytest.raises(TypeError):
        certify_gaussian_inverse_predictor(None, (Interval.point(0),))
    result = certify_gaussian_inverse_predictor(_scalar_model(), (Interval(-0.125, 0.25),))
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.order = 0
    assert result.certificate["payload"]["order"] == 1
    assert result.certificate["payload"]["time"] == [16, 1]
    result.certificate["payload"]["order"] = 0
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("limits,target", [
    (RidgeBudget(max_matrix_entries=200), 0),
    (RidgeBudget(max_elimination_work=1500), 0),
    (RidgeBudget(max_coefficient_bits=64), 1e100),
])
def test_resource_failure_is_inconclusive(limits, target):
    model = GaussianHeatMap(((0,), (0,)), (0, 1), budget=limits)
    assert certify_gaussian_inverse_predictor(model, (Interval.point(target),)) is None
