# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gaussian box moments against independent high-precision Gaussian weights."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import product

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified._gaussian_moments import gaussian_moments
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval


def _model(**kwargs):
    return GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=2, **kwargs)


def _mp(mp, value):
    exact = F(value)
    return mp.mpf(exact.numerator)/exact.denominator


def _oracle(mp, model, point):
    centers = [tuple(_mp(mp, x) for x in row) for row in model.centers]
    denominator = 4*_mp(mp, model.viscosity)*_mp(mp, model.time)
    logits = [_mp(mp, b)-sum((x-a)**2 for x, a in zip(point, row, strict=True))/denominator
              for b, row in zip(model.biases, centers, strict=True)]
    weights = [mp.exp(value-max(logits)) for value in logits]
    posterior = [weight/sum(weights) for weight in weights]
    mean = tuple(sum(p*row[axis] for p, row in zip(posterior, centers, strict=True)) for axis in range(model.dimension))
    centered = [tuple(a-mu for a, mu in zip(row, mean, strict=True)) for row in centers]
    covariance = tuple(tuple(sum(p*row[a]*row[b] for p, row in zip(posterior, centered, strict=True))
                             for b in range(model.dimension)) for a in range(model.dimension))
    third = tuple(sum(p*row[a]*sum(x*x for x in row) for p, row in zip(posterior, centered, strict=True))
                  for a in range(model.dimension))
    return posterior, tuple(mu-a for mu, a in zip(mean, centers[0], strict=True)), covariance, third


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.mark.parametrize("point", [-20.0, -0.5, 0.0, 0.5, 20.0])
def test_binary_family_matches_analytic_bernoulli_central_moments(point):
    mp = pytest.importorskip("mpmath")
    moments = gaussian_moments(_model(), (Interval.point(point),))
    with mp.workdps(90):
        p = 1/(1+mp.exp(-mp.mpf(point)/2-mp.mpf(1)/3))
        _contains(mp, moments.posterior[0], 1-p)
        _contains(mp, moments.posterior[1], p)
        _contains(mp, moments.relative_mean[0], 2*p)
        _contains(mp, moments.covariance[0][0], 4*p*(1-p))
        _contains(mp, moments.third_trace[0], 8*p*(1-p)*(1-2*p))
    assert moments.covariance[0][0].lo >= 0


def test_coupled_box_contains_moments_at_corners_and_interior():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 2), (2, -3), (0, 4)), (F(2, 3), F(-3, 7), F(5, 11)),
                            viscosity=F(3, 2), time=F(5, 4))
    box = (Interval(-0.5, 0.75), Interval(-0.25, 1.0))
    moments = gaussian_moments(model, box)
    with mp.workdps(90):
        for point in product((-0.5, 0.125, 0.75), (-0.25, 0.25, 1.0)):
            posterior, mean, covariance, third = _oracle(mp, model, tuple(map(mp.mpf, point)))
            for actual, bound in zip(posterior, moments.posterior, strict=True):
                _contains(mp, bound, actual)
            for actual, bound in zip(mean, moments.relative_mean, strict=True):
                _contains(mp, bound, actual)
            for a in range(model.dimension):
                _contains(mp, moments.third_trace[a], third[a])
                for b in range(model.dimension):
                    _contains(mp, moments.covariance[a][b], covariance[a][b])
                    assert moments.covariance[a][b] == moments.covariance[b][a]


def test_third_trace_matches_independent_spatial_derivative_of_covariance_trace():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 2)), (F(1, 3), F(-1, 5), F(2, 7)),
                            viscosity=F(3, 2), time=F(5, 4))
    point = (0.25, -0.5)
    moments = gaussian_moments(model, tuple(Interval.point(x) for x in point))
    with mp.workdps(90):
        for axis in range(2):
            def trace(delta, axis=axis):
                coordinates = [mp.mpf(x) for x in point]
                coordinates[axis] += delta
                covariance = _oracle(mp, model, tuple(coordinates))[2]
                return sum(covariance[j][j] for j in range(2))
            derivative = 2*_mp(mp, model.viscosity)*_mp(mp, model.time)*mp.diff(trace, 0)
            _contains(mp, moments.third_trace[axis], derivative)


def test_common_exact_bias_offset_does_not_lose_posterior_information():
    pytest.importorskip("mpmath")
    box = (Interval.point(0.25),)
    ordinary = gaussian_moments(_model(), box)
    huge = 10**400
    shifted = GaussianHeatMap(((-1,), (1,)), (huge, huge+F(1, 3)), time=2)
    assert gaussian_moments(shifted, box) == ordinary


def test_common_spatial_translation_preserves_relative_moments():
    pytest.importorskip("mpmath")
    shift = 2**40
    ordinary = gaussian_moments(_model(), (Interval.point(0.25),))
    model = GaussianHeatMap(((shift-1,), (shift+1,)), (0, F(1, 3)), time=2)
    translated = gaussian_moments(model, (Interval.point(shift+0.25),))
    assert translated == ordinary


@pytest.mark.parametrize("count", [1, 2])
def test_identical_enormous_centers_have_exact_zero_central_moments(count):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((10**400, -10**400),)*count, (10**400,)*count)
    moments = gaussian_moments(model, (Interval.point(0.0), Interval(-1.0, 1.0)))
    assert moments.relative_mean == (Interval.point(0.0),)*2
    assert moments.covariance == ((Interval.point(0.0),)*2,)*2
    assert moments.third_trace == (Interval.point(0.0),)*2


def test_softmax_lower_underflow_on_wide_box_still_returns_finite_bounds():
    mp = pytest.importorskip("mpmath")
    model = _model()
    moments = gaussian_moments(model, (Interval(-10000.0, 10000.0),))
    assert all(0 <= value.lo <= value.hi <= 1 for value in moments.posterior)
    assert moments.relative_mean[0].lo >= 0 and moments.relative_mean[0].hi <= 2
    with mp.workdps(90):
        for point in (-10000, -1, 0, 1, 10000):
            posterior, mean, covariance, third = _oracle(mp, model, (mp.mpf(point),))
            for actual, bound in zip(posterior, moments.posterior, strict=True):
                _contains(mp, bound, actual)
            _contains(mp, moments.relative_mean[0], mean[0])
            _contains(mp, moments.covariance[0][0], covariance[0][0])
            _contains(mp, moments.third_trace[0], third[0])


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_rigorous_backend_preserves_sticky_history_and_restores_strict_mode(monkeypatch, sticky, strict):
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    gaussian_moments(_model(), (Interval.point(0.0),))
    assert transcend.libm_fallback_used() is sticky
    assert transcend.strict_backend() is strict


def test_missing_backend_is_explicit_and_does_not_change_state(monkeypatch):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        gaussian_moments(_model(), (Interval.point(0.0),))
    assert transcend.libm_fallback_used() is True and transcend.strict_backend() is False


def test_numeric_overflow_refuses_and_restores_backend_mode(monkeypatch):
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    model = GaussianHeatMap(((0,), (10**200,)), (0, 0), viscosity=10**200, time=10**200)
    with pytest.raises(ArithmeticError, match="nonfinite"):
        gaussian_moments(model, (Interval.point(0.0),))
    assert transcend.strict_backend() is False


@pytest.mark.parametrize("box,error", [
    ((), ValueError), ((Interval.point(0.0),)*2, ValueError), ((0.0,), TypeError),
    ((Interval(True, True),), TypeError), ((Interval(F(1, 10), F(1, 10)),), TypeError),
    ((Interval(2**53+1, 2**53+1),), ValueError), ((Interval(0.0, float('inf')),), ValueError),
])
def test_box_shape_endpoint_and_type_validation(box, error):
    with pytest.raises(error):
        gaussian_moments(_model(), box)


def test_model_meter_types_immutable_output_and_resource_refusal():
    pytest.importorskip("mpmath")
    with pytest.raises(TypeError):
        gaussian_moments(object(), (Interval.point(0.0),))
    with pytest.raises(TypeError):
        gaussian_moments(_model(), iter((Interval.point(0.0),)))
    with pytest.raises(TypeError):
        gaussian_moments(_model(), (Interval.point(0.0),), arithmetic=object())
    with pytest.raises(ValueError, match="model budget"):
        gaussian_moments(_model(), (Interval.point(0.0),), arithmetic=_Arithmetic(RidgeBudget(max_order=2)))
    with pytest.raises(ValueError, match="entries"):
        gaussian_moments(_model(budget=RidgeBudget(max_matrix_entries=100)), (Interval.point(0.0),))
    with pytest.raises(ValueError, match="work"):
        gaussian_moments(_model(budget=RidgeBudget(max_elimination_work=1400)), (Interval.point(0.0),))
    budget = RidgeBudget(max_elimination_work=3000)
    model, arithmetic = _model(budget=budget), _Arithmetic(budget)
    moments = gaussian_moments(model, (Interval.point(0.0),), arithmetic=arithmetic)
    assert arithmetic.used > 0
    with pytest.raises(ValueError, match="work"):
        gaussian_moments(model, (Interval.point(0.0),), arithmetic=arithmetic)
    with pytest.raises(FrozenInstanceError):
        moments.third_trace = ()
