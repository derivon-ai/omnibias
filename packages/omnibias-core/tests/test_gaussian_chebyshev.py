# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent high-precision and refusal checks for inverse interpolation."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import mpmath as mp
import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_chebyshev import (
    GaussianInverseApproximation,
    approximate_gaussian_inverse,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_tube import certify_gaussian_inverse_tube


def _mp(value):
    exact = F(value)
    return mp.mpf(exact.numerator) / exact.denominator


def _inverse(model, target):
    """Independent physical Gaussian fixed point at high precision."""
    centers = [[_mp(value) for value in row] for row in model.centers]
    nu, age = _mp(model.viscosity), _mp(model.time)
    wanted = list(map(_mp, target))
    point = [age * value for value in wanted]
    for _ in range(400):
        logits = [_mp(bias) - sum((x - a)**2 for x, a in zip(point, center, strict=True)) / (4 * nu * age)
                  for bias, center in zip(model.biases, centers, strict=True)]
        weights = [mp.exp(value - max(logits)) for value in logits]
        following = [age * wanted[axis] + sum(weight * center[axis] for weight, center in zip(weights, centers, strict=True)) / sum(weights)
                     for axis in range(model.dimension)]
        if max(abs(x - y) for x, y in zip(point, following, strict=True)) < mp.mpf('1e-85'):
            return following
        point = following
    return point


def _tube(model=None):
    if model is None:
        model = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=2)
    result = certify_gaussian_inverse_tube(model)
    assert result is not None
    return result


def _approximation(**kwargs):
    result = approximate_gaussian_inverse(_tube(), (F(1, 4),), (1,), half_width=F(1, 4), **kwargs)
    assert result is not None
    return result


def test_unequal_bias_original_family_uniform_bound_and_outward_evaluation():
    approximation = _approximation(degree=24, ellipse_parameter=2)
    assert approximation.uniform_error_bounds[0] < F(1, 10**6)
    assert approximation.degree == 24
    with mp.workdps(90):
        for t in (F(-1, 4), F(1, 4), F(0), *(F(k, 97 * 4) for k in range(-96, 97, 11))):
            target = (F(1, 4) + t,)
            actual = _inverse(approximation.tube.model, target)[0]
            nominal = approximation.evaluate(t)[0]
            assert abs(actual - _mp(nominal)) <= _mp(approximation.uniform_error_bounds[0])
            box = approximation.enclose(t)[0]
            assert _mp(box.lo) <= actual <= _mp(box.hi)


def test_coupled_ideal_dct_coefficients_are_enclosed_with_both_endpoint_halfweights():
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (F(1, 5), F(-1, 4), F(2, 7)), time=2)
    base, direction, width, degree = (F(2, 5), F(-3, 7)), (1, F(1, 3)), F(1, 4), 6
    approximation = approximate_gaussian_inverse(_tube(model), base, direction, half_width=width, degree=degree, ellipse_parameter=2)
    assert approximation is not None
    with mp.workdps(90):
        samples = []
        for j in range(degree + 1):
            node = mp.cos(mp.pi * j / degree)
            target = [_mp(a) + _mp(width) * _mp(h) * node for a, h in zip(base, direction, strict=True)]
            # _inverse accepts rational inputs, so use a high-precision direct
            # findroot at the ideal irrational target for this separate oracle.
            def residual(x, y, wanted=target):
                logits = [_mp(b) - ((x-_mp(a[0]))**2 + (y-_mp(a[1]))**2)/8
                          for a, b in zip(model.centers, model.biases, strict=True)]
                weights = list(map(mp.exp, logits))
                return tuple((point - sum(w*_mp(a[axis]) for w, a in zip(weights, model.centers, strict=True))/sum(weights))/2 - wanted[axis]
                             for axis, point in enumerate((x, y)))
            root = mp.findroot(residual, (2*target[0], 2*target[1]), tol=mp.mpf('1e-80'))
            samples.append([root[i] - 2*target[i] for i in range(2)])
        for k, row in enumerate(approximation.coefficient_enclosures):
            outer = mp.mpf(1 if k in (0, degree) else 2) / degree
            for axis, box in enumerate(row):
                coefficient = outer * sum((mp.mpf('0.5') if j in (0, degree) else 1) * samples[j][axis] * mp.cos(mp.pi*k*j/degree)
                                          for j in range(degree + 1))
                assert _mp(box.lo) <= coefficient <= _mp(box.hi)
        for t in (F(-1, 4), F(-1, 7), F(0), F(2, 9), F(1, 4)):
            true = _inverse(model, tuple(a + t*h for a, h in zip(base, direction, strict=True)))
            for actual, nominal, error, box in zip(true, approximation.evaluate(t), approximation.uniform_error_bounds,
                                                  approximation.enclose(t), strict=True):
                assert abs(actual - _mp(nominal)) <= _mp(error)
                assert _mp(box.lo) <= actual <= _mp(box.hi)


def test_exact_nominal_clenshaw_matches_independent_chebyshev_basis():
    approximation = _approximation(degree=12, ellipse_parameter=2)
    for t in (F(-1, 4), F(1, 4), F(2, 37)):
        z = t / approximation.half_width
        basis = [F(1), z]
        for _ in range(2, approximation.degree + 1):
            basis.append(2*z*basis[-1] - basis[-2])
        expected = 2*(F(1, 4) + t) - 1 + sum(F(row[0])*value for row, value in zip(approximation.coefficients, basis, strict=True))
        assert approximation.evaluate(t) == (expected,)


def test_higher_degree_reduces_certified_error_and_coarse_sample_solve_is_honest():
    low = _approximation(degree=4, ellipse_parameter=2)
    high = _approximation(degree=24, ellipse_parameter=2)
    assert high.uniform_error_bounds[0] < low.uniform_error_bounds[0] / 1000
    coarse = _approximation(degree=8, ellipse_parameter=2, max_iterations=0)
    assert coarse.coefficient_error_bounds[0] > high.coefficient_error_bounds[0]
    with mp.workdps(90):
        for t in (F(-1, 4), F(1, 19), F(1, 4)):
            actual = _inverse(coarse.tube.model, (F(1, 4) + t,))[0]
            assert abs(actual - _mp(coarse.evaluate(t)[0])) <= _mp(coarse.uniform_error_bounds[0])


def test_affine_case_is_exact_without_transcendental_backend(monkeypatch):
    model = GaussianHeatMap(((2, -3), (2, -3)), (0, F(7, 3)), time=F(2, 3))
    tube = _tube(model)
    monkeypatch.setattr(transcend, '_MPMATH', None)
    monkeypatch.setattr(transcend, '_MPMATH_RESOLVED', True)
    monkeypatch.setattr(transcend, '_STRICT_BACKEND', False)
    approximation = approximate_gaussian_inverse(tube, (1, 2), (3, -1), half_width=100)
    assert approximation is not None and approximation.degree == 0
    assert approximation.uniform_error_bounds == (0, 0)
    assert approximation.certificate['meta']['transcend_backend'] == 'not_used'
    for t in (F(-100), F(1, 7), F(100)):
        actual = (F(2, 3)*(1+3*t)+2, F(2, 3)*(2-t)-3)
        assert approximation.evaluate(t) == actual
        for exact, box in zip(actual, approximation.enclose(t), strict=True):
            assert F(box.lo) <= exact <= F(box.hi)


def test_constant_path_keeps_sample_error_but_has_no_interpolation_tail():
    approximation = approximate_gaussian_inverse(_tube(), (F(1, 4),), (0,), half_width=7)
    assert approximation is not None and approximation.degree == 0
    assert approximation.ellipse_parameter is None and approximation.interpolation_error_bound == 0
    assert approximation.coefficient_error_bounds[0] > 0
    assert approximation.evaluate(-7) == approximation.evaluate(7)


def test_automatic_ellipse_is_strictly_inside_the_tube():
    approximation = _approximation(degree=8)
    rho = approximation.ellipse_parameter
    assert rho is not None
    assert approximation.half_width * (rho - 1/rho) / 2 < approximation.tube.target_radius


def test_unproved_and_boundary_ellipses_are_refused_before_sampling(monkeypatch):
    import omnibias.core.verified.gaussian_chebyshev as module
    monkeypatch.setattr(module, 'inverse_gaussian_heat', lambda *a, **k: pytest.fail('sampling must not run'))
    tube = _tube()
    assert tube.target_radius is not None
    assert approximate_gaussian_inverse(tube, (0,), (1,), half_width=1, ellipse_parameter=100) is None
    assert approximate_gaussian_inverse(tube, (0,), (1,), half_width=4*tube.target_radius/3, ellipse_parameter=2) is None


@pytest.mark.parametrize('kwargs,error', [
    ({'degree': 0}, ValueError), ({'degree': True}, TypeError), ({'max_iterations': -1}, ValueError),
    ({'max_iterations': 1.5}, TypeError), ({'half_width': 0}, ValueError), ({'half_width': 1.0}, TypeError),
    ({'ellipse_parameter': 1}, ValueError), ({'ellipse_parameter': 2.0}, TypeError),
    ({'absolute_tolerance': 0}, ValueError), ({'absolute_tolerance': float('nan')}, ValueError),
])
def test_invalid_configuration_raises(kwargs, error):
    with pytest.raises(error):
        approximate_gaussian_inverse(_tube(), (0,), (1,), **kwargs)


@pytest.mark.parametrize('base,direction,error', [
    ((0.0,), (1,), TypeError), ((False,), (1,), TypeError), ((0,), (1.0,), TypeError),
    ((0, 1), (1,), ValueError), ((0,), (), ValueError),
])
def test_invalid_line_metadata_raises(base, direction, error):
    with pytest.raises(error):
        approximate_gaussian_inverse(_tube(), base, direction)


def test_frozen_factory_record_and_evaluation_domains():
    approximation = _approximation(degree=4)
    with pytest.raises(TypeError):
        GaussianInverseApproximation()
    with pytest.raises(FrozenInstanceError):
        approximation.half_width = F(9)
    for method in (approximation.evaluate, approximation.enclose):
        for parameter in (F(-1, 3), F(1, 3)):
            with pytest.raises(ValueError):
                method(parameter)
        for parameter in (0.1, True):
            with pytest.raises(TypeError):
                method(parameter)


@pytest.mark.parametrize('limits,degree', [
    (RidgeBudget(max_order=3), 4), (RidgeBudget(max_matrix_entries=2500), 32),
    (RidgeBudget(max_elimination_work=3000), 8), (RidgeBudget(max_coefficient_bits=64), 32),
])
def test_resource_failures_are_inconclusive(limits, degree):
    model = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=2, budget=limits)
    tube = certify_gaussian_inverse_tube(model, spatial_radius=1)
    assert tube is not None
    assert approximate_gaussian_inverse(tube, (F(1, 4),), (1,), degree=degree, ellipse_parameter=F(31, 19)) is None


def test_fresh_rigorous_provenance_preserves_ambient_sticky_state(monkeypatch):
    monkeypatch.setattr(transcend, '_LIBM_FALLBACK_USED', True)
    previous = transcend.strict_backend()
    approximation = _approximation(degree=4, ellipse_parameter=2)
    assert transcend.libm_fallback_used() and transcend.strict_backend() == previous
    assert approximation.certificate['meta']['transcend_backend'] == 'mpmath'
    assert verify_certificate_digest(approximation.certificate)
    assert 'lean_verified' not in approximation.certificate['honesty']


def test_missing_rigorous_backend_raises_and_restores_strict_state(monkeypatch):
    tube = _tube()
    monkeypatch.setattr(transcend, '_MPMATH', None)
    monkeypatch.setattr(transcend, '_MPMATH_RESOLVED', True)
    monkeypatch.setattr(transcend, '_STRICT_BACKEND', False)
    with pytest.raises(RuntimeError, match='rigorous'):
        approximate_gaussian_inverse(tube, (0,), (1,), degree=4)
    assert not transcend.strict_backend()


def test_dependency_certificate_is_checked_and_snapshot_is_not_aliased():
    tube = _tube()
    approximation = approximate_gaussian_inverse(tube, (0,), (1,), degree=4)
    assert approximation is not None
    tube.certificate['payload']['biases'][0][0] = 99
    assert not verify_certificate_digest(tube.certificate)
    assert verify_certificate_digest(approximation.certificate)
    assert approximate_gaussian_inverse(tube, (0,), (1,), degree=4) is None
