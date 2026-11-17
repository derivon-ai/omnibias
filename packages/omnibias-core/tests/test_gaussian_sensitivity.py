# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Matrix-free interval actions and root-plus-gradient error certificates."""

import math
from copy import deepcopy
from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_sensitivity import certify_gaussian_target_vjp
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import inverse_gaussian_heat
from omnibias.core.verified.mean_map import MeanMap


def _contains(mp, interval, expected):
    assert mp.mpf(interval.lo) <= expected <= mp.mpf(interval.hi)


def test_mean_covariance_action_against_independent_directional_difference():
    mp = pytest.importorskip("mpmath")
    modes = ((0, 0), (1, 0), (-1, 2))
    model = MeanMap(modes, (0, 1, -1), (0, 0))
    vector = (0.5, -0.25)
    box = (Interval(0.1, 0.2), Interval(-0.3, -0.2))
    action = model.jacobian_action(box, tuple(Interval.point(value) for value in vector))
    with mp.workdps(80):
        for x, y in ((0.1, -0.3), (0.2, -0.2), (0.15, -0.25)):
            for axis in range(2):
                def mean(t, x=x, y=y, axis=axis):
                    weights = [mp.exp(a[0]*(mp.mpf(x)+t/2)+a[1]*(mp.mpf(y)-t/4)+b)
                               for a, b in zip(modes, (0, 1, -1), strict=True)]
                    return sum(p*a[axis] for p, a in zip(weights, modes, strict=True))/sum(weights)
                _contains(mp, action[axis], mp.diff(mean, 0))


def test_gaussian_action_does_not_construct_dense_jacobian(monkeypatch):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 0, 0, 0), (1, 0, 0, 0)), (0, 0), time=2)
    monkeypatch.setattr(MeanMap, "jacobian", lambda *_: pytest.fail("dense covariance used"))
    monkeypatch.setattr(GaussianHeatMap, "scaled_jacobian", lambda *_: pytest.fail("dense Jacobian used"))
    point = tuple(Interval.point(v) for v in (0.2, 0.0, 0.0, 0.0))
    vector = tuple(Interval.point(v) for v in (1.0, 2.0, 3.0, 4.0))
    action = model.scaled_jacobian_action(point, vector)
    with mp.workdps(80):
        _contains(mp, action[0], 1-mp.sech(mp.mpf(0.2)/2)**2/4)
    for value, expected in zip(action[1:], (2, 3, 4), strict=True):
        assert value.contains(expected)


@pytest.mark.parametrize("vector,error", [((), ValueError), ((1.0,), TypeError),
    ((Interval.point(math.inf),), ValueError), ((Interval(True, True),), TypeError)])
def test_interval_action_validates_its_direction(vector, error):
    model = MeanMap(((0,), (1,)), (0, 0), (0,))
    with pytest.raises(error):
        model.jacobian_action((Interval.point(0.0),), vector)


def test_interval_action_refuses_overflow_before_reusing_infinite_products():
    pytest.importorskip("mpmath")
    model = MeanMap(((0,), (10**150,)), (0, 0), (0,))
    with pytest.raises(ArithmeticError, match="nonfinite"):
        model.jacobian_action((Interval.point(0.0),), (Interval.point(1e100),))


@pytest.mark.parametrize("proposal", [(0.0,), (2.6,), (8/3,)])
def test_certifies_actual_target_vjp_including_bad_linear_solve(proposal):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=2)
    certificate = certify_gaussian_target_vjp(model, (0,), (0.0,), (1.0,), proposal)
    assert certificate is not None
    with mp.workdps(80):
        _contains(mp, certificate.enclosure[0], mp.mpf(8)/3)
        assert abs(mp.mpf(proposal[0])-mp.mpf(8)/3) <= certificate.error_bound
    assert certificate.jacobian_lipschitz_bound >= Fraction(1, 8)
    assert certificate_is_unconditional(certificate.certificate)
    assert schema_errors_v1(certificate.certificate) == []
    assert "theorem_prover_verified" not in certificate.certificate["honesty"]
    changed = deepcopy(certificate.certificate)
    changed["payload"]["proposal"] = [123.0]
    assert not verify_certificate_digest(changed)


def test_root_error_is_included_even_when_local_linear_solve_is_accurate():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=2)
    with mp.workdps(80):
        target = (Fraction(1, 3),)
        root = mp.findroot(lambda x: (x-mp.tanh(x/4))/2-mp.mpf(1)/3, 1)
        expected = 2/(1-mp.sech(root/4)**2/4)
        coarse_y = 0.3
        local_gradient = float(2/(1-mp.sech(mp.mpf(coarse_y)/2)**2/4))
        coarse = certify_gaussian_target_vjp(model, target, (coarse_y,), (1.0,), (local_gradient,))
        assert coarse is not None
        _contains(mp, coarse.enclosure[0], expected)
        assert abs(mp.mpf(local_gradient)-expected) > coarse.residual_bound*1000
        fine = inverse_gaussian_heat(model, target, absolute_tolerance=1e-11)
        fine_y = fine.scaled_point[0]
        fine_gradient = float(2/(1-mp.sech(mp.mpf(fine_y)/2)**2/4))
        certificate = certify_gaussian_target_vjp(model, target, (fine_y,), (1.0,), (fine_gradient,))
        assert certificate is not None and certificate.error_bound < coarse.error_bound/1000
        _contains(mp, certificate.enclosure[0], expected)


def test_coupled_vector_gradient_with_independent_root_and_jacobian():
    mp = pytest.importorskip("mpmath")
    centers = ((0, 0), (1, 0), (0, 2), (1, 1))
    model = GaussianHeatMap(centers, (0, 1, -1, 0), viscosity=Fraction(3, 2), time=2)
    target = (Fraction(1, 3), Fraction(-1, 4))
    root = inverse_gaussian_heat(model, target, absolute_tolerance=1e-11)
    with mp.workdps(80):
        def velocity(x, y):
            weights = [mp.exp(b-((x-a[0])**2+(y-a[1])**2)/12)
                       for a, b in zip(centers, (0, 1, -1, 0), strict=True)]
            mean = [sum(p*a[j] for p, a in zip(weights, centers, strict=True))/sum(weights) for j in range(2)]
            return mp.matrix(((x-mean[0])/2, (y-mean[1])/2))
        exact = mp.findroot(lambda x, y: velocity(x,y)-mp.matrix((mp.mpf(1)/3, -mp.mpf(1)/4)), (1,0))
        upstream = (0.75, -0.5)
        expected = mp.lu_solve(mp.matrix(mp.calculus.optimization.jacobian(mp, velocity, exact)), mp.matrix(upstream))
        proposal = tuple(float(value) for value in expected)
        certificate = certify_gaussian_target_vjp(model, target, root.scaled_point, upstream, proposal)
        assert certificate is not None and certificate.error_bound < 1e-8
        for interval, value in zip(certificate.enclosure, expected, strict=True):
            _contains(mp, interval, value)


def test_single_center_gives_scaled_identity_sensitivity():
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((2, -1),), (10**100,), time=3)
    certificate = certify_gaussian_target_vjp(model, (1, 2), (1.0, 0.0), (0.5, -0.25), (1.5, -0.75))
    assert certificate is not None and certificate.jacobian_lipschitz_bound == 0
    assert certificate.enclosure[0].contains(1.5) and certificate.enclosure[1].contains(-0.75)
    assert certificate.error_bound < 1e-12


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_provenance_survives_fresh_gradient_proof(monkeypatch, sticky, strict):
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=2)
    certificate = certify_gaussian_target_vjp(model, (0,), (0.0,), (1.0,), (8/3,))
    assert certificate is not None and certificate_is_unconditional(certificate.certificate)
    assert transcend.libm_fallback_used() is sticky and transcend.strict_backend() is strict


def test_invalid_regime_and_missing_rigorous_backend(monkeypatch):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=Fraction(1, 4))
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    assert certify_gaussian_target_vjp(model, (0,), (0.0,), (1.0,), (1.0,)) is None
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_target_vjp(GaussianHeatMap(((-1,), (1,)), (0, 0)), (0,), (0.0,), (1.0,), (1.0,))


@pytest.mark.parametrize("upstream,proposal,error", [((1, 2), (1,), ValueError), ((1,), (), ValueError),
    ((True,), (1,), TypeError), ((Fraction(1, 3),), (1,), TypeError), ((1,), (math.inf,), ValueError)])
def test_invalid_gradient_inputs_raise(upstream, proposal, error):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    with pytest.raises(error):
        certify_gaussian_target_vjp(model, (0,), (0.0,), upstream, proposal)


def test_resource_and_numerical_failures_are_inconclusive():
    pytest.importorskip("mpmath")
    limited = GaussianHeatMap(((-1,), (1,)), (0, 0), budget=RidgeBudget(max_elimination_work=1500))
    assert certify_gaussian_target_vjp(limited, (0,), (0.0,), (1.0,), (1.0,)) is None
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=2)
    assert certify_gaussian_target_vjp(model, (0,), (0.0,), (1.7e308,), (1.0,)) is None
