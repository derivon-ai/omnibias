# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original Gaussian root secants and heat PDE checks for certified bias calculus."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_entropy_bias import certify_gaussian_entropy_bias
from omnibias.core.verified.gaussian_entropy_face import certify_gaussian_entropy_face
from omnibias.core.verified.gaussian_entropy_sensitivity import certify_gaussian_entropy_sensitivity
from omnibias.core.verified.gaussian_entropy_solver import solve_gaussian_entropy
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _case(model, rows=(), upper=(), target=None, reference=None):
    projector = compile_polyhedral_projector(model.dimension, rows, upper)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, target or (0,)*model.dimension)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=reference)
    assert unique is not None
    solution = solve_gaussian_entropy(unique, tolerance=1e-12, max_iterations=12)
    assert solution is not None and solution.status == "tolerance"
    face = certify_gaussian_entropy_face(solution.residual)
    assert face is not None
    sensitivity = certify_gaussian_entropy_sensitivity(face, terms=16)
    assert sensitivity is not None
    result = certify_gaussian_entropy_bias(sensitivity)
    assert result is not None
    return result


def _potential(mp, model, point, offsets):
    logits = tuple(_mp(mp, b-model.biases[0])+change
                   -sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
                   /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                   for b, change, center in zip(model.biases, offsets, model.centers, strict=True))
    return -2*_mp(mp, model.viscosity)*mp.log(sum(mp.exp(logit) for logit in logits))


def _stationarity(mp, model, point, target, offsets):
    logits = tuple(_mp(mp, b-model.biases[0])+change
                   -sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
                   /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                   for b, change, center in zip(model.biases, offsets, model.centers, strict=True))
    weights = tuple(mp.exp(logit) for logit in logits)
    mean = tuple(sum(weight*_mp(mp, center[j]) for weight, center in zip(weights, model.centers, strict=True))/sum(weights)
                 for j in range(model.dimension))
    return tuple((x-a)/_mp(mp, model.time)-v for x, a, v in zip(point, mean, target, strict=True))


def _contains(mp, box, values):
    assert all(_mp(mp, axis.lo) <= value <= _mp(mp, axis.hi)
               for axis, value in zip(box, values, strict=True))


def _zero(box):
    assert all(axis.lo == axis.hi == 0 for axis in box)


def test_nonconvex_bias_columns_joint_response_and_loss_pullbacks():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
    result = _case(model)
    assert model.contraction_bound == 4
    with mp.workdps(100):
        zero = (mp.mpf(0),)*3
        root = mp.findroot(lambda x, y: _stationarity(mp, model, (x, y), (0, 0), zero), (mp.mpf("0.000025"),)*2)
        x, y = root
        hxx = mp.diff(lambda a: _potential(mp, model, (a, y), zero), x, 2)
        hyy = mp.diff(lambda a: _potential(mp, model, (x, a), zero), y, 2)
        hxy = mp.diff(lambda a: mp.diff(lambda b: _potential(mp, model, (a, b), zero), y), x)
        target_jacobian = mp.matrix([[hxx, hxy], [hxy, hyy]])**-1
        columns = []
        step = mp.mpf("1e-25")
        for j in range(3):
            direction = tuple(mp.mpf(int(k == j)) for k in range(3))
            def perturbed_root(epsilon, d=direction):
                return mp.findroot(lambda a, b: _stationarity(mp, model, (a, b), (0, 0), tuple(epsilon*t for t in d)), (x, y))
            column = (perturbed_root(step)-perturbed_root(-step))/(2*step)
            columns.append(column)
            _contains(mp, tuple(row[j] for row in result.bias_jacobian), column)
        bias_jacobian = mp.matrix([[column[i] for column in columns] for i in range(2)])
        beta, xi, upstream = (F(5, 7), F(-3, 2), F(4, 3)), (F(1, 3), F(-2, 7)), (2, -3)
        expected = target_jacobian*mp.matrix([_mp(mp, v) for v in xi])+bias_jacobian*mp.matrix([_mp(mp, b) for b in beta])
        _contains(mp, result.jvp(xi, beta), expected)
        target_box, bias_box = result.vjp(upstream)
        _contains(mp, target_box, target_jacobian*mp.matrix(upstream))
        _contains(mp, bias_box, bias_jacobian.T*mp.matrix(upstream))
        assert abs(sum(bias_jacobian.T*mp.matrix(upstream))) < mp.mpf("1e-45")
        # Mixed paths are checked separately against original nonlinear roots.
        def mixed(epsilon):
            target = tuple(epsilon*_mp(mp, v) for v in xi)
            offsets = tuple(epsilon*_mp(mp, b) for b in beta)
            return mp.findroot(lambda a, b: _stationarity(mp, model, (a, b), target, offsets), (x, y))
        secant = (mixed(step)-mixed(-step))/(2*step)
        assert mp.norm(secant-expected) < mp.mpf("1e-45")


def test_redundant_constrained_nonconvex_original_bias_path():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    result = _case(model, ((1, 0), (2, 0)), (0, 0), (1, 0))
    beta, xi = (F(2, 3), -1, F(5, 4)), (7, F(1, 3))
    _zero(result.jvp((0, 0), (10**400,)*3))
    assert result.jvp(xi, tuple(F(b)+10**400 for b in beta)) == result.jvp(xi, beta)
    target_normal, bias_normal = result.vjp((2**80, 0))
    _zero((*target_normal, *bias_normal))
    with mp.workdps(100):
        step = mp.mpf("1e-25")
        def physical_root(epsilon):
            return mp.findroot(lambda y: _stationarity(mp, model, (mp.mpf(0), y),
                               (1+epsilon*7, epsilon/3), tuple(epsilon*_mp(mp, b) for b in beta))[1], mp.mpf("1.5e-10"))
        expected = (physical_root(step)-physical_root(-step))/(2*step)
        _contains(mp, result.jvp(xi, beta), (mp.mpf(0), expected))
        y = physical_root(0)
        zero = (mp.mpf(0),)*3
        assert _stationarity(mp, model, (mp.mpf(0), y), (1, 0), zero)[0] < -1


def test_oblique_equality_bias_secants_and_normal_invariance():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    result = _case(model, ((1, 1), (-1, -1), (2, 2)), (0, 0, 0))
    beta, xi = (0, 1, -2), (F(1, 5), F(-1, 7))
    _zero(result.jvp((0, 0), (10**400,)*3))
    normal = result.vjp((2**80, 2**80))
    _zero((*normal[0], *normal[1]))
    with mp.workdps(100):
        step = mp.mpf("1e-25")
        def root(epsilon):
            target = tuple(epsilon*_mp(mp, v) for v in xi)
            offsets = tuple(epsilon*b for b in beta)
            def tangent(z):
                ux, uy = _stationarity(mp, model, (z, -z), target, offsets)
                return ux-uy
            return mp.findroot(tangent, 0)
        derivative = (root(step)-root(-step))/(2*step)
        _contains(mp, result.jvp(xi, beta), (derivative, -derivative))


def test_all_projected_modes_equal_have_zero_bias_effect_on_tangent_decision():
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, F(1, 3)))
    result = _case(model, ((1, 0), (-1, 0)), (0, 0), (1, F(1, 3)), reference=(0, 0))
    assert result.projected_centers == ((0, 0), (0, 0))
    _zero(tuple(axis for row in result.bias_jacobian for axis in row))
    _zero(result.jvp((0, 0), (-10**400, 10**400)))
    target_box, bias_box = result.vjp((0, 1))
    _zero(bias_box)
    assert target_box[0].lo == target_box[0].hi == 0
    assert target_box[1].lo <= 1 <= target_box[1].hi


def test_nonunit_age_viscosity_and_bias_gauge_against_exact_uniform_formula():
    mp = pytest.importorskip("mpmath")
    centers = ((0, 0), (1, 0), (0, 1))
    wanted, nu, age = (F(1, 4), F(-1, 5)), F(3, 2), F(5, 3)
    biases = tuple(10**400+sum((x-a)**2 for x, a in zip(wanted, center, strict=True))/(4*nu*age) for center in centers)
    target = tuple((x-F(1, 3))/age for x in wanted)
    result = _case(GaussianHeatMap(centers, biases, viscosity=nu, time=age), target=target)
    diagonal = 1/age-F(2, 9)/(2*nu*age**2)
    cross = F(1, 9)/(2*nu*age**2)
    determinant = diagonal**2-cross**2
    with mp.workdps(100):
        j = mp.matrix([[_mp(mp, diagonal/determinant), _mp(mp, -cross/determinant)],
                       [_mp(mp, -cross/determinant), _mp(mp, diagonal/determinant)]])
        for mode, center in enumerate(centers):
            forcing = mp.matrix([_mp(mp, (F(a)-F(1, 3))/(3*age)) for a in center])
            _contains(mp, tuple(row[mode] for row in result.bias_forcing), forcing)
            _contains(mp, tuple(row[mode] for row in result.bias_jacobian), j*forcing)


def test_original_heat_bias_variation_satisfies_coupled_linearized_burgers():
    mp = pytest.importorskip("mpmath")
    # Differentiate original normalized Gaussian kernels, independently of
    # covariance, PDE, series, or sensitivity implementation helpers.
    centers = ((0, 0), (1, 2), (-1, 1))
    bias, beta = (F(0), F(1, 3), F(-2, 5)), (F(2), F(-1), F(3, 4))
    with mp.workdps(75):
        nu = _mp(mp, F(3, 4))
        point = (_mp(mp, F(1, 5)), _mp(mp, F(-1, 10)), _mp(mp, F(5, 3)))
        def tau(x, y, age, epsilon):
            return sum(mp.exp(_mp(mp, b)+epsilon*_mp(mp, be)-((x-a)**2+(y-c)**2)/(4*nu*age))
                       for b, be, (a, c) in zip(bias, beta, centers, strict=True))/(4*mp.pi*nu*age)
        def velocity(axis, x, y, age, epsilon):
            if axis == 0:
                return -2*nu*mp.diff(lambda z: mp.log(tau(z, y, age, epsilon)), x)
            return -2*nu*mp.diff(lambda z: mp.log(tau(x, z, age, epsilon)), y)
        def variation(axis, x, y, age):
            return mp.diff(lambda epsilon: velocity(axis, x, y, age, epsilon), mp.mpf(0))
        x, y, age = point
        u = tuple(velocity(i, x, y, age, 0) for i in range(2))
        du = tuple(variation(i, x, y, age) for i in range(2))
        for axis in range(2):
            temporal = mp.diff(lambda z, i=axis: variation(i, x, y, z), age)
            dx = mp.diff(lambda z, i=axis: variation(i, z, y, age), x)
            dy = mp.diff(lambda z, i=axis: variation(i, x, z, age), y)
            ux = mp.diff(lambda z, i=axis: velocity(i, z, y, age, 0), x)
            uy = mp.diff(lambda z, i=axis: velocity(i, x, z, age, 0), y)
            laplace = (mp.diff(lambda z, i=axis: variation(i, z, y, age), x, 2)
                       +mp.diff(lambda z, i=axis: variation(i, x, z, age), y, 2))
            residual = temporal+u[0]*dx+u[1]*dy+du[0]*ux+du[1]*uy-nu*laplace
            assert abs(residual) < mp.mpf("1e-65")
        weights = tuple(mp.exp(_mp(mp, b)-((x-a)**2+(y-c)**2)/(4*nu*age))
                        for b, (a, c) in zip(bias, centers, strict=True))
        p = tuple(w/sum(weights) for w in weights)
        mean_beta = sum(w*_mp(mp, be) for w, be in zip(p, beta, strict=True))
        for axis in range(2):
            expected = -sum(w*a[axis]*(_mp(mp, be)-mean_beta)
                            for w, a, be in zip(p, centers, beta, strict=True))/age
            assert abs(du[axis]-expected) < mp.mpf("1e-65")
