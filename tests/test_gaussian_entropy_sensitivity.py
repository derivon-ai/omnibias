# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stable-face Gaussian derivatives against original potentials and roots."""

from fractions import Fraction as F

import pytest
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


def _potential(mp, model, point):
    logits = tuple(_mp(mp, bias-model.biases[0])
                   -sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
                   /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                   for bias, center in zip(model.biases, model.centers, strict=True))
    return -2*_mp(mp, model.viscosity)*mp.log(sum(mp.exp(value) for value in logits))


def _solve(model, rows=(), upper=(), target=None, reference=None):
    projector = compile_polyhedral_projector(model.dimension, rows, upper)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, target or (0,)*model.dimension)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=reference)
    assert unique is not None
    solved = solve_gaussian_entropy(unique, tolerance=1e-12, max_iterations=12)
    assert solved is not None and solved.status == "tolerance"
    return solved


def _encloses(mp, matrix, expected):
    for row, values in zip(matrix, expected.tolist(), strict=True):
        for interval, value in zip(row, values, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _normal_zero(sensitivity, cotangent):
    result = sensitivity.vjp(cotangent)
    assert all(axis.lo == axis.hi == 0 for axis in result)


def test_nonconvex_original_model_target_jacobian_and_independent_secants():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
    solved = _solve(model)
    face = certify_gaussian_entropy_face(solved.residual)
    assert face is not None and face.active_indices == ()
    sensitivity = certify_gaussian_entropy_sensitivity(face, terms=16)
    assert sensitivity is not None and model.contraction_bound == 4
    with mp.workdps(100):
        r = mp.findroot(lambda x: x-4*mp.exp(-12+2*x)/(1+2*mp.exp(-12+2*x)), mp.mpf("0.000025"))
        # Differentiate original Gaussian distances, without production moments.
        hxx = mp.diff(lambda x: _potential(mp, model, (x, r)), r, 2)
        hxy = mp.diff(lambda x: mp.diff(lambda y: _potential(mp, model, (x, y)), r), r)
        jacobian = mp.matrix([[hxx, hxy], [hxy, hxx]])**-1
        _encloses(mp, sensitivity.target_jacobian, jacobian)
        assert mp.diff(lambda x: _potential(mp, model, (x, mp.mpf(0))), 6, 2) < -mp.mpf("0.9")
        step = mp.mpf("1e-30")
        def stationary(x, y, v):
            p = mp.exp(-12+2*x)
            q = mp.exp(-12+2*y)
            return x-4*p/(1+p+q)-v, y-4*q/(1+p+q)
        plus = mp.findroot(lambda x, y: stationary(x, y, step), (r, r))
        minus = mp.findroot(lambda x, y: stationary(x, y, -step), (r, r))
        for axis in range(2):
            assert abs((plus[axis]-minus[axis])/(2*step)-jacobian[axis, 0]) < mp.mpf("1e-55")
        vector = sensitivity.vjp((3, -2))
        wanted = jacobian*mp.matrix([3, -2])
        assert all(_mp(mp, axis.lo) <= value <= _mp(mp, axis.hi)
                   for axis, value in zip(vector, wanted, strict=True))


def test_redundant_halfspace_face_and_exact_normal_null():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    solved = _solve(model, ((1, 0), (2, 0)), (0, 0), (1, 0))
    # The current isotropic image bound does not collapse to the actual face.
    assert solved.unique.domain.physical_bounds[0][0] < 0 < solved.unique.domain.physical_bounds[0][1]
    face = certify_gaussian_entropy_face(solved.residual)
    assert face is not None and face.active_indices == (0, 1)
    assert all(axis.lo > 0 for axis in face.projection_multipliers)
    sensitivity = certify_gaussian_entropy_sensitivity(face)
    assert sensitivity is not None
    _normal_zero(sensitivity, (2**80, 0))
    assert sensitivity.vjp((2**80, 3)) == sensitivity.vjp((0, 3))
    with mp.workdps(100):
        # Nonconvexity persists on the same feasible boundary, far from root.
        assert mp.diff(lambda y: _potential(mp, model, (mp.mpf(0), y)), 12, 2) < -mp.mpf("0.9")
        r = mp.findroot(lambda x: x-4*mp.exp(-24+2*x)/(1+mp.exp(-24)+mp.exp(-24+2*x)), mp.mpf("1.5e-10"))
        curvature = mp.diff(lambda y: _potential(mp, model, (mp.mpf(0), y)), r, 2)
        _encloses(mp, sensitivity.target_jacobian, mp.matrix([[0, 0], [0, 1/curvature]]))


def test_oblique_equality_opposing_and_redundant_normals():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    solved = _solve(model, ((1, 1), (-1, -1), (2, 2)), (0, 0, 0))
    face = certify_gaussian_entropy_face(solved.residual)
    assert face is not None and face.active_indices == (0, 1, 2)
    assert len(face.basis_indices) == 1
    assert all(axis.lo > 0 for axis in face.projection_multipliers)
    sensitivity = certify_gaussian_entropy_sensitivity(face)
    assert sensitivity is not None
    _normal_zero(sensitivity, (2**80, 2**80))
    with mp.workdps(100):
        curvature = mp.diff(lambda t: _potential(mp, model, (t, -t)), mp.mpf(0), 2)
        expected = mp.matrix([[1, -1], [-1, 1]])/curvature
        _encloses(mp, sensitivity.target_jacobian, expected)
        step = mp.mpf("1e-30")
        root = mp.findroot(lambda t: mp.diff(lambda z: _potential(mp, model, (z, -z)), t)-step, 0)
        assert abs(root/step-expected[0, 0]) < mp.mpf("1e-55")


def test_strict_vertex_has_exact_zero_derivative():
    model = GaussianHeatMap(((0, 0),), (0,))
    solved = _solve(model, ((-1, 0), (0, -1), (-2, 0)), (0, 0, 0), (-1, -2))
    face = certify_gaussian_entropy_face(solved.residual)
    assert face is not None and face.active_indices == (0, 1, 2)
    sensitivity = certify_gaussian_entropy_sensitivity(face)
    assert sensitivity is not None
    assert sensitivity.tail_norm_bound == 0
    assert all(axis.lo == axis.hi == 0 for row in sensitivity.target_jacobian for axis in row)
    _normal_zero(sensitivity, (F(1, 3), F(-2, 7)))


def test_unique_weak_switch_refuses_ordinary_target_derivative():
    model = GaussianHeatMap(((0,),), (0,))
    solved = _solve(model, ((-1,),), (0,))
    assert solved.residual.point == (0,) and solved.residual.error_bound == 0
    assert certify_gaussian_entropy_face(solved.residual) is None
    # The original decision is max(v,0), with unequal one-sided derivatives.
    projector = solved.unique.domain.projector
    assert projector.project((-1,)).point == (0,)
    assert projector.project((1,)).point == (1,)


def test_nonunit_time_viscosity_and_huge_common_bias_exact_jacobian():
    mp = pytest.importorskip("mpmath")
    centers = ((0, 0), (1, 0), (0, 1))
    wanted, nu, age = (F(1, 4), F(-1, 5)), F(3, 2), F(5, 3)
    biases = tuple(10**400+sum((x-a)**2 for x, a in zip(wanted, center, strict=True))/(4*nu*age) for center in centers)
    target = tuple((x-F(1, 3))/age for x in wanted)
    solved = _solve(GaussianHeatMap(centers, biases, viscosity=nu, time=age), target=target)
    face = certify_gaussian_entropy_face(solved.residual)
    assert face is not None
    sensitivity = certify_gaussian_entropy_sensitivity(face, terms=16)
    assert sensitivity is not None
    diagonal = 1/age-F(2, 9)/(2*nu*age**2)
    cross = F(1, 9)/(2*nu*age**2)
    determinant = diagonal**2-cross**2
    with mp.workdps(100):
        expected = mp.matrix([[_mp(mp, diagonal/determinant), _mp(mp, -cross/determinant)],
                              [_mp(mp, -cross/determinant), _mp(mp, diagonal/determinant)]])
        _encloses(mp, sensitivity.target_jacobian, expected)


def test_more_terms_improve_certified_analytic_tail_without_new_root_search():
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    face = certify_gaussian_entropy_face(_solve(model, reference=(0,)).residual)
    assert face is not None
    coarse = certify_gaussian_entropy_sensitivity(face, terms=1)
    fine = certify_gaussian_entropy_sensitivity(face, terms=16)
    assert coarse is not None and fine is not None
    assert 0 < fine.tail_norm_bound < coarse.tail_norm_bound
    assert fine.contraction_bound == coarse.contraction_bound
    assert coarse.target_jacobian[0][0].lo <= 2 <= coarse.target_jacobian[0][0].hi
    assert fine.target_jacobian[0][0].lo <= 2 <= fine.target_jacobian[0][0].hi
