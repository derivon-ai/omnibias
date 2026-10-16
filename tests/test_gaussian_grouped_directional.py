# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Grouped-posterior critical derivatives against independent QPs and secants."""

from fractions import Fraction as F
from itertools import combinations

import pytest
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_grouped_directional import (
    certify_gaussian_grouped_directional,
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_grouped_equilibrium import (
    certify_gaussian_grouped_equilibrium,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from test_gaussian_polyhedral_directional import _mp, _nonlinear_minimum


def _fixture(kind):
    centers, age = (((-1, 0), (1, 0), (0, -1), (0, 1)), 1) if kind == "axis" else (
        ((-1, -1), (1, 1), (-1, 1), (1, -1)), 2,
    )
    source = GaussianHeatMap(centers, (F(1, 4), F(1, 4), F(5, 4), F(5, 4)), time=age)
    inequalities, upper = ((1, 2), (-1, 0), (0, -1)), (0, 2, 3)
    return source, (), (), inequalities, upper, (0, 0), (0, 0), (), (0,)


def _equilibrium(kind):
    data = _fixture(kind)
    source, equalities, rhs, inequalities, upper, target, point, _, _ = data
    base = restrict_gaussian_heat(source, equalities, rhs)
    face = compile_gaussian_polyhedral_face(base, inequalities, upper)
    assert face is not None
    equilibrium = certify_gaussian_grouped_equilibrium(face, target, point)
    assert equilibrium is not None
    return equilibrium, data


def _hessian(mp, source, point):
    """Original Gaussian posterior, without the grouped evaluator."""
    age, nu = _mp(mp, source.time), _mp(mp, source.viscosity)
    centers = tuple(tuple(_mp(mp, value) for value in row) for row in source.centers)
    scores = tuple(_mp(mp, bias)-sum((_mp(mp, x)-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
                   for bias, row in zip(source.biases, centers, strict=True))
    weights = tuple(mp.exp(score-max(scores)) for score in scores)
    posterior = tuple(weight/sum(weights) for weight in weights)
    mean = tuple(sum(p*row[i] for p, row in zip(posterior, centers, strict=True)) for i in range(source.dimension))
    matrix = mp.matrix([[((1 if i == j else 0)-sum(p*(row[i]-mean[i])*(row[j]-mean[j])
                         for p, row in zip(posterior, centers, strict=True))/(2*nu*age))/age
                         for j in range(source.dimension)] for i in range(source.dimension)])
    return matrix, posterior


def _critical_qp(mp, data, direction):
    source, equalities, _, inequalities, _, _, point, strong, weak = data
    hessian, posterior = _hessian(mp, source, point)
    dimension = source.dimension
    fixed = equalities+tuple(inequalities[i] for i in strong)
    candidates = []
    for size in range(len(weak)+1):
        for selected in combinations(weak, size):
            rows = fixed+tuple(inequalities[i] for i in selected)
            kkt = mp.matrix(dimension+len(rows))
            for i in range(dimension):
                for j in range(dimension):
                    kkt[i, j] = hessian[i, j]
                for j, row in enumerate(rows):
                    kkt[i, dimension+j] = kkt[dimension+j, i] = _mp(mp, row[i])
            solved = mp.lu_solve(kkt, mp.matrix([_mp(mp, value) for value in direction]+[mp.mpf(0)]*len(rows)))
            value, multipliers = tuple(solved[:dimension]), tuple(solved[dimension:])
            if any(sum(_mp(mp, a)*x for a, x in zip(inequalities[i], value, strict=True)) > mp.mpf("1e-85") for i in weak):
                continue
            if any(mu < -mp.mpf("1e-85") for mu in multipliers[len(fixed):]):
                continue
            quadratic = sum(value[i]*sum(hessian[i, j]*value[j] for j in range(dimension)) for i in range(dimension))/2
            quadratic -= sum(_mp(mp, d)*x for d, x in zip(direction, value, strict=True))
            candidates.append((quadratic, selected, value, multipliers))
    assert candidates
    candidates.sort(key=lambda row: row[0])
    return candidates[0], hessian, posterior


def _assert_enclosed(mp, intervals, expected):
    for interval, value in zip(intervals, expected, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.mark.parametrize("kind", ["axis", "rotated"])
@pytest.mark.parametrize("direction", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def test_nonuniform_critical_qp_enclosure_and_original_nonlinear_secants(kind, direction):
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _equilibrium(kind)
    with mp.workdps(110):
        (_, selected, expected, multipliers), hessian, posterior = _critical_qp(mp, data, direction)
        assert abs(posterior[0]-posterior[2]) > mp.mpf("0.1")
        face = compile_gaussian_grouped_critical_face(equilibrium, selected)
        assert face is not None
        proposal = tuple(float(value) for value in expected)
        result = certify_gaussian_grouped_directional(face, direction, proposal)
        assert result is not None
        _assert_enclosed(mp, result.enclosure, expected)
        _assert_enclosed(mp, result.equality_multipliers, multipliers)
        actual_error = mp.sqrt(sum((_mp(mp, center)-exact)**2 for center, exact in zip(result.value, expected, strict=True)))
        assert actual_error <= _mp(mp, result.error_bound)
        assert result.value == result.projected_proposal
        assert result.error_bound < F(1, 10**10)
        residual = tuple(sum(hessian[i, j]*_mp(mp, result.value[j]) for j in range(2))-_mp(mp, direction[i]) for i in range(2))
        if selected:
            # Exact orthogonal projection onto x+2y=0, independent of core.
            residual = ((4*residual[0]-2*residual[1])/5, (-2*residual[0]+residual[1])/5)
            assert result.value[0]+2*result.value[1] == 0
        _assert_enclosed(mp, result.residual_enclosure, residual)
        assert mp.sqrt(sum(value**2 for value in residual)) <= _mp(mp, result.residual_norm_bound)
        errors = []
        for step in (F(1, 10), F(1, 100), F(1, 1000)):
            epsilon = _mp(mp, step)
            wanted = tuple(epsilon*_mp(mp, value) for value in direction)
            root = _nonlinear_minimum(mp, data, wanted)
            errors.append(mp.sqrt(sum((value/epsilon-exact)**2 for value, exact in zip(root, expected, strict=True))))
        assert errors[-1] < mp.mpf("1e-5")
        assert errors[1] < errors[0]/40 and errors[2] < errors[1]/40


def test_axis_group_fixture_has_the_independent_transcendental_closed_forms():
    mp = pytest.importorskip("mpmath")
    _, data = _equilibrium("axis")
    with mp.workdps(110):
        positive, hessian, _ = _critical_qp(mp, data, (1, 0))
        negative, _, _ = _critical_qp(mp, data, (-1, 0))
        e = mp.exp(1)
        expected_hessian = (((1+2*e)/(2*(1+e)), 0), (0, (2+e)/(2*(1+e))))
        for i in range(2):
            for j in range(2):
                assert abs(hessian[i, j]-expected_hessian[i][j]) < mp.mpf("1e-100")
        expected_positive = (8*(1+e)/(6+9*e), -4*(1+e)/(6+9*e))
        expected_negative = (-2*(1+e)/(1+2*e), 0)
        assert max(abs(a-b) for a, b in zip(positive[2], expected_positive, strict=True)) < mp.mpf("1e-100")
        assert max(abs(a-b) for a, b in zip(negative[2], expected_negative, strict=True)) < mp.mpf("1e-100")


@pytest.mark.parametrize("direction,selected", [((1, 0), (0,)), ((-1, 0), ())])
def test_coarse_proposals_still_certify_when_true_sign_margins_survive(direction, selected):
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _equilibrium("axis")
    face = compile_gaussian_grouped_critical_face(equilibrium, selected)
    with mp.workdps(110):
        (_, _, expected, _), _, _ = _critical_qp(mp, data, direction)
        accurate = certify_gaussian_grouped_directional(face, direction, tuple(float(value) for value in expected))
        coarse = certify_gaussian_grouped_directional(face, direction, tuple(float(value*mp.mpf("0.97")) for value in expected))
        assert accurate is not None and coarse is not None
        assert coarse.error_bound > 10**6*accurate.error_bound
        _assert_enclosed(mp, coarse.enclosure, expected)
        error = mp.sqrt(sum((_mp(mp, value)-true)**2 for value, true in zip(coarse.value, expected, strict=True)))
        assert mp.mpf("0.01") < error <= _mp(mp, coarse.error_bound)


def test_wrong_faces_are_refused_without_calling_a_directional_solution_exact():
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _equilibrium("axis")
    with mp.workdps(110):
        positive, _, _ = _critical_qp(mp, data, (1, 0))
        negative, _, _ = _critical_qp(mp, data, (-1, 0))
        active = compile_gaussian_grouped_critical_face(equilibrium, (0,))
        free = compile_gaussian_grouped_critical_face(equilibrium, ())
        assert certify_gaussian_grouped_directional(free, (1, 0), tuple(float(value) for value in positive[2])) is None
        assert certify_gaussian_grouped_directional(active, (-1, 0), tuple(float(value) for value in negative[2])) is None
