# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Projected-centroid switching derivatives against original Gaussian KKT."""

from fractions import Fraction as F

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
from omnibias.core.verified.gaussian_projected_equilibrium import (
    certify_gaussian_projected_equilibrium,
)
from omnibias.core.verified.interval import Interval
from test_gaussian_grouped_directional import _assert_enclosed, _critical_qp, _hessian
from test_gaussian_polyhedral_directional import _mp, _nonlinear_minimum


def _fixture(cross):
    centers = ((-1, 0, -1), (1, 0, 1), (0, -1, 3), (0, 1, 5)) if cross else (
        (-1, 0, 0), (1, 0, 0), (0, -1, 4), (0, 1, 4),
    )
    biases = tuple(F(group)+sum(F(value)**2 for value in row)/4
                   for group, row in zip((0, 0, 1, 1), centers, strict=True))
    source = GaussianHeatMap(centers, biases)
    return source, ((0, 0, 1),), (0,), ((1, 2, 0), (0, 1, 0)), (0, 0), (1, 2, 0), (0, 0, 0), (0,), (1,)


def _equilibrium(cross):
    data = _fixture(cross)
    source, equalities, rhs, inequalities, upper, target, point, strong, _ = data
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert base is not None and base.model.contraction_bound == F(1, 2)
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, strong)
    assert face is not None
    assert certify_gaussian_grouped_equilibrium(face, target, point) is None
    equilibrium = certify_gaussian_projected_equilibrium(face, target, point)
    assert equilibrium is not None
    return equilibrium, data


@pytest.mark.parametrize("cross", [False, True], ids=("normal_group_offsets", "within_group_cross_covariance"))
@pytest.mark.parametrize("direction", [(1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 1, 0)])
def test_projected_centroid_qp_certificates_match_original_nonlinear_secants(cross, direction):
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _equilibrium(cross)
    assert equilibrium.group_centroids == ((0, 0, 0), (0, 0, 4))
    assert equilibrium.projected_centroid == (0, 0, 0)
    assert equilibrium.strong_indices == (0,) and equilibrium.weak_indices == (1,)
    with mp.workdps(110):
        (_, selected, expected, multipliers), hessian, _ = _critical_qp(mp, data, direction)
        if selected == (1,):
            # Three independent active rows force the response to zero.
            # Solve E.T*lambda=direction directly to avoid LU roundoff at
            # these analytically exact point-interval multipliers.
            exact_multipliers = (direction[2], direction[0], direction[1]-2*direction[0])
            assert max(abs(value) for value in expected) < mp.mpf("1e-100")
            assert max(abs(value-exact) for value, exact in zip(multipliers, exact_multipliers, strict=True)) < mp.mpf("1e-100")
            expected = (mp.mpf(0),)*3
            multipliers = tuple(mp.mpf(value) for value in exact_multipliers)
        face = compile_gaussian_grouped_critical_face(equilibrium, selected)
        assert face is not None
        result = certify_gaussian_grouped_directional(face, direction, tuple(float(value) for value in expected))
        assert result is not None
        _assert_enclosed(mp, result.enclosure, expected)
        _assert_enclosed(mp, result.equality_multipliers, multipliers)
        assert result.value[2] == 0 and result.enclosure[2] == Interval.point(0)
        assert result.coordinate_error_bounds[2] == 0
        actual_error = mp.sqrt(sum((_mp(mp, value)-true)**2 for value, true in zip(result.value, expected, strict=True)))
        assert actual_error <= _mp(mp, result.error_bound)
        projector = mp.matrix([[_mp(mp, value) for value in row] for row in face.projector])
        residual = projector*(hessian*mp.matrix([_mp(mp, value) for value in result.value])
                              -mp.matrix([_mp(mp, value) for value in direction]))
        _assert_enclosed(mp, result.residual_enclosure, residual)
        source, _, _, _, _, target, point, _, _ = data
        assert source.contraction_bound > 1 and hessian[2, 2] < 0
        errors = []
        for step in (F(1, 10), F(1, 100), F(1, 1000)):
            epsilon = _mp(mp, step)
            wanted = tuple(_mp(mp, value)+epsilon*_mp(mp, delta) for value, delta in zip(target, direction, strict=True))
            root = _nonlinear_minimum(mp, data, wanted)
            secant = tuple((value-_mp(mp, base))/epsilon for value, base in zip(root, point, strict=True))
            errors.append(mp.sqrt(sum((value-true)**2 for value, true in zip(secant, expected, strict=True))))
        assert errors[-1] < mp.mpf("1e-5")
        if errors[0] > mp.mpf("1e-60"):
            assert errors[1] < errors[0]/40 and errors[2] < errors[1]/40


@pytest.mark.parametrize("cross", [False, True])
def test_original_hessian_between_group_correction_and_normal_tangent_actions(cross):
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _equilibrium(cross)
    face = compile_gaussian_grouped_critical_face(equilibrium)
    assert face is not None
    with mp.workdps(110):
        hessian, posterior = _hessian(mp, data[0], data[6])
        e = mp.exp(1)
        mean_normal = 4*e/(1+e)
        _assert_enclosed(mp, equilibrium.equality_multipliers, (mean_normal,))
        _assert_enclosed(mp, equilibrium.inequality_multipliers, (mp.mpf(1), mp.mpf(0)))
        expected_normal_curvature = (mp.mpf("0.5") if cross else 1)-8*e/(1+e)**2
        assert abs(hessian[2, 2]-expected_normal_curvature) < mp.mpf("1e-100")
        assert face.hessian[2][2].hi < 0
        projector = mp.matrix([[_mp(mp, value) for value in row] for row in face.projector])
        left = mp.matrix([[_mp(mp, value) for value in row] for row in face.left_inverse])
        php, lhp = projector*hessian*projector, left*hessian*projector
        for i in range(3):
            _assert_enclosed(mp, face.hessian[i], tuple(hessian[i, j] for j in range(3)))
            _assert_enclosed(mp, face.projected_hessian[i], tuple(php[i, j] for j in range(3)))
        for i in range(2):
            _assert_enclosed(mp, face.multiplier_hessian[i], tuple(lhp[i, j] for j in range(3)))
        _assert_enclosed(mp, face.posterior, posterior)
        if cross:
            assert abs(lhp[0, 0]) > mp.mpf("0.01") and abs(lhp[0, 1]) > mp.mpf("0.01")
        else:
            assert face.multiplier_hessian[0] == (Interval.point(0),)*3
        hx, hy = (1+2*e)/(2*(1+e)), (2+e)/(2*(1+e))
        expected = (4/(4*hx+hy), -2/(4*hx+hy), mp.mpf(0))
        (qp, _, value, _), _, _ = _critical_qp(mp, data, (1, 0, 0))
        assert qp < 0
        assert max(abs(a-b) for a, b in zip(value, expected, strict=True)) < mp.mpf("1e-100")


@pytest.mark.parametrize("cross", [False, True])
def test_coarse_tangent_proposal_certifies_when_weak_slack_margin_survives(cross):
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _equilibrium(cross)
    face = compile_gaussian_grouped_critical_face(equilibrium)
    with mp.workdps(110):
        (_, _, expected, _), _, _ = _critical_qp(mp, data, (1, 0, 0))
        proposal = tuple(float(mp.mpf("0.97")*value) for value in expected)
        result = certify_gaussian_grouped_directional(face, (1, 0, 0), proposal)
        assert result is not None and result.error_bound > F(1, 100)
        assert result.weak_slacks[1].lo > 0
        _assert_enclosed(mp, result.enclosure, expected)
        actual_error = mp.sqrt(sum((_mp(mp, value)-true)**2 for value, true in zip(result.value, expected, strict=True)))
        assert actual_error <= _mp(mp, result.error_bound)
