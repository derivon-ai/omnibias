# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Mathematical enclosure and geometry laws for grouped critical responses."""

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
from omnibias.core.verified.interval import Interval

ZERO = Interval.point(0)


def _equilibrium(*, gauge=0, repeats=1, inequalities=((1, 2), (-1, 0)), upper=(0, 2), active=(), target=(0, 0)):
    centers = ((-1, 0), (1, 0))+((0, -1), (0, 1))*repeats
    biases = (gauge+F(1, 4),)*2+(gauge+F(5, 4),)*(2*repeats)
    source = GaussianHeatMap(centers, biases)
    base = restrict_gaussian_heat(source, (), ())
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, active)
    assert face is not None
    equilibrium = certify_gaussian_grouped_equilibrium(face, target, (0, 0))
    assert equilibrium is not None
    return equilibrium


def _compile(equilibrium, active=()):
    face = compile_gaussian_grouped_critical_face(equilibrium, active)
    assert face is not None
    return face


def _mp(mp, value):
    if isinstance(value, (int, float, F)):
        exact = F(value)
        return mp.mpf(exact.numerator)/exact.denominator
    return value


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_compiled_exact_tangent_geometry_and_nonuniform_hessian_enclosures():
    mp = pytest.importorskip("mpmath")
    equilibrium = _equilibrium()
    face = _compile(equilibrium, (0,))
    assert equilibrium.groups == ((0, 1), (2, 3)) and equilibrium.group_logits == (0, 1)
    assert face.constraints == ((1, 2),)
    assert face.projector == ((F(4, 5), F(-2, 5)), (F(-2, 5), F(1, 5)))
    assert face.left_inverse == ((F(1, 5), F(2, 5)),)
    with mp.workdps(100):
        e = mp.exp(1)
        probabilities = (1/(2*(1+e)),)*2+(e/(2*(1+e)),)*2
        h = (((1+2*e)/(2*(1+e)), mp.mpf(0)), (mp.mpf(0), (2+e)/(2*(1+e))))
        for interval, value in zip(face.posterior, probabilities, strict=True):
            _contains(mp, interval, value)
        for interval, value in zip(face.group_probabilities, (1/(1+e), e/(1+e)), strict=True):
            _contains(mp, interval, value)
        for i in range(2):
            for j in range(2):
                _contains(mp, face.hessian[i][j], h[i][j])
                php = sum(_mp(mp, face.projector[i][a])*h[a][b]*_mp(mp, face.projector[b][j])
                          for a in range(2) for b in range(2))
                _contains(mp, face.projected_hessian[i][j], php)
                if i == 0:
                    lhp = sum(_mp(mp, face.left_inverse[0][a])*h[a][b]*_mp(mp, face.projector[b][j])
                              for a in range(2) for b in range(2))
                    _contains(mp, face.multiplier_hessian[0][j], lhp)
        assert face.hessian[0][1] == face.hessian[1][0] == ZERO


def test_group_cardinalities_weight_the_compiled_posterior_and_directional_response():
    mp = pytest.importorskip("mpmath")
    face = _compile(_equilibrium(repeats=2), (0,))
    assert face.equilibrium.groups == ((0, 1), (2, 3, 4, 5))
    with mp.workdps(100):
        e = mp.exp(1)
        group_probability = (1/(1+2*e), 2*e/(1+2*e))
        posterior = (group_probability[0]/2,)*2+(group_probability[1]/4,)*4
        for interval, value in zip(face.group_probabilities, group_probability, strict=True):
            _contains(mp, interval, value)
        for interval, value in zip(face.posterior, posterior, strict=True):
            _contains(mp, interval, value)
        expected = (4*(1+2*e)/(3+9*e), -2*(1+2*e)/(3+9*e))
        result = certify_gaussian_grouped_directional(face, (1, 0), tuple(float(value) for value in expected))
        assert result is not None
        for interval, value in zip(result.enclosure, expected, strict=True):
            _contains(mp, interval, value)


@pytest.mark.parametrize("active", [(), (0,)])
def test_zero_direction_has_an_exact_answer_even_with_bad_finite_proposals(active):
    face = _compile(_equilibrium(), active)
    result = certify_gaussian_grouped_directional(face, (0, 0), (17.0, -9.0))
    assert result is not None and result.value == (0, 0)
    assert result.projected_proposal != result.value
    assert result.enclosure == result.residual_enclosure == (ZERO, ZERO)
    assert result.error_bound == result.residual_norm_bound == 0
    assert result.coordinate_error_bounds == (0, 0)
    assert result.weak_multipliers == result.weak_slacks == (ZERO, ZERO)


def test_exact_normal_direction_uses_zero_response_with_nonzero_weak_multiplier():
    face = _compile(_equilibrium(), (0,))
    result = certify_gaussian_grouped_directional(face, (1, 2), (37.0, -11.0))
    assert result is not None and result.value == (0, 0)
    assert result.error_bound == result.residual_norm_bound == 0
    assert result.enclosure == (ZERO, ZERO)
    assert result.equality_multipliers == (Interval.point(1),)
    assert result.weak_multipliers == (Interval.point(1), ZERO)
    assert certify_gaussian_grouped_directional(face, (-1, -2), (0.0, 0.0)) is None


def test_exact_projection_discards_normal_proposal_components_before_residual_evaluation():
    face = _compile(_equilibrium(), (0,))
    baseline = certify_gaussian_grouped_directional(face, (1, 0), (1.0, -0.5))
    shifted = certify_gaussian_grouped_directional(face, (1, 0), (17.0, 31.5))
    assert baseline is not None and shifted is not None
    assert baseline.value == shifted.value == (F(1), F(-1, 2))
    for name in ("enclosure", "error_bound", "residual_enclosure", "residual_norm_bound", "equality_multipliers",
                 "weak_multipliers", "weak_slacks", "coordinate_error_bounds"):
        assert getattr(baseline, name) == getattr(shifted, name)
    assert baseline.error_bound > 0
    assert baseline.weak_slacks[0] == ZERO


@pytest.mark.parametrize("scale", [F(1, 2), F(2), F(4)])
def test_positive_homogeneity_is_enclosed_for_scaled_directions_and_proposals(scale):
    mp = pytest.importorskip("mpmath")
    face = _compile(_equilibrium(), (0,))
    result = certify_gaussian_grouped_directional(face, (scale, 0), (float(scale), float(-scale/2)))
    assert result is not None and result.value == (scale, -scale/2)
    with mp.workdps(100):
        e = mp.exp(1)
        expected = (_mp(mp, scale)*8*(1+e)/(6+9*e), -_mp(mp, scale)*4*(1+e)/(6+9*e))
        for interval, value in zip(result.enclosure, expected, strict=True):
            _contains(mp, interval, value)
        error = mp.sqrt(sum((_mp(mp, value)-true)**2 for value, true in zip(result.value, expected, strict=True)))
        assert error <= _mp(mp, result.error_bound)


def test_strong_directional_multipliers_are_unrestricted_while_weak_multipliers_are_nonnegative():
    equilibrium = _equilibrium(inequalities=((1, 0), (0, 1)), upper=(0, 0), active=(0,), target=(1, 0))
    assert equilibrium.strong_indices == (0,) and equilibrium.weak_indices == (1,)
    face = _compile(equilibrium, (1,))
    result = certify_gaussian_grouped_directional(face, (-5, 2), (99.0, -37.0))
    assert result is not None and result.value == (0, 0)
    assert result.equality_multipliers == (Interval.point(-5), Interval.point(2))
    assert result.weak_multipliers == (ZERO, Interval.point(2))
    assert result.error_bound == 0
    assert certify_gaussian_grouped_directional(face, (-5, -2), (0.0, 0.0)) is None


def test_unselected_redundant_weak_rows_have_structurally_zero_slacks():
    equilibrium = _equilibrium(inequalities=((1, 2), (2, 4)), upper=(0, 0))
    first, second = _compile(equilibrium, (0,)), _compile(equilibrium, (1,))
    a = certify_gaussian_grouped_directional(first, (1, 2), (7.0, -3.0))
    b = certify_gaussian_grouped_directional(second, (1, 2), (7.0, -3.0))
    assert a is not None and b is not None and a.value == b.value == (0, 0)
    assert a.weak_slacks == b.weak_slacks == (ZERO, ZERO)
    assert a.weak_multipliers == (Interval.point(1), ZERO)
    assert b.weak_multipliers == (ZERO, Interval.point(0.5))
    with pytest.raises(ValueError, match="full row rank"):
        compile_gaussian_grouped_critical_face(equilibrium, (0, 1))


def test_opposite_unselected_weak_row_has_exact_zero_slack_despite_nonzero_error_box():
    equilibrium = _equilibrium(inequalities=((1, 2), (-2, -4)), upper=(0, 0))
    face = _compile(equilibrium, (0,))
    result = certify_gaussian_grouped_directional(face, (1, 0), (1.0, -0.5))
    assert result is not None and result.value == (1, F(-1, 2))
    assert result.error_bound > 0
    assert all(interval.lo < interval.hi for interval in result.enclosure)
    # The coordinate box extends off the selected line; exact projected
    # geometry must nevertheless preserve the opposite weak row's equality.
    assert result.weak_slacks == (ZERO, ZERO)
    assert result.weak_multipliers[0].lo > 0 and result.weak_multipliers[1] == ZERO


def test_negative_strong_multiplier_is_accepted_with_nonzero_directional_response():
    equilibrium = _equilibrium(active=(0,), target=(1, 2))
    assert equilibrium.strong_indices == (0,) and equilibrium.weak_indices == ()
    face = _compile(equilibrium)
    result = certify_gaussian_grouped_directional(face, (-1, 0), (-1.0, 0.5))
    assert result is not None and result.value == (-1, F(1, 2))
    assert result.error_bound > 0
    assert result.equality_multipliers[0].hi < 0
    assert result.weak_multipliers == result.weak_slacks == (ZERO, ZERO)


def test_opposing_weak_rows_show_that_refusal_is_inconclusive_even_for_a_zero_true_response():
    equilibrium = _equilibrium(inequalities=((1, 0), (-1, 0)), upper=(0, 0))
    first, second = _compile(equilibrium, (0,)), _compile(equilibrium, (1,))
    assert certify_gaussian_grouped_directional(first, (-1, 0), (0.0, 0.0)) is None
    result = certify_gaussian_grouped_directional(second, (-1, 0), (17.0, -11.0))
    assert result is not None and result.value == (0, 0)
    assert result.weak_multipliers == (ZERO, Interval.point(1))
    assert result.error_bound == 0


def test_huge_common_bias_changes_neither_compiled_enclosures_nor_query_bounds():
    normal, shifted = _compile(_equilibrium(), (0,)), _compile(_equilibrium(gauge=10**400), (0,))
    for field in ("projector", "left_inverse", "posterior", "group_probabilities", "hessian", "projected_hessian", "multiplier_hessian"):
        assert getattr(normal, field) == getattr(shifted, field)
    first = certify_gaussian_grouped_directional(normal, (1, 0), (1.0, -0.5))
    second = certify_gaussian_grouped_directional(shifted, (1, 0), (1.0, -0.5))
    assert first is not None and second is not None
    for field in ("value", "enclosure", "error_bound", "coordinate_error_bounds", "residual_enclosure",
                  "residual_norm_bound", "equality_multipliers", "weak_multipliers", "weak_slacks"):
        assert getattr(first, field) == getattr(second, field)


def test_indefinite_ambient_hessian_with_positive_base_tangent_curvature():
    mp = pytest.importorskip("mpmath")
    point = (0, 1, 0)
    centers = ((1, 2, 0), (-1, -2, 0), (0, 2, 1), (0, -2, -1))
    biases = tuple(F(group)+sum(F(x-a)**2 for x, a in zip(point, center, strict=True))/4
                   for group, center in zip((0, 0, 1, 1), centers, strict=True))
    source = GaussianHeatMap(centers, biases)
    base = restrict_gaussian_heat(source, ((0, 1, 0),), (1,))
    assert base is not None
    assert source.contraction_bound == F(5, 2) and base.model.contraction_bound == F(1, 2)
    polyhedral_face = compile_gaussian_polyhedral_face(base, ((1, 0, 2),), (0,))
    assert polyhedral_face is not None
    equilibrium = certify_gaussian_grouped_equilibrium(polyhedral_face, (0, 1, 0), point)
    assert equilibrium is not None and equilibrium.group_logits == (0, 1)
    face = _compile(equilibrium, (0,))
    assert face.constraints == ((0, 1, 0), (1, 0, 2))
    assert face.hessian[1][1] == Interval.point(-1)
    assert face.projected_hessian[1] == (ZERO,)*3
    with mp.workdps(110):
        # Form the original per-mode posterior directly, then differentiate
        # its mean. This does not use grouped moments or compiled curvature.
        original = tuple(tuple(_mp(mp, value) for value in row) for row in source.centers)
        scores = tuple(_mp(mp, bias)-sum((_mp(mp, x)-a)**2 for x, a in zip(point, row, strict=True))/4
                       for bias, row in zip(source.biases, original, strict=True))
        weights = tuple(mp.exp(score) for score in scores)
        posterior = tuple(weight/sum(weights) for weight in weights)
        mean = tuple(sum(p*row[i] for p, row in zip(posterior, original, strict=True)) for i in range(3))
        hessian = mp.matrix([[(1 if i == j else 0)-sum(p*(row[i]-mean[i])*(row[j]-mean[j])
                              for p, row in zip(posterior, original, strict=True))/2 for j in range(3)] for i in range(3)])
        # Every centered y-coordinate has square 4. Remove only the oracle's
        # posterior-normalization roundoff from this exact constant entry.
        assert abs(hessian[1, 1]+1) < mp.mpf("1e-100")
        hessian[1, 1] = -1
        projector = mp.matrix([[_mp(mp, value) for value in row] for row in face.projector])
        left = mp.matrix([[_mp(mp, value) for value in row] for row in face.left_inverse])
        projected, multiplier_action = projector*hessian*projector, left*hessian*projector
        for i in range(3):
            for j in range(3):
                _contains(mp, face.hessian[i][j], hessian[i, j])
                _contains(mp, face.projected_hessian[i][j], projected[i, j])
                if i < 2:
                    _contains(mp, face.multiplier_hessian[i][j], multiplier_action[i, j])
        kkt = mp.matrix(5)
        for i in range(3):
            for j in range(3):
                kkt[i, j] = hessian[i, j]
            for j, row in enumerate(face.constraints):
                kkt[i, 3+j] = kkt[3+j, i] = _mp(mp, row[i])
        solution = mp.lu_solve(kkt, mp.matrix((1, 0, 0, 0, 0)))
        e = mp.exp(1)
        expected = (8*(1+e)/(6+9*e), mp.mpf(0), -4*(1+e)/(6+9*e))
        assert max(abs(a-b) for a, b in zip(solution[:3], expected, strict=True)) < mp.mpf("1e-100")
        result = certify_gaussian_grouped_directional(face, (1, 0, 0), tuple(float(value) for value in expected))
        assert result is not None and result.value[1] == 0 and result.enclosure[1] == ZERO
        assert result.coordinate_error_bounds[1] == 0 and result.residual_enclosure[1] == ZERO
        for interval, value in zip(result.enclosure, expected, strict=True):
            _contains(mp, interval, value)
        for interval, value in zip(result.equality_multipliers, solution[3:], strict=True):
            _contains(mp, interval, value)
        assert result.equality_multipliers[0].hi < 0
        assert result.equality_multipliers[1].lo > 0
        residual = projector*(hessian*mp.matrix([_mp(mp, value) for value in result.value])-mp.matrix((1, 0, 0)))
        for interval, value in zip(result.residual_enclosure, residual, strict=True):
            _contains(mp, interval, value)
        error = mp.sqrt(sum((_mp(mp, value)-true)**2 for value, true in zip(result.value, expected, strict=True)))
        assert error <= _mp(mp, result.error_bound)
