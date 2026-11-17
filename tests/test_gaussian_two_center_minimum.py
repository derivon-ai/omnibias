# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global two-center decisions checked against the original Gaussian family."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_two_center import certify_gaussian_two_center_minimum


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _original_logits(mp, model, point):
    denominator = 4*_mp(mp, model.viscosity)*_mp(mp, model.time)
    return tuple(_mp(mp, bias-model.biases[0])
                 -sum((x-_mp(mp, center))**2 for x, center in zip(point, row, strict=True))/denominator
                 for bias, row in zip(model.biases, model.centers, strict=True))


def _original_velocity(mp, model, point):
    logits = _original_logits(mp, model, point)
    weights = tuple(mp.exp(logit-max(logits)) for logit in logits)
    total = sum(weights)
    mean = tuple(sum(weight*_mp(mp, center[axis]) for weight, center in zip(weights, model.centers, strict=True))/total
                 for axis in range(model.dimension))
    return tuple((x-a)/_mp(mp, model.time) for x, a in zip(point, mean, strict=True))


def _original_energy(mp, model, point, target):
    # The common heat prefactor and exact common bias gauge cancel when
    # comparing values for this SAME model and target.
    logits = _original_logits(mp, model, point)
    peak = max(logits)
    log_heat = peak+mp.log(sum(mp.exp(logit-peak) for logit in logits))
    return -2*_mp(mp, model.viscosity)*log_heat-sum(_mp(mp, v)*x for v, x in zip(target, point, strict=True))


def _contains(mp, box, point):
    return all(_mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
               for interval, value in zip(box, point, strict=True))


def _rotated(*, age=F(2), gauge=0, reverse=False, midpoint=(F(2, 7), F(-3, 11))):
    half = (F(3, 5), F(4, 5))
    contrast, transverse = F(5, 7), (F(-12, 25), F(9, 25))
    centers = tuple(tuple(c+sign*a for c, a in zip(midpoint, half, strict=True)) for sign in (-1, 1))
    biases = (gauge-F(2, 9), gauge-F(2, 9)+contrast)
    if reverse:
        centers, biases = centers[::-1], biases[::-1]
    return GaussianHeatMap(centers, biases, viscosity=F(1, 8), time=age), midpoint, half, contrast, transverse


def _three_original_stationary_points(mp, model, target, midpoint, half, transverse):
    """Use original posterior equations, without the production tanh reduction."""
    alpha = sum(a*a for a in half)
    q = alpha/(2*model.viscosity*model.time)
    offset = -(model.biases[1]-model.biases[0])/(2*q)

    def lift(coordinate):
        return tuple(_mp(mp, c+model.time*u)+_mp(mp, a)*coordinate
                     for c, u, a in zip(midpoint, transverse, half, strict=True))

    def residual(coordinate):
        velocity = _original_velocity(mp, model, lift(coordinate))
        return sum(_mp(mp, a)*(value-_mp(mp, v)) for a, value, v in zip(half, velocity, target, strict=True))

    coordinates = tuple(mp.findroot(residual, (_mp(mp, offset)+seed-mp.mpf("0.05"),
                                               _mp(mp, offset)+seed+mp.mpf("0.05")))
                        for seed in (-1, 0, 1))
    assert coordinates[0] < coordinates[1] < coordinates[2]
    return tuple(lift(coordinate) for coordinate in coordinates)


@pytest.mark.parametrize("field", [F(-1, 100), F(0), F(1, 100)])
@pytest.mark.parametrize("geometry", ["scalar", "rotated"])
def test_certified_global_minima_beat_other_original_gaussian_stationary_points(field, geometry):
    mp = pytest.importorskip("mpmath")
    if geometry == "scalar":
        model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4))
        midpoint, half, transverse, contrast = (F(0),), (F(1),), (F(0),), F(0)
    else:
        model, midpoint, half, contrast, transverse = _rotated()
    q = model.contraction_bound
    assert q == 2
    target = tuple(u+a/model.time*(field-contrast/(2*q)) for u, a in zip(transverse, half, strict=True))
    result = certify_gaussian_two_center_minimum(model, target)
    assert result is not None and result.model is model and result.field == field
    assert result.selection_value == field/model.time
    assert result.unique == (field != 0)
    assert result.branch_signs == ((-1, 1) if not field else ((1,) if field > 0 else (-1,)))
    if result.termination == "tolerance":
        assert result.positive_root_bounds[1]-result.positive_root_bounds[0] <= result.tolerance
    with mp.workdps(100):
        points = _three_original_stationary_points(mp, model, target, midpoint, half, transverse)
        energies = tuple(_original_energy(mp, model, point, target) for point in points)
        for point in points:
            assert max(abs(value-_mp(mp, v)) for value, v in zip(_original_velocity(mp, model, point), target, strict=True)) < mp.mpf("1e-90")
        assert not any(_contains(mp, box, points[1]) for box in result.minimizer_boxes)
        if field:
            winning = 2 if field > 0 else 0
            assert all(energies[winning] < energy for index, energy in enumerate(energies) if index != winning)
            assert _contains(mp, result.minimizer_boxes[0], points[winning])
            assert not _contains(mp, result.minimizer_boxes[0], points[2-winning])
        else:
            assert abs(energies[0]-energies[2]) < mp.mpf("1e-90")
            assert energies[0] < energies[1]
            assert _contains(mp, result.minimizer_boxes[0], points[0])
            assert _contains(mp, result.minimizer_boxes[1], points[2])


@pytest.mark.parametrize("margin", [F(-1, 1000), F(0), F(1, 1000)])
def test_exact_selection_hyperplane_is_independent_of_age_and_midpoint(margin):
    selections = []
    for age in (F(1), F(2), F(3)):
        for midpoint in ((F(2, 7), F(-3, 11)), (F(5), F(-9))):
            model, _, half, contrast, transverse = _rotated(age=age, midpoint=midpoint)
            target = tuple(u+a*(margin-model.viscosity*contrast) for u, a in zip(transverse, half, strict=True))
            result = certify_gaussian_two_center_minimum(model, target)
            assert result is not None and model.contraction_bound > 1
            assert result.selection_value == margin
            assert result.field == age*margin
            assert result.midpoint == midpoint
            assert result.orthogonal_projector == ((F(16, 25), F(-12, 25)), (F(-12, 25), F(9, 25)))
            selections.append(result.branch_signs)
    assert all(selection == selections[0] for selection in selections)
    assert selections[0] == ((-1, 1) if not margin else ((1,) if margin > 0 else (-1,)))


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_subnormal_bias_contrast_selects_an_exact_branch_without_turning_into_a_tie(sign):
    mp = pytest.importorskip("mpmath")
    contrast = sign*F(1, 10**400)
    assert float(contrast) == 0
    model = GaussianHeatMap(((-1,), (1,)), (0, contrast), viscosity=F(1, 4))
    result = certify_gaussian_two_center_minimum(model, (0,))
    assert result is not None and result.field == contrast/4
    assert result.branch_signs == ((-1, 1) if sign == 0 else (sign,))
    assert len(result.minimizer_boxes) == (2 if sign == 0 else 1)
    with mp.workdps(450):
        for branch, box in zip(result.branch_signs, result.minimizer_boxes, strict=True):
            root = mp.findroot(lambda x: _original_velocity(mp, model, (x,))[0], branch)
            assert _contains(mp, box, (root,))
            assert branch*root > mp.mpf("0.9")
            if sign:
                opposite = mp.findroot(lambda x: _original_velocity(mp, model, (x,))[0], -branch)
                assert _original_energy(mp, model, (root,), (0,)) < _original_energy(mp, model, (opposite,), (0,))


@pytest.mark.parametrize("field", [F(0), F(1, 100)])
def test_mode_reversal_and_huge_common_bias_preserve_physical_global_minima(field):
    model, _, half, contrast, transverse = _rotated()
    gauged, _, _, _, _ = _rotated(gauge=10**400)
    reversed_model, _, _, _, _ = _rotated(reverse=True)
    target = tuple(u+a/model.time*(field-contrast/4) for u, a in zip(transverse, half, strict=True))
    ordinary = certify_gaussian_two_center_minimum(model, target)
    shifted = certify_gaussian_two_center_minimum(gauged, target)
    reversed_result = certify_gaussian_two_center_minimum(reversed_model, target)
    assert ordinary is not None and shifted is not None and reversed_result is not None
    assert ordinary.field == shifted.field == -reversed_result.field
    assert ordinary.positive_root_bounds == shifted.positive_root_bounds == reversed_result.positive_root_bounds
    assert ordinary.minimizer_boxes == shifted.minimizer_boxes
    assert ordinary.minimizer_boxes == (reversed_result.minimizer_boxes[::-1] if not field else reversed_result.minimizer_boxes)


@pytest.mark.parametrize("max_steps", [0, 1])
def test_analytic_or_early_stopped_brackets_still_enclose_all_global_minima(max_steps):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4))
    result = certify_gaussian_two_center_minimum(model, (0,), max_steps=max_steps)
    assert result is not None and result.termination == "steps"
    assert result.iterations == max_steps and result.branch_signs == (-1, 1)
    assert result.positive_root_bounds[1]-result.positive_root_bounds[0] > result.tolerance
    with mp.workdps(100):
        for sign, box in zip(result.branch_signs, result.minimizer_boxes, strict=True):
            root = mp.findroot(lambda x: _original_velocity(mp, model, (x,))[0], sign)
            assert _contains(mp, box, (root,))


def test_selection_jumps_across_the_tie_even_though_the_source_velocity_is_smooth():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4))
    left = certify_gaussian_two_center_minimum(model, (-F(1, 10**20),))
    tie = certify_gaussian_two_center_minimum(model, (0,))
    right = certify_gaussian_two_center_minimum(model, (F(1, 10**20),))
    assert left is not None and tie is not None and right is not None
    assert left.unique and right.unique and not tie.unique
    assert left.minimizer_boxes[0][0].hi < -0.9
    assert right.minimizer_boxes[0][0].lo > 0.9
    with mp.workdps(100):
        # The smooth velocity has a negative derivative at zero. The global
        # decision jumps between its outer positive-curvature stationary roots.
        derivative = mp.diff(lambda x: _original_velocity(mp, model, (x,))[0], 0)
        assert abs(derivative+1) < mp.mpf("1e-90")


@pytest.mark.parametrize("target_increment", [F(-1, 100), F(0), F(1, 100)])
def test_affine_restriction_composes_to_original_nonconvex_constrained_global_minima(target_increment):
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((-2, -1), (2, 1)), (0, 0))
    restriction = restrict_gaussian_heat(source, ((0, 1),), (1,))
    assert restriction is not None
    assert source.contraction_bound == F(5, 2)
    assert restriction.model.contraction_bound == 2
    assert restriction.model.centers == ((F(-2), F(1)), (F(2), F(1)))
    assert restriction.model.biases == (F(0), F(1))
    target = (-F(1, 2)+target_increment, F(7, 3))
    projected_target = restriction.project_target(target)
    result = certify_gaussian_two_center_minimum(restriction.model, projected_target)
    assert result is not None and result.selection_value == 2*target_increment
    assert result.branch_signs == ((-1, 1) if not target_increment else ((1,) if target_increment > 0 else (-1,)))
    with mp.workdps(100):
        # Solve the original Gaussian tangential KKT equation on x_2=1.
        points = tuple((mp.findroot(lambda x: _original_velocity(mp, source, (x, mp.mpf(1)))[0]-_mp(mp, target[0]), seed),
                        mp.mpf(1)) for seed in (-mp.mpf("2.5"), -mp.mpf("0.5"), mp.mpf("1.5")))
        assert points[0][0] < points[1][0] < points[2][0]
        energies = tuple(_original_energy(mp, source, point, target) for point in points)
        if target_increment:
            winner = 2 if target_increment > 0 else 0
            assert all(energies[winner] < energy for index, energy in enumerate(energies) if index != winner)
            assert _contains(mp, result.minimizer_boxes[0], points[winner])
        else:
            assert abs(energies[0]-energies[2]) < mp.mpf("1e-90")
            assert energies[0] < energies[1]
            assert _contains(mp, result.minimizer_boxes[0], points[0])
            assert _contains(mp, result.minimizer_boxes[1], points[2])
        for box in result.minimizer_boxes:
            assert box[1].lo == box[1].hi == 1
