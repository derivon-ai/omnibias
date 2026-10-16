# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Critical two-center inverse charts against original Gaussian equations."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_critical import certify_gaussian_critical_chart
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.inverse_gaussian_heat import (
    certify_gaussian_inverse,
    inverse_gaussian_heat,
)


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _fixture(kind="rotated", *, gauge=0, reverse=False):
    if kind == "scalar":
        midpoint, half, age, viscosity, contrast, transverse = (F(0),), (F(1),), F(1), F(1, 2), F(0), (F(0),)
    else:
        midpoint, half = (F(2, 7), F(-3, 11)), (F(3, 5), F(4, 5))
        age = F(2) if kind == "rotated_age" else F(1)
        viscosity, contrast, transverse = 1/(2*age), F(5, 7), (F(-12, 25), F(9, 25))
    centers = tuple(tuple(c+sign*a for c, a in zip(midpoint, half, strict=True)) for sign in (-1, 1))
    biases = (gauge+F(-2, 9), gauge+F(-2, 9)+contrast)
    if reverse:
        centers, biases = centers[::-1], biases[::-1]
    return GaussianHeatMap(centers, biases, viscosity=viscosity, time=age), midpoint, half, contrast, transverse


def _original_velocity(mp, model, point):
    # Remove only the exact common log-bias gauge; the distances and each
    # original mode posterior are otherwise evaluated independently.
    centers = tuple(tuple(_mp(mp, a) for a in row) for row in model.centers)
    denominator = 4*_mp(mp, model.viscosity)*_mp(mp, model.time)
    logits = tuple(_mp(mp, b-model.biases[0])-sum((x-a)**2 for x, a in zip(point, center, strict=True))/denominator
                   for b, center in zip(model.biases, centers, strict=True))
    weights = tuple(mp.exp(value-max(logits)) for value in logits)
    posterior = tuple(weight/sum(weights) for weight in weights)
    mean = tuple(sum(p*center[i] for p, center in zip(posterior, centers, strict=True)) for i in range(model.dimension))
    return tuple((x-a)/_mp(mp, model.time) for x, a in zip(point, mean, strict=True)), posterior


def _chi(mp, parameter):
    if not parameter:
        return mp.mpf(0)
    return mp.findroot(lambda value: value-mp.tanh(value)-parameter**3/3,
                       (parameter, parameter*(1+parameter*parameter)))


def _assert_contains(mp, enclosure, expected):
    for interval, value in zip(enclosure, expected, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.mark.parametrize("kind", ["scalar", "rotated", "rotated_age"])
@pytest.mark.parametrize("parameter", [F(-1, 5), F(-1, 8), F(0), F(1, 8), F(1, 5)])
def test_critical_chart_contains_unique_original_gaussian_inverse_for_signed_and_zero_parameters(kind, parameter):
    mp = pytest.importorskip("mpmath")
    model, midpoint, half, contrast, transverse = _fixture(kind)
    chart = certify_gaussian_critical_chart(model, order=9)
    assert chart is not None and model.contraction_bound == 1
    assert chart.model is model
    assert chart.midpoint == midpoint and chart.half_difference == half
    assert chart.squared_half_distance == sum(a*a for a in half) == 2*model.viscosity*model.time
    assert chart.bias_contrast == contrast
    assert chart.parameter_radius == F(1, 4) and chart.shape_radius == F(1, 8)
    assert chart.complex_contraction_bound == F(37321, 120704)
    target = chart.target(parameter, transverse_target=transverse)
    expected_target = tuple(u+a/model.time*(parameter**3/3-contrast/2) for u, a in zip(transverse, half, strict=True))
    assert target == expected_target
    assert chart.target(parameter) == tuple(value-u for value, u in zip(target, transverse, strict=True))
    with mp.workdps(140):
        t = _mp(mp, parameter)
        chi = _chi(mp, t)
        expected = tuple(_mp(mp, c+model.time*u)+_mp(mp, a)*(chi-_mp(mp, contrast)/2)
                         for c, u, a in zip(midpoint, transverse, half, strict=True))
        _assert_contains(mp, chart.enclose(parameter, transverse_target=transverse), expected)
        _assert_contains(mp, chart.enclose(parameter), tuple(value-_mp(mp, model.time*u)
                                                           for value, u in zip(expected, transverse, strict=True)))
        velocity, posterior = _original_velocity(mp, model, expected)
        assert max(abs(value-_mp(mp, wanted)) for value, wanted in zip(velocity, target, strict=True)) < mp.mpf("1e-120")
        assert abs((posterior[1]-posterior[0])-mp.tanh(chi)) < mp.mpf("1e-120")
        if not parameter:
            assert max(abs(p-mp.mpf("0.5")) for p in posterior) < mp.mpf("1e-120")
        # The scalar map z-tanh(z) is strictly increasing: its derivative
        # tanh(z)^2 is nonnegative and vanishes only at z=0. Transverse
        # Gaussian velocity is affine, so this original root is unique.
        for displacement in (-mp.mpf("0.1"), mp.mpf("0.1")):
            trial = tuple(value+_mp(mp, a)*displacement for value, a in zip(expected, half, strict=True))
            trial_velocity, _ = _original_velocity(mp, model, trial)
            normal_residual = sum(_mp(mp, a)*(value-_mp(mp, goal)) for a, value, goal in zip(half, trial_velocity, target, strict=True))
            assert normal_residual*displacement > 0


@pytest.mark.parametrize("order", [0, 1, 2, 3, 5, 9])
def test_order_dependent_scalar_tail_bounds_cover_an_independent_transcendental_root(order):
    mp = pytest.importorskip("mpmath")
    model, _, _, _, _ = _fixture("scalar")
    chart = certify_gaussian_critical_chart(model, order=order)
    assert chart is not None
    for radius in (F(0), F(1, 16), F(1, 8), F(1, 5)):
        if order == 0:
            expected_tail = 9*radius/8
        else:
            degree = (order-1)//2
            ratio = radius/chart.parameter_radius
            expected_tail = radius/8*ratio**(2*degree+2)/(1-ratio*ratio)
        assert chart.tail_bound(radius) == expected_tail
        with mp.workdps(130):
            for parameter in (-radius, radius):
                t = _mp(mp, parameter)
                true_root = _chi(mp, t)
                polynomial = _mp(mp, chart.series.evaluate(parameter))
                assert abs(true_root-polynomial) <= _mp(mp, expected_tail)
                _assert_contains(mp, chart.enclose(parameter), (true_root,))


def test_strict_contraction_certifier_refuses_critical_model_while_new_chart_certifies_it():
    model, _, _, _, transverse = _fixture()
    chart = certify_gaussian_critical_chart(model)
    assert chart is not None
    target = chart.target(F(1, 8), transverse_target=transverse)
    assert certify_gaussian_inverse(model, target, (0.0, 0.0)) is None
    old = inverse_gaussian_heat(model, target)
    assert old.status == "outside_contraction_regime" and old.certificate is None
    assert all(interval.lo < interval.hi for interval in chart.enclose(F(1, 8), transverse_target=transverse))


def test_rotated_chart_preserves_transverse_linearity_and_exposes_cubic_root_singularity():
    mp = pytest.importorskip("mpmath")
    model, midpoint, half, contrast, transverse = _fixture()
    chart = certify_gaussian_critical_chart(model)
    assert chart is not None
    assert chart.orthogonal_projector == ((F(16, 25), F(-12, 25)), (F(-12, 25), F(9, 25)))
    with mp.workdps(130):
        ratios = []
        for parameter in (F(1, 8), F(1, 16), F(1, 32)):
            t = _mp(mp, parameter)
            chi = _chi(mp, t)
            normal_target_increment = t**3/(3*_mp(mp, model.time))
            inverse_difference_quotient = chi/normal_target_increment
            ratios.append(inverse_difference_quotient)
            assert abs(inverse_difference_quotient*t*t/(3*_mp(mp, model.time))-1) < mp.mpf("0.01")
        assert ratios[1] > 3*ratios[0] and ratios[2] > 3*ratios[1]
        critical_point = tuple(_mp(mp, c-a*contrast/2) for c, a in zip(midpoint, half, strict=True))
        shifted_point = tuple(value+_mp(mp, model.time*u) for value, u in zip(critical_point, transverse, strict=True))
        at_zero, _ = _original_velocity(mp, model, critical_point)
        shifted, _ = _original_velocity(mp, model, shifted_point)
        assert max(abs(right-left-_mp(mp, u)) for left, right, u in zip(at_zero, shifted, transverse, strict=True)) < mp.mpf("1e-120")
        _assert_contains(mp, chart.enclose(0, transverse_target=transverse), shifted_point)


def test_common_bias_gauge_and_mode_reversal_preserve_the_same_physical_chart():
    model, _, _, _, transverse = _fixture()
    shifted, _, _, _, _ = _fixture(gauge=10**400)
    reversed_model, _, _, _, _ = _fixture(reverse=True)
    chart = certify_gaussian_critical_chart(model)
    gauged = certify_gaussian_critical_chart(shifted)
    reversed_chart = certify_gaussian_critical_chart(reversed_model)
    assert chart is not None and gauged is not None and reversed_chart is not None
    assert reversed_chart.half_difference == tuple(-a for a in chart.half_difference)
    assert reversed_chart.bias_contrast == -chart.bias_contrast
    for parameter in (F(-1, 8), F(0), F(1, 8)):
        assert (chart.target(parameter, transverse_target=transverse) == gauged.target(parameter, transverse_target=transverse)
                == reversed_chart.target(-parameter, transverse_target=transverse))
        assert (chart.enclose(parameter, transverse_target=transverse) == gauged.enclose(parameter, transverse_target=transverse)
                == reversed_chart.enclose(-parameter, transverse_target=transverse))


@pytest.mark.parametrize("normal_target", [F(-1, 3003), F(0), F(1, 3003)])
def test_direct_exact_target_enclosure_handles_irrational_cube_root_and_critical_value(normal_target):
    mp = pytest.importorskip("mpmath")
    model, midpoint, half, contrast, transverse = _fixture("rotated_age")
    if normal_target < 0:
        transverse = (F(0), F(0))
    chart = certify_gaussian_critical_chart(model, order=9)
    assert chart is not None
    target = tuple(u+a/model.time*(normal_target-contrast/2) for u, a in zip(transverse, half, strict=True))
    assert abs(3*normal_target) < chart.parameter_radius**3
    with mp.workdps(150):
        cube = _mp(mp, 3*normal_target)
        parameter = mp.sign(cube)*mp.root(abs(cube), 3)
        chi = _chi(mp, parameter)
        expected = tuple(_mp(mp, c+model.time*u)+_mp(mp, a)*(chi-_mp(mp, contrast)/2)
                         for c, u, a in zip(midpoint, transverse, half, strict=True))
        for precision in (20, 80):
            enclosure = chart.enclose_target(target, precision=precision)
            _assert_contains(mp, enclosure, expected)
        velocity, _ = _original_velocity(mp, model, expected)
        assert max(abs(value-_mp(mp, wanted)) for value, wanted in zip(velocity, target, strict=True)) < mp.mpf("1e-130")
        if not normal_target:
            assert chart.enclose_target(target) == chart.enclose(0, transverse_target=transverse)


@pytest.mark.parametrize("viscosity", [F(1), F(1, 4)])
def test_noncritical_contraction_parameters_are_not_misclassified_as_the_critical_chart(viscosity):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=viscosity)
    assert model.contraction_bound != 1
    assert certify_gaussian_critical_chart(model) is None


def test_three_center_model_at_the_same_diameter_bound_need_not_have_a_cubic_singularity():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (0,), (1,)), (F(1, 2), 0, F(1, 2)), viscosity=F(1, 2))
    assert model.contraction_bound == 1 and certify_gaussian_critical_chart(model) is None
    with mp.workdps(120):
        # The original logits agree at zero, but the added central mode
        # lowers the posterior variance to 2/3. Consequently U'(0)=1/3,
        # so the two-mode cubic chart would describe the wrong inverse.
        value, posterior = _original_velocity(mp, model, (mp.mpf(0),))
        assert value == (mp.mpf(0),)
        assert max(abs(p-mp.mpf(1)/3) for p in posterior) < mp.mpf("1e-110")
        derivative = mp.diff(lambda x: _original_velocity(mp, model, (x,))[0][0], 0)
        assert abs(derivative-mp.mpf(1)/3) < mp.mpf("1e-110")
