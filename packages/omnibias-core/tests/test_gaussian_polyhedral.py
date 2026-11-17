# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact clamp, corner and row-coordinate laws for Gaussian polyhedra."""

import math
from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_parameters import certify_gaussian_affine_parameter_vjp
from omnibias.core.verified.gaussian_affine_stationarity import certify_gaussian_affine_stationarity
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import (
    certify_gaussian_polyhedral_minimum,
    certify_gaussian_polyhedral_parameter_vjp,
    compile_gaussian_polyhedral_face,
)
from omnibias.core.verified.interval import Interval

ZERO = Interval.point(0)


def _contains(interval, value):
    assert F(interval.lo) <= F(value) <= F(interval.hi)


def _box_contains(box, values):
    for interval, value in zip(box, values, strict=True):
        _contains(interval, value)


def _face(source, inequalities, upper, active=(), *, equalities=(), rhs=()):
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, active)
    assert face is not None
    return face


@pytest.mark.parametrize("center,age,upper", [(0, 1, 0), (F(2, 3), F(3, 2), F(7, 5)), (-3, 2, -1)])
@pytest.mark.parametrize("target", [-2, 0, 1, 3])
def test_single_gaussian_scalar_clamp_and_exact_parameter_derivatives(center, age, upper, target):
    # For one Gaussian, the objective differs from (x-a)^2/(2s)-v*x
    # only by a parameter-dependent constant: X=min(a+s*v,h).
    center, age, upper, target = map(F, (center, age, upper, target))
    source = GaussianHeatMap(((center,),), (17,), viscosity=F(5, 4), time=age)
    free = center+age*target
    active = (0,) if free >= upper else ()
    face = _face(source, ((1,),), (upper,), active)
    minimum = certify_gaussian_polyhedral_minimum(face, (target,), (73.0,))
    assert minimum is not None
    expected = min(free, upper)
    _box_contains(minimum.root_enclosure, (expected,))
    _box_contains(minimum.slacks, (upper-expected,))
    _box_contains(minimum.inequality_multipliers, (max(F(0), target-(upper-center)/age),))
    assert minimum.strong_convexity_bound == 1/age
    assert minimum.active_set_stable == (free != upper)
    derivative = certify_gaussian_polyhedral_parameter_vjp(minimum, (1.5,), (-123.0,))
    if free == upper:
        assert derivative is None
        return
    assert derivative is not None
    assert derivative.equality_gradient == derivative.equality_rhs_gradient == ()
    assert derivative.bias_gradient == (ZERO,) and derivative.viscosity_gradient == ZERO
    if active:
        assert derivative.target_gradient == derivative.center_gradient[0] == (ZERO,)
        assert derivative.time_gradient == ZERO
        _box_contains(derivative.inequality_rhs_gradient, (F(3, 2),))
        _box_contains(derivative.inequality_gradient[0], (-F(3, 2)*upper,))
    else:
        _box_contains(derivative.target_gradient, (F(3, 2)*age,))
        _box_contains(derivative.center_gradient[0], (F(3, 2),))
        _contains(derivative.time_gradient, F(3, 2)*target)
        assert derivative.inequality_rhs_gradient == (ZERO,)
        assert derivative.inequality_gradient == ((ZERO,),)


@pytest.mark.parametrize("active", [(), (0,)])
def test_both_boundary_active_set_representations_refuse_derivatives(active):
    source = GaussianHeatMap(((0,),), (0,))
    minimum = certify_gaussian_polyhedral_minimum(_face(source, ((1,),), (0,), active), (0,), (999.0,))
    assert minimum is not None and not minimum.active_set_stable
    assert minimum.root_enclosure == minimum.slacks == minimum.inequality_multipliers == (ZERO,)
    assert certify_gaussian_polyhedral_parameter_vjp(minimum, (1.0,), (0.0,)) is None


@pytest.mark.parametrize("duplicate_bound,stable,slack", [(2, False, 0), (3, True, 1)])
def test_nonbinary_rational_face_preserves_exact_active_and_parallel_slacks(duplicate_bound, stable, slack):
    source = GaussianHeatMap(((0, 0),), (0,))
    inequalities = ((3, 0), (6, 0), (-3, 0), (0, 1), (0, -1))
    face = _face(source, inequalities, (1, duplicate_bound, 2, 1, 1), (0,))
    minimum = certify_gaussian_polyhedral_minimum(face, (1, 0), (19.0, -31.0))
    assert minimum is not None and minimum.active_set_stable is stable
    _box_contains(minimum.root_enclosure, (F(1, 3), 0))
    assert minimum.root_enclosure[0].lo < minimum.root_enclosure[0].hi
    assert minimum.slacks[0] == ZERO
    assert minimum.slacks[1] == Interval.point(slack)
    _box_contains(minimum.slacks, (0, slack, 3, 1, 1))
    _box_contains(minimum.inequality_multipliers, (F(2, 9), 0, 0, 0, 0))
    derivative = certify_gaussian_polyhedral_parameter_vjp(minimum, (2.0, 3.0), (100.0, -200.0))
    if not stable:
        assert derivative is None
    else:
        assert derivative is not None
        _box_contains(derivative.target_gradient, (0, 3))
        _box_contains(derivative.center_gradient[0], (0, 3))
        _box_contains(derivative.inequality_gradient[0], (F(-2, 9), F(-2, 3)))
        _box_contains(derivative.inequality_rhs_gradient, (F(2, 3), 0, 0, 0, 0))


def _corner(inequalities=((1, 2), (2, -1), (-1, 0), (0, -1)), upper=(3, 1, 10, 10), active=(0, 1)):
    # At X=(1,1), target-X=(8,1)=2*(1,2)+3*(2,-1).
    source = GaussianHeatMap(((0, 0),), (7,))
    face = _face(source, inequalities, upper, active)
    minimum = certify_gaussian_polyhedral_minimum(face, (9, 2), (999.0, -999.0))
    assert minimum is not None and minimum.active_set_stable
    derivative = certify_gaussian_polyhedral_parameter_vjp(minimum, (2.0, 3.0), (-71.0, 23.0))
    assert derivative is not None
    return minimum, derivative


def test_full_rank_corner_keeps_inequality_derivatives_and_annihilates_model_derivatives():
    minimum, derivative = _corner()
    _box_contains(minimum.root_enclosure, (1, 1))
    _box_contains(minimum.inequality_multipliers, (2, 3, 0, 0))
    _box_contains(minimum.slacks, (0, 0, 11, 11))
    assert derivative.target_gradient == (ZERO, ZERO)
    assert derivative.center_gradient == ((ZERO, ZERO),)
    assert derivative.bias_gradient == (ZERO,)
    assert derivative.viscosity_gradient == derivative.time_gradient == ZERO
    _box_contains(derivative.inequality_rhs_gradient, (F(8, 5), F(1, 5), 0, 0))
    _box_contains(derivative.inequality_gradient[0], (F(-8, 5), F(-8, 5)))
    _box_contains(derivative.inequality_gradient[1], (F(-1, 5), F(-1, 5)))
    assert derivative.inequality_gradient[2:] == ((ZERO, ZERO),)*2


def test_positive_row_scaling_and_permutation_remap_multipliers_and_parameter_gradients():
    rows, upper = ((1, 2), (2, -1), (-1, 0), (0, -1)), (3, 1, 10, 10)
    permutation, scales = (2, 1, 3, 0), (3, 2, F(1, 2), F(5, 3))
    changed_rows = tuple(tuple(scale*value for value in rows[index]) for index, scale in zip(permutation, scales, strict=True))
    changed_upper = tuple(scale*upper[index] for index, scale in zip(permutation, scales, strict=True))
    minimum, derivative = _corner(changed_rows, changed_upper, (3, 1))
    assert minimum.face.active_indices == (1, 3)
    _box_contains(minimum.root_enclosure, (1, 1))
    original_multipliers, original_rhs_gradient = (2, 3, 0, 0), (F(8, 5), F(1, 5), 0, 0)
    for j, (index, scale) in enumerate(zip(permutation, scales, strict=True)):
        _contains(minimum.inequality_multipliers[j], F(original_multipliers[index])/scale)
        _contains(derivative.inequality_rhs_gradient[j], F(original_rhs_gradient[index])/scale)
        _box_contains(derivative.inequality_gradient[j], (-F(original_rhs_gradient[index])/scale,)*2)
    for j in (0, 2):
        assert derivative.inequality_gradient[j] == (ZERO, ZERO)
        assert derivative.inequality_rhs_gradient[j] == ZERO


def test_common_bias_larger_than_binary64_preserves_partial_face_proofs_and_gradients():
    def query(gauge):
        source = GaussianHeatMap(((0, 0), (1, 2), (3, -1)),
                                 (gauge, gauge+F(1, 3), gauge-F(1, 4)), time=4)
        face = _face(source, ((1, 0), (-1, 0)), (0, 2), (0,))
        minimum = certify_gaussian_polyhedral_minimum(face, (1, 0), (0.0, 0.0))
        assert minimum is not None and minimum.active_set_stable
        derivative = certify_gaussian_polyhedral_parameter_vjp(minimum, (1.0, 3.0), (0.0, 12.0))
        assert derivative is not None
        return minimum, derivative

    baseline, baseline_gradient = query(0)
    shifted, shifted_gradient = query(10**400)
    for field in ("root_enclosure", "equality_multipliers", "inequality_multipliers", "slacks", "strong_convexity_bound"):
        assert getattr(baseline, field) == getattr(shifted, field)
    assert baseline.stationarity.posterior == shifted.stationarity.posterior
    for field in ("target_gradient", "center_gradient", "bias_gradient", "viscosity_gradient", "time_gradient",
                  "equality_gradient", "equality_rhs_gradient", "inequality_gradient", "inequality_rhs_gradient"):
        assert getattr(baseline_gradient, field) == getattr(shifted_gradient, field)


def test_empty_inequalities_reduce_to_exact_affine_optimizer_and_parameter_derivatives():
    source = GaussianHeatMap(((2, -3),), (17,), time=3)
    face = _face(source, (), (), equalities=((1, 2),), rhs=(1,))
    minimum = certify_gaussian_polyhedral_minimum(face, (1, -2), (100.0, -100.0))
    assert minimum is not None and minimum.active_set_stable
    assert minimum.inequality_multipliers == minimum.slacks == ()
    assert minimum.strong_convexity_bound == F(1, 3)
    _box_contains(minimum.root_enclosure, (F(39, 5), F(-17, 5)))
    derivative = certify_gaussian_polyhedral_parameter_vjp(minimum, (2.0, 3.0), (7.0, -11.0))
    affine = certify_gaussian_affine_parameter_vjp(face.geometry, (1, -2), (0.0, 0.0), (2.0, 3.0), (7.0, -11.0))
    assert derivative is not None and affine is not None
    assert derivative.inequality_gradient == derivative.inequality_rhs_gradient == ()
    assert derivative.target_gradient == affine.target_gradient
    assert derivative.equality_gradient == affine.constraint_gradient
    assert derivative.equality_rhs_gradient == affine.rhs_gradient
    _box_contains(derivative.target_gradient, (F(6, 5), F(-3, 5)))
    _box_contains(derivative.center_gradient[0], (F(2, 5), F(-1, 5)))
    _box_contains(derivative.equality_rhs_gradient, (F(8, 5),))
    _box_contains(derivative.equality_gradient[0], (F(-284, 25), F(122, 25)))
    _contains(derivative.time_gradient, F(4, 5))


def test_exact_two_dimensional_counterexample_requires_base_convexity_not_face_convexity():
    source = GaussianHeatMap(((-2, 0), (4, 0)), (1, 4))
    face = _face(source, ((1, 0),), (0,), (0,))
    assert face.base_restriction.model.contraction_bound == F(9, 2)
    assert face.geometry.restriction.model.contraction_bound == 0
    stationary = certify_gaussian_affine_stationarity(face.geometry, (0, 0), (0.0, 0.0))
    assert stationary is not None
    _box_contains(stationary.root_enclosure, (0, 0))
    _box_contains(stationary.primal_multiplier, (1,))
    # Exact scores, omitting the common positive heat-kernel prefactor:
    # the mixture sum is 2 at zero and e+e^-5>2 at the feasible (-2,0).
    def scores(point):
        return tuple(bias-sum((x-a)**2 for x, a in zip(point, center, strict=True))/4
                     for bias, center in zip(source.biases, source.centers, strict=True))

    assert scores((0, 0)) == (0, 0)
    assert scores((-2, 0)) == (1, -5)
    assert math.exp(1)+math.exp(-5) > 2
    assert certify_gaussian_polyhedral_minimum(face, (0, 0), (0.0, 0.0)) is None
