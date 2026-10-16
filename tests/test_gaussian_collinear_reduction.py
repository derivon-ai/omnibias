# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Collinear global decisions checked against original multimode Gaussians."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_collinear_minimum import certify_gaussian_collinear_minimum
from omnibias.core.verified.gaussian_collinear_reduction import compile_gaussian_collinear_reduction
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right, strict=True))


def _original_logits(mp, model, point, age=None):
    age = _mp(mp, model.time) if age is None else age
    return tuple(_mp(mp, bias-model.biases[0])
                 -sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
                 /(4*_mp(mp, model.viscosity)*age)
                 for bias, center in zip(model.biases, model.centers, strict=True))


def _velocity(mp, model, point, age=None):
    age = _mp(mp, model.time) if age is None else age
    logits = _original_logits(mp, model, point, age)
    weights = tuple(mp.exp(value-max(logits)) for value in logits)
    mean = tuple(sum(weight*_mp(mp, center[axis]) for weight, center in zip(weights, model.centers, strict=True))/sum(weights)
                 for axis in range(model.dimension))
    return tuple((x-a)/age for x, a in zip(point, mean, strict=True))


def _energy(mp, model, point, target):
    logits = _original_logits(mp, model, point)
    peak = max(logits)
    return (-2*_mp(mp, model.viscosity)*(peak+mp.log(sum(mp.exp(value-peak) for value in logits)))
            -sum(_mp(mp, v)*x for v, x in zip(target, point, strict=True)))


def _reduced_energy(mp, model, point, target):
    # Independent additive normalization from original metadata. The original
    # oracle omits the heat prefactor and common log-bias b0; completing the
    # transverse square therefore adds v.c+s*||Pv||²/2.
    origin = model.centers[0]
    direction = next(tuple(a-c for a, c in zip(row, origin, strict=True))
                     for row in model.centers if row != origin)
    alpha = _dot(direction, direction)
    transverse = tuple(v-a*_dot(direction, target)/alpha for v, a in zip(target, direction, strict=True))
    constant = _dot(target, origin)+model.time*_dot(transverse, transverse)/2
    return _energy(mp, model, point, target)+_mp(mp, constant)


def _contains(mp, box, point):
    return all(_mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
               for interval, value in zip(box, point, strict=True))


def _check(mp, result, model, rows, bounds, target, minimizers):
    assert result is not None
    assert all(isinstance(x, F) for x in result.sample_point)
    assert all(_dot(row, result.sample_point) <= bound for row, bound in zip(rows, bounds, strict=True))
    sample = tuple(_mp(mp, value) for value in result.sample_point)
    sample_value = _reduced_energy(mp, model, sample, target)
    assert _mp(mp, result.sample_value.lo) <= sample_value <= _mp(mp, result.sample_value.hi)
    for point in minimizers:
        value = _reduced_energy(mp, model, point, target)
        assert _mp(mp, result.minimum.lo) <= value <= _mp(mp, result.minimum.hi)
        assert any(_contains(mp, box, point) for box in result.candidate_boxes)
        assert value <= sample_value+mp.mpf("1e-80")
    if result.status == "value_tolerance":
        assert result.gap_upper <= result.value_tolerance


@pytest.mark.parametrize("max_boxes", [1, 255])
def test_three_local_wells_keep_both_outer_global_minima_and_do_not_promote_the_middle(max_boxes):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-2,), (0,), (2,)), (0, -1, 0), viscosity=F(1, 8))
    reduction = compile_gaussian_collinear_reduction(model, (), (), (0,))
    assert reduction is not None
    assert reduction.center_coordinates == (0, 1, 2)
    assert reduction.logit_offsets == (0, -9, -32)
    assert reduction.axial_scale == F(1, 16) and reduction.mean_bounds == (0, 2)
    assert reduction.pieces[0].search_interval == (0, 32)
    assert model.contraction_bound == (reduction.mean_bounds[1]-reduction.mean_bounds[0])**2/(4*reduction.axial_scale)
    result = certify_gaussian_collinear_minimum(reduction, max_boxes_per_piece=max_boxes)
    with mp.workdps(100):
        outer = mp.findroot(lambda x: _velocity(mp, model, (x,))[0], (mp.mpf("1.9"), mp.mpf(2)))
        inner = mp.findroot(lambda x: _velocity(mp, model, (x,))[0], (mp.mpf("0.7"), mp.mpf("0.9")))
        assert mp.mpf("1.99") < outer < 2 and mp.mpf("0.8") < inner < mp.mpf("0.9")
        # The original Gaussian derivative changes sign in each outer bracket.
        assert _velocity(mp, model, (mp.mpf("1.9"),))[0] < 0 < _velocity(mp, model, (mp.mpf(2),))[0]
        for root in (-outer, mp.mpf(0), outer):
            assert mp.diff(lambda x: _velocity(mp, model, (x,))[0], root) > 0
        for root in (-inner, inner):
            assert mp.diff(lambda x: _velocity(mp, model, (x,))[0], root) < 0
        assert _energy(mp, model, (outer,), (0,)) < _energy(mp, model, (mp.mpf(0),), (0,))
        assert abs(_energy(mp, model, (outer,), (0,))-_energy(mp, model, (-outer,), (0,))) < mp.mpf("1e-90")
        _check(mp, result, model, (), (), (0,), ((-outer,), (outer,)))
        if max_boxes == 255:
            assert result is not None and result.minimum.hi < float(_reduced_energy(mp, model, (mp.mpf(0),), (0,)))
            assert not any(_contains(mp, box, (mp.mpf(0),)) for box in result.candidate_boxes)


@pytest.mark.parametrize("permutation", [(0, 1, 2), (1, 2, 0), (2, 1, 0)])
def test_exact_origin_direction_and_bias_gauges_preserve_original_global_minima(permutation):
    mp = pytest.importorskip("mpmath")
    original_centers, original_biases = ((-2,), (0,), (2,)), (0, -1, 0)
    model = GaussianHeatMap(tuple(original_centers[index] for index in permutation),
                            tuple(10**400+original_biases[index] for index in permutation), viscosity=F(1, 8))
    reduction = compile_gaussian_collinear_reduction(model, (), (), (0,))
    assert reduction is not None
    for coordinate, center in zip(reduction.center_coordinates, model.centers, strict=True):
        assert tuple(c+coordinate*a for c, a in zip(reduction.origin, reduction.direction, strict=True)) == center
    result = certify_gaussian_collinear_minimum(reduction)
    with mp.workdps(100):
        root = mp.findroot(lambda x: _velocity(mp, model, (x,))[0], 2)
        _check(mp, result, model, (), (), (0,), ((-root,), (root,)))


def test_duplicate_modes_retain_their_weights_instead_of_being_deduplicated():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-2,), (0,), (2,), (2,)), (0, -1, 0, 0), viscosity=F(1, 8))
    reduction = compile_gaussian_collinear_reduction(model, (), (), (0,))
    assert reduction is not None and reduction.center_coordinates == (0, 1, 2, 2)
    assert len(reduction.logit_offsets) == model.mode_count == 4
    result = certify_gaussian_collinear_minimum(reduction)
    with mp.workdps(100):
        right = mp.findroot(lambda x: _velocity(mp, model, (x,))[0], 2)
        left = mp.findroot(lambda x: _velocity(mp, model, (x,))[0], -2)
        assert _energy(mp, model, (right,), (0,)) < _energy(mp, model, (left,), (0,))
        _check(mp, result, model, (), (), (0,), ((right,),))
        assert result is not None and not any(_contains(mp, box, (left,)) for box in result.candidate_boxes)


@pytest.mark.parametrize("constrained", [False, True])
def test_translated_rotated_three_mode_field_matches_original_objective_and_constraints(constrained):
    mp = pytest.importorskip("mpmath")
    center, axis, transverse = (F(2, 7), F(-3, 11)), (F(3, 5), F(4, 5)), (F(-4, 5), F(3, 5))
    centers = tuple(tuple(c+t*a for c, a in zip(center, axis, strict=True)) for t in (-2, 0, 2))
    model = GaussianHeatMap(centers, (0, -1, 0), viscosity=F(1, 8))
    target = tuple(b/2 for b in transverse)
    rows = tuple(tuple(sign*a-b for a, b in zip(axis, transverse, strict=True)) for sign in (-1, 1)) if constrained else ()
    bounds = tuple(_dot(row, center)-3 for row in rows)
    reduction = compile_gaussian_collinear_reduction(model, rows, bounds, target)
    assert reduction is not None
    result = certify_gaussian_collinear_minimum(reduction)
    with mp.workdps(100):
        if constrained:
            point = tuple(_mp(mp, c+3*b) for c, b in zip(center, transverse, strict=True))
            points = (point,)
            assert result is not None and result.sample_point == tuple(c+3*b for c, b in zip(center, transverse, strict=True))
            # On either active face r=3+|z|, the original directional
            # derivative away from z=0 is positive: |mean axis|<2 and
            # the transverse restoring force is >=3-1/2>2.
            assert all(piece.search_interval == (16, 16) for piece in reduction.pieces)
        else:
            def point_at(z):
                return tuple(_mp(mp, c+b/2)+_mp(mp, a)*z for c, a, b in zip(center, axis, transverse, strict=True))

            def stationarity(z):
                return sum(_mp(mp, a)*(u-_mp(mp, v)) for a, u, v in
                           zip(axis, _velocity(mp, model, point_at(z)), target, strict=True))

            root = mp.findroot(stationarity, 2)
            points = (point_at(-root), point_at(root))
        _check(mp, result, model, rows, bounds, target, points)


def test_coupled_three_dimensional_constraints_have_an_independent_original_kkt_root():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-2, 0, 0), (0, 0, 0), (2, 0, 0)), (0, -1, 0), viscosity=F(1, 8))
    rows = ((-1, 1, 1), (1, -1, -1), (2, 1, 2), (-2, -1, -2))
    bounds, target = (1, -1, 3, -3), (0, 0, 0)
    reduction = compile_gaussian_collinear_reduction(model, rows, bounds, target)
    assert reduction is not None and any(len(piece.active_indices) == 2 for piece in reduction.pieces)
    result = certify_gaussian_collinear_minimum(reduction, max_boxes_per_piece=63)
    with mp.workdps(100):
        def stationarity(z):
            velocity = _velocity(mp, model, (z, 4*z-1, 2-3*z))
            return velocity[0]+4*velocity[1]-3*velocity[2]

        root = mp.findroot(stationarity, mp.mpf("0.4"))
        # Along this line, U has derivative >=26-4²/(8nu*s)=10.
        assert mp.diff(stationarity, root) > 10
        _check(mp, result, model, rows, bounds, target, ((root, 4*root-1, 2-3*root),))


def test_noncollinear_and_all_coincident_geometry_are_refused_not_called_infeasible():
    for centers in (((0, 0), (1, 0), (0, 1)), ((1, 2), (1, 2), (1, 2))):
        model = GaussianHeatMap(centers, (0, 0, 0))
        assert compile_gaussian_collinear_reduction(model, (), (), (0, 0)) is None
    model = GaussianHeatMap(((-2,), (0,), (2,)), (0, -1, 0), viscosity=F(1, 8))
    impossible = compile_gaussian_collinear_reduction(model, ((0,),), (-1,), (0,))
    assert impossible is not None and not impossible.feasible and impossible.pieces == ()
    assert certify_gaussian_collinear_minimum(impossible) is None


def test_original_rotated_multimode_velocity_satisfies_the_coupled_burgers_equation():
    mp = pytest.importorskip("mpmath")
    center, axis = (F(2, 7), F(-3, 11)), (F(3, 5), F(4, 5))
    centers = tuple(tuple(c+t*a for c, a in zip(center, axis, strict=True)) for t in (-2, 0, 2))
    model = GaussianHeatMap(centers, (0, -1, 0), viscosity=F(1, 8))
    with mp.workdps(90):
        point, age = (mp.mpf("0.2"), mp.mpf("-0.3")), mp.mpf("1.2")
        velocity = _velocity(mp, model, point, age)
        for component in range(2):
            time_derivative = mp.diff(lambda s, component=component: _velocity(mp, model, point, s)[component], age)
            convection, laplacian = mp.mpf(0), mp.mpf(0)
            for axis_index in range(2):
                def field(value, *, component=component, axis_index=axis_index):
                    argument = (*point[:axis_index], value, *point[axis_index+1:])
                    return _velocity(mp, model, argument, age)[component]

                convection += velocity[axis_index]*mp.diff(field, point[axis_index])
                laplacian += mp.diff(field, point[axis_index], 2)
            assert abs(time_derivative+convection-_mp(mp, model.viscosity)*laplacian) < mp.mpf("1e-75")
