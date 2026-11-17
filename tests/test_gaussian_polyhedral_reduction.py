# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact transverse reduction and global bounds against original Gaussians."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    certify_gaussian_polyhedral_reduced_minimum,
)
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    compile_gaussian_polyhedral_reduction,
)


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right, strict=True))


def _original_logits(mp, model, point):
    return tuple(_mp(mp, bias-model.biases[0])
                 -sum((x-_mp(mp, center))**2 for x, center in zip(point, row, strict=True))
                 /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                 for bias, row in zip(model.biases, model.centers, strict=True))


def _original_energy(mp, model, point, target):
    logits = _original_logits(mp, model, point)
    peak = max(logits)
    return (-2*_mp(mp, model.viscosity)*(peak+mp.log(sum(mp.exp(value-peak) for value in logits)))
            -sum(_mp(mp, v)*x for v, x in zip(target, point, strict=True)))


def _original_velocity(mp, model, point):
    logits = _original_logits(mp, model, point)
    weights = tuple(mp.exp(value-max(logits)) for value in logits)
    mean = tuple(sum(weight*_mp(mp, center[axis]) for weight, center in zip(weights, model.centers, strict=True))/sum(weights)
                 for axis in range(model.dimension))
    return tuple((x-a)/_mp(mp, model.time) for x, a in zip(point, mean, strict=True))


def _reduced_original_energy(mp, model, point, target):
    # The reference has y=0 and r=sPv, hence reduced objective zero. It need
    # not be feasible: subtracting its ORIGINAL Gaussian energy fixes the
    # common additive constant without reading any compiled quadratic piece.
    half = tuple((plus-minus)/2 for minus, plus in zip(model.centers[0], model.centers[1], strict=True))
    midpoint = tuple(minus+a for minus, a in zip(model.centers[0], half, strict=True))
    alpha = _dot(half, half)
    inverse_q = 2*model.viscosity*model.time/alpha
    delta = model.biases[1]-model.biases[0]
    transverse = tuple(v-a*_dot(half, target)/alpha for v, a in zip(target, half, strict=True))
    reference = tuple(_mp(mp, c+model.time*u-inverse_q*a*delta/2)
                      for c, u, a in zip(midpoint, transverse, half, strict=True))
    return _original_energy(mp, model, point, target)-_original_energy(mp, model, reference, target)


def _contains(mp, box, point):
    return all(_mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
               for interval, value in zip(box, point, strict=True))


def _check_global_result(mp, result, model, inequalities, bounds, target, minimizers):
    assert result is not None
    assert all(isinstance(x, F) for x in result.sample_point)
    assert all(_dot(row, result.sample_point) <= bound for row, bound in zip(inequalities, bounds, strict=True))
    sample = tuple(_mp(mp, value) for value in result.sample_point)
    sample_value = _reduced_original_energy(mp, model, sample, target)
    assert _mp(mp, result.sample_value.lo) <= sample_value <= _mp(mp, result.sample_value.hi)
    for point in minimizers:
        value = _reduced_original_energy(mp, model, point, target)
        assert _mp(mp, result.minimum.lo) <= value <= _mp(mp, result.minimum.hi)
        assert any(_contains(mp, box, point) for box in result.candidate_boxes)
        assert value <= sample_value+mp.mpf("1e-80")
    if result.status == "value_tolerance":
        assert result.gap_upper <= result.value_tolerance


def test_breakpoint_global_minimum_survives_degenerate_active_bases():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=F(1, 4))
    rows, bounds, target = ((1, -1), (-1, -1), (1, -1), (0, 0)), (-1, -1, -1, 0), (0, 0)
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, target)
    assert reduction is not None and reduction.pieces
    for index, piece in enumerate(reduction.pieces):
        assert piece.search_interval == (F(0), F(0))
        for parameter in (F(-2), F(-1), F(0), F(1), F(2)):
            if (piece.lower is None or piece.lower <= parameter) and (piece.upper is None or parameter <= piece.upper):
                point = reduction.lift(index, parameter)
                assert point == (parameter/2, 1+abs(parameter)/2)
                assert reduction.transverse_value(index, parameter) == (1+abs(parameter)/2)**2/2
    result = certify_gaussian_polyhedral_reduced_minimum(reduction)
    with mp.workdps(100):
        _check_global_result(mp, result, model, rows, bounds, target, ((mp.mpf(0), mp.mpf(1)),))
    assert result is not None and result.sample_point == (F(0), F(1))
    assert result.minimum.lo <= 0.5 <= result.minimum.hi


@pytest.mark.parametrize("target_value", [F(1, 100), F(2)])
def test_negative_halfline_can_choose_metastable_branch_or_boundary(target_value):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4))
    rows, bounds, target = ((1,),), (0,), (target_value,)
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, target)
    assert reduction is not None and reduction.pieces
    result = certify_gaussian_polyhedral_reduced_minimum(reduction, value_tolerance=1e-7, max_boxes_per_piece=255)
    with mp.workdps(100):
        if target_value == 2:
            minimum = (mp.mpf(0),)
        else:
            minimum = (mp.findroot(lambda x: _original_velocity(mp, model, (x,))[0]-_mp(mp, target_value), -1),)
            unconstrained = (mp.findroot(lambda x: _original_velocity(mp, model, (x,))[0]-_mp(mp, target_value), 1),)
            assert _original_energy(mp, model, unconstrained, target) < _original_energy(mp, model, minimum, target)
            assert _original_energy(mp, model, minimum, target) < _original_energy(mp, model, (mp.mpf(0),), target)
        _check_global_result(mp, result, model, rows, bounds, target, (minimum,))


@pytest.mark.parametrize("viscosity", [F(1, 4), F(1, 2), F(1)])
@pytest.mark.parametrize("box_budget", [1, 255])
def test_unconstrained_cover_includes_all_minima_in_nonconvex_critical_and_convex_regimes(viscosity, box_budget):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=viscosity)
    reduction = compile_gaussian_polyhedral_reduction(model, (), (), (0,))
    assert reduction is not None and len(reduction.pieces) == 1
    result = certify_gaussian_polyhedral_reduced_minimum(reduction, max_boxes_per_piece=box_budget)
    with mp.workdps(100):
        if viscosity < F(1, 2):
            positive = mp.findroot(lambda x: _original_velocity(mp, model, (x,))[0], 1)
            points = ((-positive,), (positive,))
        else:
            points = ((mp.mpf(0),),)
        _check_global_result(mp, result, model, (), (), (0,), points)
    if box_budget == 1:
        assert result is not None and result.status == "box_budget"


def test_oblique_transverse_quadratic_and_lift_match_original_feasible_geometry():
    mp = pytest.importorskip("mpmath")
    center, half, transverse_axis = (F(2, 7), F(-3, 11)), (F(3, 5), F(4, 5)), (F(-4, 5), F(3, 5))
    centers = tuple(tuple(c+sign*a for c, a in zip(center, half, strict=True)) for sign in (-1, 1))
    model = GaussianHeatMap(centers, (0, 0), viscosity=F(1, 8), time=2)
    rows = tuple(tuple(sign*a-b for a, b in zip(half, transverse_axis, strict=True)) for sign in (-1, 1))
    bounds = tuple(_dot(row, center)-1 for row in rows)
    target = tuple(F(3, 10)*b for b in transverse_axis)
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, target)
    assert reduction is not None and reduction.pieces
    for index, piece in enumerate(reduction.pieces):
        for parameter in (F(-3, 2), F(-1, 3), F(0), F(1, 3), F(3, 2)):
            if (piece.lower is None or piece.lower <= parameter) and (piece.upper is None or parameter <= piece.upper):
                expected = tuple(c+a*parameter/2+b*(1+abs(parameter)/2)
                                 for c, a, b in zip(center, half, transverse_axis, strict=True))
                assert reduction.lift(index, parameter) == expected
                assert reduction.transverse_value(index, parameter) == (F(2, 5)+abs(parameter)/2)**2/4
    result = certify_gaussian_polyhedral_reduced_minimum(reduction)
    with mp.workdps(100):
        minimum = tuple(_mp(mp, c+b) for c, b in zip(center, transverse_axis, strict=True))
        _check_global_result(mp, result, model, rows, bounds, target, (minimum,))
    assert result is not None and result.sample_point == tuple(c+b for c, b in zip(center, transverse_axis, strict=True))


def test_singleton_polyhedron_has_exact_feasible_witness_without_interior_assumptions():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, F(2, 7)), viscosity=F(1, 4))
    rows, bounds = ((1, 0), (-1, 0), (0, 1), (0, -1)), (F(1, 7), F(-1, 7), 2, -2)
    target = (F(2, 5), F(-3, 7))
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, target)
    assert reduction is not None and reduction.pieces
    expected_parameter = F(3, 7)  # y=2*x1+delta/2.
    assert all(piece.lower == piece.upper == expected_parameter and piece.search_interval == (expected_parameter, expected_parameter)
               for piece in reduction.pieces)
    result = certify_gaussian_polyhedral_reduced_minimum(reduction)
    assert result is not None and result.sample_point == (F(1, 7), F(2))
    with mp.workdps(100):
        _check_global_result(mp, result, model, rows, bounds, target, ((_mp(mp, F(1, 7)), mp.mpf(2)),))


def test_coupled_three_dimensional_equalities_exercise_nonorthogonal_transverse_bases():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 0, 0), (1, 0, 0)), (0, 0), viscosity=F(1, 4))
    # x2+x3=1+x1 and x2+2*x3=3-x1 give x=(z,3*z-1,2-2*z).
    # Their independent projected normals have Gram matrix [[2,3],[3,5]].
    rows = ((-1, 1, 1), (1, -1, -1), (1, 1, 2), (-1, -1, -2))
    bounds, target = (1, -1, 3, -3), (0, 0, 0)
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, target)
    assert reduction is not None and reduction.pieces and reduction.enumerated_bases == 11
    assert any(len(piece.active_indices) == 2 for piece in reduction.pieces)
    for index, piece in enumerate(reduction.pieces):
        lower, upper = piece.search_interval
        parameter = (lower+upper)/2
        assert reduction.lift(index, parameter) == (parameter/2, 3*parameter/2-1, 2-parameter)
        assert reduction.transverse_value(index, parameter) == F(5, 2)-F(7, 2)*parameter+F(13, 8)*parameter**2
    # Five retained pieces fit the shared default storage budget at 63 boxes
    # each; reserving 255 for every piece would intentionally refuse here.
    result = certify_gaussian_polyhedral_reduced_minimum(reduction, max_boxes_per_piece=63)
    with mp.workdps(100):
        def stationarity(z):
            velocity = _original_velocity(mp, model, (z, 3*z-1, 2-2*z))
            return velocity[0]+3*velocity[1]-2*velocity[2]

        coordinate = mp.findroot(stationarity, mp.mpf("0.55"))
        point = (coordinate, 3*coordinate-1, 2-2*coordinate)
        # Along this line, the original Hessian is 14-2*sech(2z)^2 >=12,
        # so this stationary point is the unique global feasible minimum.
        assert mp.diff(stationarity, coordinate) > 12
        _check_global_result(mp, result, model, rows, bounds, target, (point,))


@pytest.mark.parametrize("rows,bounds", [(((0,),), (-1,)), (((1,), (-1,)), (0, -1))])
def test_empty_exact_piece_cover_means_infeasible_not_an_optimizer_result(rows, bounds):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4))
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, (0,))
    assert reduction is not None and reduction.pieces == ()
    assert certify_gaussian_polyhedral_reduced_minimum(reduction) is None
