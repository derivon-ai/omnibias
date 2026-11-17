# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global uniqueness and target sensitivities against original Gaussian KKT."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    certify_gaussian_polyhedral_reduced_minimum,
)
from omnibias.core.verified.gaussian_polyhedral_reduced_sensitivity import (
    certify_gaussian_polyhedral_reduced_sensitivity,
)
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    compile_gaussian_polyhedral_reduction,
)
from omnibias.core.verified.gaussian_polyhedral_unique import certify_gaussian_polyhedral_unique


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right, strict=True))


def _solve(model, rows, bounds, target, *, max_boxes=255):
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, target)
    assert reduction is not None and reduction.feasible
    minimum = certify_gaussian_polyhedral_reduced_minimum(reduction, max_boxes_per_piece=max_boxes)
    assert minimum is not None
    return reduction, minimum


def _original_velocity(mp, model, point):
    logits = tuple(_mp(mp, bias-model.biases[0])
                   -sum((x-_mp(mp, center))**2 for x, center in zip(point, row, strict=True))
                   /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                   for bias, row in zip(model.biases, model.centers, strict=True))
    weights = tuple(mp.exp(value-max(logits)) for value in logits)
    mean = tuple(sum(weight*_mp(mp, center[axis]) for weight, center in zip(weights, model.centers, strict=True))/sum(weights)
                 for axis in range(model.dimension))
    return tuple((x-a)/_mp(mp, model.time) for x, a in zip(point, mean, strict=True))


def _original_kkt_root(mp, model, rows, bounds, target, start):
    dimension = model.dimension

    def equations(*values):
        point, multipliers = values[:dimension], values[dimension:]
        velocity = _original_velocity(mp, model, point)
        stationary = tuple(velocity[axis]-target[axis]
                           +sum(_mp(mp, row[axis])*multiplier for row, multiplier in zip(rows, multipliers, strict=True))
                           for axis in range(dimension))
        constraints = tuple(sum(_mp(mp, a)*x for a, x in zip(row, point, strict=True))-_mp(mp, bound)
                            for row, bound in zip(rows, bounds, strict=True))
        return (*stationary, *constraints)

    return tuple(mp.findroot(equations, start))


def _assert_contains(mp, enclosure, value):
    assert _mp(mp, enclosure.lo) <= value <= _mp(mp, enclosure.hi)


def test_pruned_middle_piece_still_blocks_false_uniqueness_of_two_outer_minima():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=F(1, 4))
    reduction, minimum = _solve(model, ((2, -1), (-2, -1)), (F(1, 2), F(1, 2)), (0, 0))
    middle = next(index for index, piece in enumerate(reduction.pieces) if piece.active_indices == ())
    assert reduction.pieces[middle].scalar_curvature == F(1, 2)
    assert middle not in minimum.candidate_piece_indices
    assert all(reduction.pieces[index].scalar_curvature == F(5, 2) for index in minimum.candidate_piece_indices)
    assert certify_gaussian_polyhedral_unique(minimum) is None
    with mp.workdps(100):
        # On the positive outer face x=(z,2z-1/2), use original Gaussian
        # stationarity along its tangent, then reflect to obtain the tied root.
        def stationary(z):
            velocity = _original_velocity(mp, model, (z, 2*z-mp.mpf("0.5")))
            return velocity[0]+2*velocity[1]

        z = mp.findroot(stationary, mp.mpf("0.31"))
        assert z > mp.mpf("0.25")
        for point in ((z, 2*z-mp.mpf("0.5")), (-z, 2*z-mp.mpf("0.5"))):
            assert any(all(_mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
                           for interval, value in zip(box, point, strict=True)) for box in minimum.candidate_boxes)


def test_critical_unique_minimum_has_no_finite_target_jacobian_certificate():
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 2))
    _, minimum = _solve(model, (), (), (0,))
    unique = certify_gaussian_polyhedral_unique(minimum)
    assert unique is not None and unique.method == "strict_convexity"
    assert unique.curvature_lower_bound == 0
    assert unique.parameter_hull[0] <= 0 <= unique.parameter_hull[1]
    assert unique.checked_piece_indices == (0,)
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is None


def test_singleton_breakpoint_proof_does_not_infer_a_stable_interior_face():
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=F(1, 4))
    reduction, minimum = _solve(model, ((1, -1), (-1, -1)), (-1, -1), (0, 0))
    unique = certify_gaussian_polyhedral_unique(minimum)
    assert unique is not None and unique.method == "singleton"
    assert unique.curvature_lower_bound is None and unique.parameter_hull == (0, 0)
    assert all(certify_gaussian_polyhedral_reduced_sensitivity(unique, index) is None
               for index in range(len(reduction.pieces)))


def test_unconstrained_nonconvex_tie_is_not_unique():
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4))
    _, minimum = _solve(model, (), (), (0,))
    assert certify_gaussian_polyhedral_unique(minimum) is None


@pytest.mark.parametrize("geometry", ["scalar", "rotated"])
def test_selected_nonconvex_global_branch_jacobian_contains_original_target_secants(geometry):
    mp = pytest.importorskip("mpmath")
    if geometry == "scalar":
        model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4))
        target = (F(1, 10),)
    else:
        center, half, transverse = (F(2, 7), F(-3, 11)), (F(3, 5), F(4, 5)), (F(-4, 25), F(3, 25))
        centers = tuple(tuple(c+sign*a for c, a in zip(center, half, strict=True)) for sign in (-1, 1))
        model = GaussianHeatMap(centers, (0, F(5, 7)), viscosity=F(1, 8), time=2)
        target = tuple(u+a/2*(F(1, 10)-F(5, 28)) for u, a in zip(transverse, half, strict=True))
    _, minimum = _solve(model, (), (), target)
    unique = certify_gaussian_polyhedral_unique(minimum)
    assert unique is not None and unique.method == "strong_convexity"
    assert unique.curvature_lower_bound is not None and unique.curvature_lower_bound > 0
    sensitivity = certify_gaussian_polyhedral_reduced_sensitivity(unique, 0)
    assert sensitivity is not None and sensitivity.denominator.lo > 0
    with mp.workdps(100):
        goal = tuple(_mp(mp, value) for value in target)
        root = _original_kkt_root(mp, model, (), (), goal, tuple(_mp(mp, value) for value in minimum.sample_point))
        hessian = mp.matrix(model.dimension)
        for row in range(model.dimension):
            for column in range(model.dimension):
                def entry(value, *, row=row, column=column):
                    point = (*root[:column], value, *root[column+1:])
                    return _original_velocity(mp, model, point)[row]

                hessian[row, column] = mp.diff(entry, root[column])
        expected = hessian**-1
        step = mp.mpf("1e-10")
        for column in range(model.dimension):
            upper = tuple(value+(step if index == column else 0) for index, value in enumerate(goal))
            lower = tuple(value-(step if index == column else 0) for index, value in enumerate(goal))
            plus = _original_kkt_root(mp, model, (), (), upper, root)
            minus = _original_kkt_root(mp, model, (), (), lower, root)
            for row in range(model.dimension):
                secant = (plus[row]-minus[row])/(2*step)
                assert abs(secant-expected[row, column]) < mp.mpf("1e-17")
                _assert_contains(mp, sensitivity.target_jacobian[row][column], expected[row, column])
                _assert_contains(mp, sensitivity.target_jacobian[row][column], secant)
        cotangent = tuple(F(index+1, index+2) for index in range(model.dimension))
        vjp = sensitivity.vjp(cotangent)
        for column in range(model.dimension):
            wanted = sum(expected[row, column]*_mp(mp, value) for row, value in enumerate(cotangent))
            _assert_contains(mp, vjp[column], wanted)


def test_stable_face_jacobian_is_certified_despite_indefinite_original_ambient_hessian():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 0, 0), (1, 0, 0)), (0, 0), viscosity=F(1, 4))
    rows, bounds, target = ((-1, 1, 1), (-1, -1, -2)), (1, -3), (-7, 0, 0)
    reduction, minimum = _solve(model, rows, bounds, target, max_boxes=63)
    unique = certify_gaussian_polyhedral_unique(minimum)
    assert unique is not None and unique.method == "strong_convexity"
    assert unique.parameter_hull[0] < 0 < unique.parameter_hull[1]
    index = next(index for index, piece in enumerate(reduction.pieces) if piece.active_indices == (0, 1))
    sensitivity = certify_gaussian_polyhedral_reduced_sensitivity(unique, index)
    assert sensitivity is not None
    assert sensitivity.scalar_curvature == 7 and sensitivity.point_slope == (F(1, 2), F(3, 2), F(-1))
    assert sensitivity.base_jacobian == ((0, 0, 0),)*3
    tangent = (F(1), F(3), F(-2))
    expected = tuple(tuple(a*b/12 for b in tangent) for a in tangent)
    for row, wanted in zip(sensitivity.target_jacobian, expected, strict=True):
        for interval, value in zip(row, wanted, strict=True):
            assert F(interval.lo) <= value <= F(interval.hi)
    assert sensitivity.multipliers[0].lo <= 4 <= sensitivity.multipliers[0].hi
    assert sensitivity.multipliers[1].lo <= 3 <= sensitivity.multipliers[1].hi
    assert all(interval.lo > 0 for interval in sensitivity.multipliers)
    # Contractions occur in exact geometry before interval conversion.
    for normal in rows:
        assert all(interval.lo == interval.hi == 0 for interval in sensitivity.vjp(normal))
    with mp.workdps(100):
        goal = tuple(_mp(mp, value) for value in target)
        start = tuple(_mp(mp, value) for value in (0, -1, 2, 4, 3))
        root = _original_kkt_root(mp, model, rows, bounds, goal, start)
        assert max(abs(a-b) for a, b in zip(root, start, strict=True)) < mp.mpf("1e-90")
        ambient = mp.diff(lambda x: _original_velocity(mp, model, (x, mp.mpf(-1), mp.mpf(2)))[0], 0)
        assert abs(ambient+1) < mp.mpf("1e-90")
        for column in range(3):
            step = mp.mpf("1e-10")
            plus_goal = tuple(value+(step if index == column else 0) for index, value in enumerate(goal))
            minus_goal = tuple(value-(step if index == column else 0) for index, value in enumerate(goal))
            plus = _original_kkt_root(mp, model, rows, bounds, plus_goal, root)
            minus = _original_kkt_root(mp, model, rows, bounds, minus_goal, root)
            assert all(value > 0 for value in (*plus[3:], *minus[3:]))
            for row in range(3):
                secant = (plus[row]-minus[row])/(2*step)
                assert abs(secant-_mp(mp, expected[row][column])) < mp.mpf("1e-19")
                _assert_contains(mp, sensitivity.target_jacobian[row][column], secant)
