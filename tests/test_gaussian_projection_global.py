# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global projected-simplex bounds checked against original Gaussian fields."""

from fractions import Fraction as F
from itertools import product

import pytest
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_projection_minimum import certify_gaussian_projection_minimum
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right, strict=True))


def _objective(mp, model, point, target):
    # Independent original Gaussian distances. Remove b0 before conversion,
    # omit the heat prefactor, then add target.a0 to match the reported gauge.
    nu, age = _mp(mp, model.viscosity), _mp(mp, model.time)
    logits = tuple(_mp(mp, bias-model.biases[0])
                   -sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))/(4*nu*age)
                   for bias, center in zip(model.biases, model.centers, strict=True))
    peak = max(logits)
    return (-2*nu*(peak+mp.log(sum(mp.exp(value-peak) for value in logits)))
            -sum(_mp(mp, v)*(x-_mp(mp, c)) for v, x, c in zip(target, point, model.centers[0], strict=True)))


def _contains(mp, box, point):
    return all(_mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
               for interval, value in zip(box, point, strict=True))


def _check(mp, result, model, projector, target, minimizers):
    assert result is not None
    assert all(isinstance(value, F) for value in result.sample_point)
    assert all(_dot(row, result.sample_point) <= bound
               for row, bound in zip(projector.inequalities, projector.upper_bounds, strict=True))
    sample = tuple(_mp(mp, value) for value in result.sample_point)
    sample_value = _objective(mp, model, sample, target)
    assert _mp(mp, result.sample_value.lo) <= sample_value <= _mp(mp, result.sample_value.hi)
    assert result.parameter_result.gradient_evaluations == 0
    assert len(result.candidate_boxes) == len(result.candidate_parameters)
    for point in minimizers:
        value = _objective(mp, model, point, target)
        assert _mp(mp, result.minimum.lo) <= value <= _mp(mp, result.minimum.hi)
        assert value <= sample_value+mp.mpf("1e-80")
        assert any(_contains(mp, box, point) for box in result.candidate_boxes)
    if result.status == "value_tolerance":
        assert result.gap_upper <= result.value_tolerance


@pytest.mark.parametrize("boxes", [1, 127])
def test_noncollinear_product_mixture_preserves_four_tied_global_minima(boxes):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(tuple(product((-1, 1), repeat=2)), (0,)*4, viscosity=F(1, 4))
    projector = compile_polyhedral_projector(2, (), ())
    assert projector is not None and model.contraction_bound > 1
    result = certify_gaussian_projection_minimum(model, projector, (0, 0), max_boxes=boxes)
    with mp.workdps(100):
        # The ORIGINAL Gaussian density factors into two scalar equal-weight
        # mixtures. Each has precisely two global modes ±root; 0 is unstable.
        root = mp.findroot(lambda x: x-mp.tanh(2*x), (mp.mpf("0.9"), mp.mpf(1)))
        assert mp.mpf("0.95") < root < 1
        assert 1-2/mp.cosh(2*root)**2 > 0
        points = tuple(product((-root, root), repeat=2))
        _check(mp, result, model, projector, (0, 0), points)
        assert _objective(mp, model, points[0], (0, 0)) < _objective(mp, model, (mp.mpf(0),)*2, (0, 0))


@pytest.mark.parametrize("kind", ["equality", "halfspace"])
def test_nonconvex_constraints_retain_all_feasible_winning_modes(kind):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(tuple(product((-1, 1), repeat=2)), (0,)*4, viscosity=F(1, 4))
    rows = ((1, 1), (-1, -1), (2, 2)) if kind == "equality" else ((-1, 0),)
    bounds = (0,)*len(rows)
    target = (F(1, 7), F(1, 7)) if kind == "equality" else (0, 0)
    projector = compile_polyhedral_projector(2, rows, bounds)
    assert projector is not None
    result = certify_gaussian_projection_minimum(model, projector, target, max_boxes=31)
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-mp.tanh(2*x), 1)
        points = ((root, -root), (-root, root)) if kind == "equality" else ((root, root), (root, -root))
        _check(mp, result, model, projector, target, points)
        if kind == "equality":
            assert result is not None and sum(result.sample_point) == 0


def test_projected_hull_not_hull_of_projected_vertices_contains_actual_optimizer():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 1), (1, -1)), (0, 0))
    projector = compile_polyhedral_projector(2, ((-1, 0), (0, -1)), (0, 0))
    assert projector is not None and model.contraction_bound == 1
    endpoints = tuple(projector.project(center) for center in model.centers)
    assert all(point is not None and sum(point.point) == 1 for point in endpoints)
    midpoint = projector.project((0, 0))
    assert midpoint is not None and midpoint.point == (0, 0)
    # Every convex combination of projected vertices has x+y=1, so misses
    # the actual unique global optimizer at the projected original midpoint.
    result = certify_gaussian_projection_minimum(model, projector, (0, 0), max_boxes=63)
    with mp.workdps(100):
        _check(mp, result, model, projector, (0, 0), ((mp.mpf(0), mp.mpf(0)),))
    assert result is not None and result.sample_point == (0, 0)


@pytest.mark.parametrize("count", [1, 3])
def test_one_or_coincident_modes_reduce_to_exact_oblique_projection(count):
    mp = pytest.importorskip("mpmath")
    center, target = (F(2, 7), F(-3, 11)), (F(1, 3), F(-2, 5))
    model = GaussianHeatMap((center,)*count, tuple(10**400+F(i, 3) for i in range(count)), time=F(3, 2))
    projector = compile_polyhedral_projector(2, ((1, 2), (-1, -2)), (1, -1))
    assert projector is not None
    source = tuple(c+model.time*v for c, v in zip(center, target, strict=True))
    # Independent equality projection onto x+2y=1.
    shift = (_dot((1, 2), source)-1)/5
    wanted = (source[0]-shift, source[1]-2*shift)
    result = certify_gaussian_projection_minimum(model, projector, target, max_boxes=7)
    assert result is not None and result.sample_point == wanted
    with mp.workdps(100):
        _check(mp, result, model, projector, target, (tuple(_mp(mp, value) for value in wanted),))


def test_noncollinear_equal_logit_stationary_point_on_lower_dimensional_polyhedron():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (F(1, 8),)*3)
    target = (0, 0)
    projector = compile_polyhedral_projector(2, ((1, 1), (-1, -1), (-1, 0), (0, -1)), (1, -1, 0, 0))
    assert projector is not None and model.contraction_bound == F(1, 4)
    result = certify_gaussian_projection_minimum(model, projector, target, max_boxes=31)
    with mp.workdps(100):
        # At (1/2,1/2) all ORIGINAL distances agree. The mean is (1/3,1/3)
        # and velocity (1/6,1/6) is normal to the equality; q<1 gives uniqueness.
        _check(mp, result, model, projector, target, ((mp.mpf("0.5"), mp.mpf("0.5")),))
    assert result is not None and sum(result.sample_point) == 1


def test_translated_noncollinear_model_with_nonzero_target_and_huge_bias_gauge():
    mp = pytest.importorskip("mpmath")
    origin = (F(2, 7), F(-3, 11))
    centers = (origin, (origin[0]+1, origin[1]), (origin[0], origin[1]+1))
    wanted = (origin[0]+F(1, 4), origin[1]+F(1, 5))
    mean = tuple(sum(center[axis] for center in centers)/3 for axis in range(2))
    target = tuple(x-m for x, m in zip(wanted, mean, strict=True))
    biases = tuple(10**400+sum((x-a)**2 for x, a in zip(wanted, center, strict=True))/4 for center in centers)
    model = GaussianHeatMap(centers, biases)
    projector = compile_polyhedral_projector(2, (), ())
    assert projector is not None and model.contraction_bound == F(1, 4)
    result = certify_gaussian_projection_minimum(model, projector, target, max_boxes=31)
    with mp.workdps(100):
        # Exact equal original logits fix the posterior mean, while q<1 makes
        # this stationary point the unique global minimum.
        _check(mp, result, model, projector, target, (tuple(_mp(mp, value) for value in wanted),))


def test_infeasible_projection_refuses_without_manufacturing_a_global_result():
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (0, 0, 0))
    projector = compile_polyhedral_projector(2, ((0, 0),), (-1,))
    assert projector is not None and projector.project((0, 0)) is None
    assert certify_gaussian_projection_minimum(model, projector, (0, 0)) is None
