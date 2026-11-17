# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified projected-mean residuals checked against original Gaussian roots."""

from fractions import Fraction as F
from itertools import product

import pytest
from omnibias.core.verified.gaussian_entropy_residual import certify_gaussian_entropy_residual
from omnibias.core.verified.gaussian_entropy_solver import solve_gaussian_entropy
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right, strict=True))


def _original(mp, model, point, target):
    logits = tuple(_mp(mp, bias-model.biases[0])
                   -sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
                   /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                   for bias, center in zip(model.biases, model.centers, strict=True))
    peak = max(logits)
    weights = tuple(mp.exp(value-peak) for value in logits)
    posterior = tuple(value/sum(weights) for value in weights)
    value = (-2*_mp(mp, model.viscosity)*(peak+mp.log(sum(weights)))
             -sum(_mp(mp, v)*(x-_mp(mp, c)) for v, x, c in zip(target, point, model.centers[0], strict=True)))
    return value, posterior


def _unique(model, *, rows=(), upper=(), target=None, reference=None):
    projector = compile_polyhedral_projector(model.dimension, rows, upper)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, target or (0,)*model.dimension)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=reference)
    assert unique is not None
    return unique


def _check(mp, residual, root):
    assert residual is not None
    unique = residual.unique
    model, projector, target = unique.domain.model, unique.domain.projector, unique.domain.target
    assert all(isinstance(x, F) for x in residual.point)
    assert sum(residual.probabilities) == 1 and min(residual.probabilities) >= 0
    assert all(_dot(row, residual.point) <= bound
               for row, bound in zip(projector.inequalities, projector.upper_bounds, strict=True))
    assert residual.projection_source == tuple(model.time*v+sum(p*center[axis]
                                                               for p, center in zip(residual.probabilities, model.centers, strict=True))
                                               for axis, v in enumerate(target))
    assert residual.normal == tuple((source-x)/model.time
                                    for source, x in zip(residual.projection_source, residual.point, strict=True))
    point = tuple(_mp(mp, x) for x in residual.point)
    value, posterior = _original(mp, model, point, target)
    optimum, _ = _original(mp, model, root, target)
    actual_residual = tuple(sum((_mp(mp, p)-prob)*_mp(mp, center[j])
                               for p, prob, center in zip(residual.probabilities, posterior, model.centers, strict=True))
                            /_mp(mp, model.time) for j in range(model.dimension))
    for interval, probability in zip(residual.posterior, posterior, strict=True):
        assert _mp(mp, interval.lo) <= probability <= _mp(mp, interval.hi)
    for interval, value_r in zip(residual.residual, actual_residual, strict=True):
        assert _mp(mp, interval.lo) <= value_r <= _mp(mp, interval.hi)
    assert sum(value_r**2 for value_r in actual_residual) <= _mp(mp, residual.residual_squared_bound)
    assert mp.sqrt(sum((x-y)**2 for x, y in zip(point, root, strict=True))) <= _mp(mp, residual.error_bound)
    assert -mp.mpf("1e-85") <= value-optimum <= _mp(mp, residual.suboptimality_bound)
    assert _mp(mp, residual.value.lo) <= value <= _mp(mp, residual.value.hi)
    assert _mp(mp, residual.minimum.lo) <= optimum <= _mp(mp, residual.minimum.hi)
    assert all(_mp(mp, axis.lo) <= x <= _mp(mp, axis.hi) for axis, x in zip(residual.enclosure, root, strict=True))
    rho = unique.variance_upper_bound/(2*model.viscosity*model.time)
    mu = (1-rho)/model.time
    assert residual.contraction_bound == rho
    assert residual.strong_convexity_bound == mu
    assert F(residual.error_bound)**2 >= residual.residual_squared_bound/mu**2


@pytest.mark.parametrize("probabilities", [(1, 0, 0), (0, 1, 0), (F(1, 3),)*3])
def test_nonconvex_residual_certifies_arbitrary_simplex_proposals(probabilities):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
    unique = _unique(model)
    assert model.contraction_bound == 4
    residual = certify_gaussian_entropy_residual(unique, probabilities)
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-4*mp.exp(-12+2*x)/(1+2*mp.exp(-12+2*x)), mp.mpf("0.000025"))
        _check(mp, residual, (root, root))


def test_nonconvex_solve_reports_real_accuracy_and_an_initial_evaluation_budget():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
    unique = _unique(model)
    coarse = solve_gaussian_entropy(unique, initial_probabilities=(0, 1, 0), tolerance=1e-12, max_iterations=0)
    assert coarse is not None and coarse.status == "iteration_budget"
    assert coarse.iterations == 0 and coarse.evaluations == 1 and coarse.selected_iteration == 0
    assert coarse.error_bounds == (coarse.residual.error_bound,)
    assert coarse.residual.error_bound > coarse.tolerance
    result = solve_gaussian_entropy(unique, initial_probabilities=(0, 1, 0), tolerance=1e-12, max_iterations=12)
    assert result is not None and result.status == "tolerance"
    assert result.residual.error_bound <= result.tolerance
    assert result.evaluations == result.iterations+1 == len(result.error_bounds)
    assert 0 <= result.selected_iteration <= result.iterations <= result.max_iterations
    assert result.residual.error_bound == min(result.error_bounds)
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-4*mp.exp(-12+2*x)/(1+2*mp.exp(-12+2*x)), mp.mpf("0.000025"))
        _check(mp, coarse.residual, (root, root))
        _check(mp, result.residual, (root, root))


def test_zero_tolerance_does_not_hide_stagnation_or_iteration_budget():
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
    result = solve_gaussian_entropy(_unique(model), tolerance=0, max_iterations=16)
    assert result is not None
    assert result.status in ("stagnation", "iteration_budget")
    assert result.residual.error_bound > 0
    assert result.evaluations == result.iterations+1 == len(result.error_bounds)
    assert result.residual.error_bound == min(result.error_bounds)


def test_original_stationarity_on_an_exact_equality_with_nonconvex_ambient_field():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    unique = _unique(model, rows=((1, 1), (-1, -1), (2, 2)), upper=(0, 0, 0))
    result = solve_gaussian_entropy(unique, initial_probabilities=(0, 1, 0), tolerance=1e-10, max_iterations=8)
    assert result is not None and result.status == "tolerance"
    assert sum(result.residual.point) == 0
    with mp.workdps(100):
        _check(mp, result.residual, (mp.mpf(0), mp.mpf(0)))


def test_normal_cone_corner_and_reference_center_need_no_projection_jacobian():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(tuple(product((-1, 1), repeat=2)), (0,)*4, viscosity=F(5, 4))
    unique = _unique(model, rows=((-1, 0), (0, -1)), upper=(0, 0), target=(-1, -1), reference=(0, 0))
    assert unique.variance_upper_bound == 2
    # This boundary proposal projects to the exact true optimum, but its
    # chosen normal-cone witness has a conservative nonzero residual.
    coarse = certify_gaussian_entropy_residual(unique, (1, 0, 0, 0))
    assert coarse is not None and coarse.point == (0, 0)
    result = solve_gaussian_entropy(unique, tolerance=1e-10, max_iterations=4)
    assert result is not None and result.status == "tolerance"
    with mp.workdps(100):
        _check(mp, coarse, (mp.mpf(0), mp.mpf(0)))
        _check(mp, result.residual, (mp.mpf(0), mp.mpf(0)))


def test_nonunit_time_viscosity_target_and_common_bias_use_original_scaling():
    mp = pytest.importorskip("mpmath")
    centers = ((0, 0), (1, 0), (0, 1))
    wanted, nu, age = (F(1, 4), F(-1, 5)), F(3, 2), F(5, 3)
    biases = tuple(10**400+sum((x-a)**2 for x, a in zip(wanted, center, strict=True))/(4*nu*age) for center in centers)
    target = tuple((x-F(1, 3))/age for x in wanted)
    unique = _unique(GaussianHeatMap(centers, biases, viscosity=nu, time=age), target=target)
    residual = certify_gaussian_entropy_residual(unique, (F(1, 3),)*3)
    assert residual is not None and residual.point == wanted
    with mp.workdps(100):
        _check(mp, residual, tuple(_mp(mp, x) for x in wanted))


def test_underflow_and_coincident_modes_allow_exact_zero_error_at_zero_tolerance():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((F(2, 7), F(-3, 11)),)*2, (0, -10000), viscosity=F(3, 2), time=F(5, 3))
    target = (F(1, 3), F(-2, 5))
    unique = _unique(model, target=target)
    assert not unique.domain.gradient_safe and unique.variance_upper_bound == 0
    result = solve_gaussian_entropy(unique, initial_probabilities=(0, 1), tolerance=0, max_iterations=0)
    assert result is not None and result.status == "tolerance"
    assert result.evaluations == 1 and result.iterations == 0
    assert result.residual.error_bound == result.residual.suboptimality_bound == 0
    wanted = tuple(c+model.time*v for c, v in zip(model.centers[0], target, strict=True))
    assert result.residual.point == wanted
    with mp.workdps(100):
        _check(mp, result.residual, tuple(_mp(mp, x) for x in wanted))
