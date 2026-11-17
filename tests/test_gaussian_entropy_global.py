# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent Gibbs, projection-envelope and global-coverage regressions."""

from fractions import Fraction as F
from itertools import product

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_entropy_minimum import (
    _EntropyOracle,
    certify_gaussian_entropy_minimum,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right, strict=True))


def _weights(parameters, count):
    remaining, weights = F(1), []
    for parameter in parameters[:count-1]:
        value = F(parameter)
        weights.append(remaining*value)
        remaining *= 1-value
    return (*weights, remaining)


def _original(mp, model, point, target):
    nu, age = _mp(mp, model.viscosity), _mp(mp, model.time)
    distances = tuple(sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
                      for center in model.centers)
    logits = tuple(_mp(mp, bias-model.biases[0])-distance/(4*nu*age)
                   for bias, distance in zip(model.biases, distances, strict=True))
    peak = max(logits)
    weights = tuple(mp.exp(value-peak) for value in logits)
    posterior = tuple(value/sum(weights) for value in weights)
    linear = sum(_mp(mp, v)*(x-_mp(mp, c)) for v, x, c in zip(target, point, model.centers[0], strict=True))
    value = -2*nu*(peak+mp.log(sum(weights)))-linear
    return value, posterior, distances, linear


def _contains(mp, box, point):
    return all(_mp(mp, axis.lo) <= value <= _mp(mp, axis.hi)
               for axis, value in zip(box, point, strict=True))


def _check(mp, result, domain, minimizers):
    assert result is not None
    model, projector, target = domain.model, domain.projector, domain.target
    sample = tuple(_mp(mp, value) for value in result.sample_point)
    assert all(isinstance(value, F) for value in result.sample_point)
    assert all(_dot(row, result.sample_point) <= bound
               for row, bound in zip(projector.inequalities, projector.upper_bounds, strict=True))
    value, posterior, distances, linear = _original(mp, model, sample, target)
    p = _weights(result.parameter_result.point, model.mode_count)
    entropy = sum(_mp(mp, weight)*(mp.log(_mp(mp, weight))-_mp(mp, bias-model.biases[0]))
                  for weight, bias in zip(p, model.biases, strict=True) if weight)
    joint = sum(_mp(mp, weight)*distance for weight, distance in zip(p, distances, strict=True))/(2*_mp(mp, model.time))
    joint += 2*_mp(mp, model.viscosity)*entropy-linear
    kl = sum(_mp(mp, weight)*mp.log(_mp(mp, weight)/probability)
             for weight, probability in zip(p, posterior, strict=True) if weight)
    assert abs(joint-value-2*_mp(mp, model.viscosity)*kl) < mp.mpf("1e-80")
    assert kl > -mp.mpf("1e-85")
    assert _mp(mp, result.sample_value.lo) <= value <= _mp(mp, result.sample_value.hi)
    assert _mp(mp, result.sample_entropy.lo) <= joint <= _mp(mp, result.sample_entropy.hi)
    assert len(result.candidate_boxes) == len(result.candidate_parameters)
    for point in minimizers:
        minimum, posterior, _, _ = _original(mp, model, point, target)
        assert _mp(mp, result.minimum.lo) <= minimum <= _mp(mp, result.minimum.hi)
        assert minimum <= value+mp.mpf("1e-80")
        assert any(_contains(mp, box, point) for box in result.candidate_boxes)
        assert all(_mp(mp, lo) <= x <= _mp(mp, hi)
                   for (lo, hi), x in zip(domain.physical_bounds, point, strict=True))
        for j, axis in enumerate(domain.parameter_box[:model.mode_count-1]):
            conditional = posterior[j]/sum(posterior[j:])
            assert _mp(mp, axis.lo) <= conditional <= _mp(mp, axis.hi)
    if result.status == "value_tolerance":
        assert result.gap_upper <= result.value_tolerance


@pytest.mark.parametrize("constrained", [False, True])
def test_entropy_search_preserves_all_noncollinear_global_ties(constrained):
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(tuple(product((-1, 1), repeat=2)), (0,)*4, viscosity=F(1, 4))
    rows, bounds = (((1, 1), (-1, -1)), (0, 0)) if constrained else ((), ())
    projector = compile_polyhedral_projector(2, rows, bounds)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, (0, 0))
    assert domain is not None and domain.gradient_safe
    result = certify_gaussian_entropy_minimum(domain, max_boxes=31)
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-mp.tanh(2*x), 1)
        points = ((root, -root), (-root, root)) if constrained else tuple(product((-root, root), repeat=2))
        _check(mp, result, domain, points)
    assert result is not None and result.parameter_result.gradient_evaluations > 0


def test_original_gauge_and_gibbs_identity_for_translated_nonzero_target():
    mp = pytest.importorskip("mpmath")
    origin = (F(2, 7), F(-3, 11))
    centers = (origin, (origin[0]+1, origin[1]), (origin[0], origin[1]+1))
    wanted = (origin[0]+F(1, 4), origin[1]+F(1, 5))
    target = tuple(wanted[j]-sum(center[j] for center in centers)/3 for j in range(2))
    biases = tuple(10**400+sum((x-a)**2 for x, a in zip(wanted, center, strict=True))/4 for center in centers)
    model = GaussianHeatMap(centers, biases)
    projector = compile_polyhedral_projector(2, (), ())
    assert projector is not None and model.contraction_bound == F(1, 4)
    domain = certify_gaussian_posterior_domain(model, projector, target)
    assert domain is not None and domain.gradient_safe
    result = certify_gaussian_entropy_minimum(domain, max_boxes=31)
    with mp.workdps(100):
        # Original logits agree exactly at wanted, fixing its posterior mean;
        # q<1 proves this stationary point is the unique global optimum.
        _check(mp, result, domain, (tuple(_mp(mp, x) for x in wanted),))


def test_envelope_gradient_is_valid_across_a_nondifferentiable_projection():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0))
    projector = compile_polyhedral_projector(2, ((-1, 0),), (0,))
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, (0, 0))
    assert domain is not None and domain.gradient_safe
    # For p=(z,1-z), x_p=(max(1-2z,0),0). Its left and right
    # derivatives differ at z=1/2. Direct substitution in G gives
    # E(z)=1/2-max(1-2z,0)^2/2+2[z log z+(1-z)log(1-z)].
    with mp.workdps(100), _strict():
        oracle = _EntropyOracle(domain, _Arithmetic(_combined_budget(model.budget, projector.budget)))
        box = (Interval(0.375, 0.625),)
        derivative = oracle.gradient(box)[0]
        values = oracle.value(box)
        for j in range(17):
            z = mp.mpf(3)/8+mp.mpf(j)/64
            exact_value = mp.mpf("0.5")-max(1-2*z, 0)**2/2+2*(z*mp.log(z)+(1-z)*mp.log(1-z))
            exact_derivative = 2*max(1-2*z, 0)+2*mp.log(z/(1-z))
            assert _mp(mp, values.lo) <= exact_value <= _mp(mp, values.hi)
            assert _mp(mp, derivative.lo) <= exact_derivative <= _mp(mp, derivative.hi)
        for z in (0.375, 0.5, 0.625):
            value = oracle.gradient((Interval.point(z),))[0]
            expected = 2*max(1-2*z, 0)+2*mp.log(mp.mpf(z)/(1-z))
            assert _mp(mp, value.lo) <= expected <= _mp(mp, value.hi)
            assert value.hi-value.lo < 1e-12


def test_underflowed_posterior_bounds_keep_boundary_domain_and_disable_gradient():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (0, 0)), (0, -10000))
    target = (F(1, 3), F(-2, 5))
    projector = compile_polyhedral_projector(2, (), ())
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, target)
    assert domain is not None and not domain.gradient_safe
    assert domain.parameter_box[0].hi == 1.0
    result = certify_gaussian_entropy_minimum(domain, max_boxes=7)
    with mp.workdps(100):
        _check(mp, result, domain, (tuple(_mp(mp, x) for x in target),))
    assert result is not None and result.sample_point == target
    assert result.parameter_result.gradient_evaluations == 0


def test_one_mode_dummy_coordinate_with_an_exact_equality_projection():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((F(1, 3), F(-2, 7)),), (10**400,), time=F(3, 2))
    target = (F(2, 5), F(-1, 6))
    projector = compile_polyhedral_projector(2, ((1, 2), (-1, -2)), (1, -1))
    assert projector is not None
    source = tuple(c+model.time*v for c, v in zip(model.centers[0], target, strict=True))
    displacement = (_dot((1, 2), source)-1)/5
    wanted = (source[0]-displacement, source[1]-2*displacement)
    domain = certify_gaussian_posterior_domain(model, projector, target)
    assert domain is not None and domain.parameter_box == (Interval.point(0.0),)
    result = certify_gaussian_entropy_minimum(domain, max_boxes=7)
    with mp.workdps(100):
        _check(mp, result, domain, (tuple(_mp(mp, x) for x in wanted),))
    assert result is not None and result.sample_point == wanted
