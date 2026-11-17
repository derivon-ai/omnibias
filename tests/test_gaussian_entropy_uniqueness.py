# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Posterior-radius uniqueness checked independently of spatial convexity."""

from fractions import Fraction as F
from itertools import product

import pytest
from omnibias.core.verified.gaussian_entropy_minimum import certify_gaussian_entropy_minimum
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _potential(mp, model, point):
    logits = tuple(_mp(mp, bias-model.biases[0])
                   -sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
                   /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                   for bias, center in zip(model.biases, model.centers, strict=True))
    peak = max(logits)
    return -2*_mp(mp, model.viscosity)*(peak+mp.log(sum(mp.exp(value-peak) for value in logits)))


def _weights(parameters):
    remaining, result = F(1), []
    for z in parameters:
        result.append(remaining*z)
        remaining *= 1-z
    return (*result, remaining)


def _domain(model, rows=(), upper=(), target=None):
    projector = compile_polyhedral_projector(model.dimension, rows, upper)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, target or (0,)*model.dimension)
    assert domain is not None
    return domain


def test_unique_global_minimum_with_a_negative_original_spatial_hessian():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
    domain = _domain(model)
    unique = certify_gaussian_entropy_unique(domain)
    assert unique is not None and model.contraction_bound == 4
    assert unique.reference == (0, 0) and unique.squared_radii == (0, 16, 16)
    assert unique.variance_upper_bound == 16*(1-F(domain.parameter_box[0].lo)) < 1
    assert unique.uniqueness_margin == 2-unique.variance_upper_bound > 1
    assert unique.entropy_monotonicity_lower_bound == unique.uniqueness_margin
    minimum = certify_gaussian_entropy_minimum(domain, max_boxes=31)
    assert minimum is not None
    with mp.workdps(100):
        # Differentiate the ORIGINAL Gaussian distance formula, not a
        # covariance or entropy helper: the potential is truly nonconvex.
        second = mp.diff(lambda x: _potential(mp, model, (x, mp.mpf(0))), mp.mpf(6), 2)
        assert second < mp.mpf("-0.9")
        # Reflection symmetry reduces original stationarity to this equation.
        root = mp.findroot(lambda x: x-4*mp.exp(-12+2*x)/(1+2*mp.exp(-12+2*x)), mp.mpf("0.000025"))
        assert 0 < root < mp.mpf("0.001")
        value = _potential(mp, model, (root, root))
        assert _mp(mp, minimum.minimum.lo) <= value <= _mp(mp, minimum.minimum.hi)
        assert any(all(_mp(mp, axis.lo) <= root <= _mp(mp, axis.hi) for axis in box)
                   for box in minimum.candidate_boxes)


def test_exact_expectation_extrema_and_convex_conditional_mass_image():
    model = GaussianHeatMap(tuple(product((0, 1), repeat=2)), (0,)*4)
    domain = _domain(model)
    unique = certify_gaussian_entropy_unique(domain)
    assert unique is not None
    bounds = tuple((F(axis.lo), F(axis.hi)) for axis in domain.parameter_box)
    vertices = tuple(_weights(z) for z in product(*bounds))
    # Independent enumeration of all cube corners checks the exact scalar
    # recurrence without calling the production expectation helper.
    values = tuple(sum(p*r for p, r in zip(weights, unique.squared_radii, strict=True)) for weights in vertices)
    assert unique.expectation_bounds == (min(values), max(values))
    for left in vertices:
        for right in vertices:
            p = tuple(F(2, 5)*a+F(3, 5)*b for a, b in zip(left, right, strict=True))
            inverse = []
            for j, (lo, hi) in enumerate(bounds):
                mass = sum(p[j:])
                assert lo*mass <= p[j] <= hi*mass
                inverse.append(p[j]/mass)
            assert _weights(inverse) == p


def test_reference_choice_can_change_an_inconclusive_sufficient_test():
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    domain = _domain(model)
    assert certify_gaussian_entropy_unique(domain) is None
    centered = certify_gaussian_entropy_unique(domain, reference=(0,))
    assert centered is not None
    assert centered.squared_radii == (1, 1)
    assert centered.expectation_bounds == (1, 1)
    assert centered.uniqueness_margin == centered.entropy_monotonicity_lower_bound == 1
    # Refusal for another valid reference says nothing about actual uniqueness.
    assert certify_gaussian_entropy_unique(domain, reference=(100,)) is None


def test_true_tied_global_minima_are_not_mislabeled_unique():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4))
    domain = _domain(model)
    assert model.contraction_bound == 2
    assert certify_gaussian_entropy_unique(domain) is None
    assert certify_gaussian_entropy_unique(domain, reference=(0,)) is None
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-mp.tanh(2*x), 1)
        assert root > mp.mpf("0.9")
        assert abs(_potential(mp, model, (root,))-_potential(mp, model, (-root,))) < mp.mpf("1e-90")
        assert _potential(mp, model, (root,)) < _potential(mp, model, (mp.mpf(0),))


def test_lower_dimensional_equality_constraints_need_no_interior_or_jacobian():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    domain = _domain(model, ((1, 1), (-1, -1), (2, 2)), (0, 0, 0))
    unique = certify_gaussian_entropy_unique(domain)
    assert unique is not None and model.contraction_bound == 4
    minimum = certify_gaussian_entropy_minimum(domain, max_boxes=7)
    assert minimum is not None and sum(minimum.sample_point) == 0
    with mp.workdps(100):
        # Original constrained stationarity follows from symmetry on x+y=0.
        derivative = mp.diff(lambda t: _potential(mp, model, (t, -t)), mp.mpf(0))
        assert abs(derivative) < mp.mpf("1e-90")
        value = _potential(mp, model, (mp.mpf(0), mp.mpf(0)))
        assert _mp(mp, minimum.minimum.lo) <= value <= _mp(mp, minimum.minimum.hi)
        assert any(all(axis.lo <= 0 <= axis.hi for axis in box) for box in minimum.candidate_boxes)


def test_boundary_touching_domain_can_still_prove_uniqueness():
    model = GaussianHeatMap(((0, 0), (0, 0)), (0, -10000), viscosity=F(3, 2), time=F(5, 3))
    domain = _domain(model, target=(F(1, 3), F(-2, 5)))
    assert not domain.gradient_safe and domain.parameter_box[0].hi == 1.0
    unique = certify_gaussian_entropy_unique(domain)
    assert unique is not None and unique.expectation_bounds == (0, 0)
    assert unique.uniqueness_margin == 5
    assert unique.entropy_monotonicity_lower_bound == 3


def test_unique_orthant_projection_at_a_switch_does_not_require_a_smooth_optimizer():
    model = GaussianHeatMap(((0,),), (0,))
    domain = _domain(model, ((-1,),), (0,))
    unique = certify_gaussian_entropy_unique(domain)
    assert unique is not None and unique.variance_upper_bound == 0
    # For nearby targets this optimizer is max(v,0), which has no ordinary
    # derivative at v=0 despite the global uniqueness proved here.
    projector = domain.projector
    left, right = projector.project((-1,)), projector.project((1,))
    assert left is not None and right is not None
    assert left.point == (0,) and right.point == (1,)
