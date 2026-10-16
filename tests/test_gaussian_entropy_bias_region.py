# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite original-Gaussian perturbations of certified global decisions."""

from fractions import Fraction as F
from itertools import product

import pytest
from omnibias.core.verified.gaussian_entropy_bias_face import certify_gaussian_entropy_bias_face
from omnibias.core.verified.gaussian_entropy_bias_region import certify_gaussian_entropy_bias_region
from omnibias.core.verified.gaussian_entropy_face import certify_gaussian_entropy_face
from omnibias.core.verified.gaussian_entropy_solver import solve_gaussian_entropy
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _case(model, rows=(), upper=(), target=None, reference=None):
    projector = compile_polyhedral_projector(model.dimension, rows, upper)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, target or (0,)*model.dimension)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=reference)
    assert unique is not None
    return unique


def _face(unique):
    solution = solve_gaussian_entropy(unique, tolerance=1e-12, max_iterations=12)
    assert solution is not None and solution.status == "tolerance"
    face = certify_gaussian_entropy_face(solution.residual)
    assert face is not None
    return face


def _posterior(mp, model, point, offsets):
    # Original squared-distance kernels; gauges cancel exactly before MP input.
    logits = tuple(_mp(mp, b-model.biases[0]+F(change)-F(offsets[0]))
                   -sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
                   /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                   for b, change, center in zip(model.biases, offsets, model.centers, strict=True))
    weights = tuple(mp.exp(logit) for logit in logits)
    return tuple(w/sum(weights) for w in weights)


def _mean(mp, model, point, offsets):
    p = _posterior(mp, model, point, offsets)
    return tuple(sum(w*_mp(mp, a[j]) for w, a in zip(p, model.centers, strict=True))
                 for j in range(model.dimension))


def _gradient(mp, model, point, target, offsets):
    mean = _mean(mp, model, point, offsets)
    return tuple((x-a)/_mp(mp, model.time)-_mp(mp, v)
                 for x, a, v in zip(point, mean, target, strict=True))


def _unconstrained_root(mp, model, target, offsets):
    initial = tuple(_mp(mp, model.time*v+model.centers[0][j])
                    for j, v in enumerate(target))
    return tuple(mp.findroot(lambda *point: _gradient(mp, model, point, target, offsets), initial))


def _contains(mp, bounds, point):
    assert all(_mp(mp, lo) <= x <= _mp(mp, hi)
               for (lo, hi), x in zip(bounds, point, strict=True))


def _check_drift(mp, region, model, target, old, new, offsets):
    delta = mp.matrix(new)-mp.matrix(old)
    assert sum(x*x for x in delta) <= _mp(mp, region.displacement_squared_bound)
    assert mp.norm(delta) <= _mp(mp, region.displacement_bound)
    old_mean = _mean(mp, model, old, (0,)*model.mode_count)
    new_mean = _mean(mp, model, new, offsets)
    # Targets are fixed, so projection-input displacement equals mean displacement.
    assert sum((a-b)**2 for a, b in zip(new_mean, old_mean, strict=True)) <= _mp(mp, region.displacement_squared_bound)
    same_point_old = _posterior(mp, model, new, (0,)*model.mode_count)
    same_point_new = _posterior(mp, model, new, offsets)
    tv = sum(abs(a-b) for a, b in zip(same_point_new, same_point_old, strict=True))/2
    assert tv <= _mp(mp, region.total_variation_bound)
    _contains(mp, region.unique.domain.physical_bounds, new)


def test_nonconvex_finite_bias_roots_and_physical_covariance_transfer():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
    unique = _case(model)
    region = certify_gaussian_entropy_bias_region(unique, 1)
    assert region is not None and model.contraction_bound == 4
    changes = ((0, 1, 0), (0, 0, 1), (1, 0, 0), (F(-1, 2), F(1, 2), 0))
    with mp.workdps(90):
        base = _unconstrained_root(mp, model, (0, 0), (0, 0, 0))
        for offsets in changes:
            root = _unconstrained_root(mp, model, (0, 0), offsets)
            _check_drift(mp, region, model, (0, 0), base, root, offsets)
            for x, y in product((0, 2, 4), repeat=2):
                p = _posterior(mp, model, (x, y), offsets)
                mean = _mean(mp, model, (x, y), offsets)
                second = sum(w*sum(_mp(mp, a-c)**2 for a, c in zip(center, unique.reference, strict=True))
                             for w, center in zip(p, model.centers, strict=True))
                covariance = mp.matrix([[sum(w*(_mp(mp, a[i])-mean[i])*(_mp(mp, a[j])-mean[j])
                                             for w, a in zip(p, model.centers, strict=True))
                                         for j in range(2)] for i in range(2)])
                maximum = max(mp.eigsy(covariance, eigvals_only=True))
                assert maximum <= second <= _mp(mp, region.variance_upper_bound)
                curvature = 1-maximum/2
                assert curvature >= _mp(mp, region.strong_convexity_bound)
        # The global diameter failure is real: this original potential has
        # negative physical curvature outside its invariant box [0,4]^2.
        hxx = mp.diff(lambda x: _gradient(mp, model, (x, 0), (0, 0), (0, 0, 0))[0], 6)
        assert hxx < -mp.mpf("0.99")


def test_redundant_active_face_contains_finitely_perturbed_original_roots():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    unique = _case(model, ((1, 0), (2, 0)), (0, 0), (1, 0))
    region = certify_gaussian_entropy_bias_region(unique, F(1, 8))
    assert region is not None
    stable = certify_gaussian_entropy_bias_face(_face(unique), region)
    assert stable is not None and stable.optimizer_bounds[0] == (0, 0)
    assert stable.face.active_indices == (0, 1)
    with mp.workdps(90):
        def root(offsets):
            y = mp.findroot(lambda y: _gradient(mp, model, (0, y), (1, 0), offsets)[1], 0)
            return mp.mpf(0), y
        base = root((0, 0, 0))
        for offsets in ((0, F(1, 8), 0), (0, 0, F(1, 8)), (F(1, 8), 0, 0)):
            point = root(offsets)
            _contains(mp, stable.optimizer_bounds, point)
            _check_drift(mp, region, model, (1, 0), base, point, offsets)
            gradient = _gradient(mp, model, point, (1, 0), offsets)
            assert gradient[0] < 0 and abs(gradient[1]) < mp.mpf("1e-80")
            q = tuple(_mp(mp, v)+a for v, a in zip((1, 0), _mean(mp, model, point, offsets), strict=True))
            _contains(mp, stable.projection_input_bounds, q)


def test_oblique_equality_face_and_common_bias_gauge_for_finite_changes():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    unique = _case(model, ((1, 1), (-1, -1), (2, 2)), (0, 0, 0))
    region = certify_gaussian_entropy_bias_region(unique, F(1, 8))
    assert region is not None
    stable = certify_gaussian_entropy_bias_face(_face(unique), region)
    assert stable is not None and stable.face.active_indices == (0, 1, 2)
    common = certify_gaussian_entropy_bias_region(unique, 0)
    assert common is not None and common.displacement_squared_bound == 0
    with mp.workdps(90):
        def root(offsets):
            def tangent(z):
                gx, gy = _gradient(mp, model, (z, -z), (0, 0), offsets)
                return gx-gy
            z = mp.findroot(tangent, 0)
            return z, -z
        base = root((0, 0, 0))
        for offsets in ((0, F(1, 8), 0), (0, 0, F(1, 8)), (F(1, 16), 0, F(-1, 16))):
            point = root(offsets)
            _contains(mp, stable.optimizer_bounds, point)
            _check_drift(mp, region, model, (0, 0), base, point, offsets)
            shifted = tuple(F(value)+10**400 for value in offsets)
            assert root(shifted) == point
        assert root((10**400,)*3) == base


def test_uniqueness_region_does_not_imply_finite_face_stability():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    unique = _case(model, ((1,),), (0,), (F(1, 10),), reference=(0,))
    face = _face(unique)
    small = certify_gaussian_entropy_bias_region(unique, F(1, 100))
    large = certify_gaussian_entropy_bias_region(unique, 1)
    assert small is not None and large is not None
    assert certify_gaussian_entropy_bias_face(face, small) is not None
    assert certify_gaussian_entropy_bias_face(face, large) is None
    with mp.workdps(90):
        # A finite tilt toward the negative center moves the optimizer off
        # the original active endpoint even though uniqueness persists.
        wanted = mp.mpf(1)/10
        root = mp.findroot(lambda x: x-mp.tanh(x/2-mp.mpf(1)/2)-wanted, -1)
        assert root < -mp.mpf("0.1")
        assert abs(_gradient(mp, model, (root,), (F(1, 10),), (1, 0))[0]) < mp.mpf("1e-80")
        _check_drift(mp, large, model, (F(1, 10),), (mp.mpf(0),), (root,), (1, 0))


def test_huge_finite_bias_region_and_underflow_refusal_are_distinct():
    # Fixed-radius reference makes the first mixture globally robust to
    # every finite bias change, including a large unresolved exponential.
    model = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)))
    unique = _case(model, reference=(0,))
    huge = certify_gaussian_entropy_bias_region(unique, 10**400)
    assert huge is not None and huge.variance_upper_bound == 1
    assert huge.uniqueness_margin == 1
    assert huge.total_variation_bound == 1
    # Here D=2nu*s. The mathematical finite-omega margin is positive,
    # but a zero lower exponential bound cannot prove it numerically.
    borderline = _case(GaussianHeatMap(((0,), (2,)), (0, -8), viscosity=2))
    assert max(borderline.squared_radii) == 4
    assert certify_gaussian_entropy_bias_region(borderline, 10000) is None
    # Coincident centers have no mixture-dependent spatial effect at all.
    coincident = _case(GaussianHeatMap(((3, -2),)*2, (0, -10)), target=(F(1, 7), F(-1, 3)))
    exact = certify_gaussian_entropy_bias_region(coincident, 10**400)
    assert exact is not None and exact.max_squared_radius == 0
    assert exact.variance_upper_bound == exact.displacement_squared_bound == 0


def test_nonunit_age_and_viscosity_displacement_uses_base_contraction():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (0, F(1, 3), F(-2, 5)),
                            viscosity=F(3, 2), time=F(5, 3))
    target = (F(1, 5), F(-1, 7))
    unique = _case(model, target=target)
    region = certify_gaussian_entropy_bias_region(unique, 2)
    assert region is not None
    with mp.workdps(90):
        base = _unconstrained_root(mp, model, target, (0, 0, 0))
        for offsets in ((0, 2, 0), (F(-1, 2), F(1, 2), F(3, 2))):
            point = _unconstrained_root(mp, model, target, offsets)
            _check_drift(mp, region, model, target, base, point, offsets)
        diameter_squared = mp.mpf(2)
        rho0 = _mp(mp, unique.variance_upper_bound/(2*model.viscosity*model.time))
        sharp_drift_squared = diameter_squared*mp.tanh(mp.mpf(1)/2)**2/(1-rho0)**2
        assert _mp(mp, region.displacement_squared_bound) >= sharp_drift_squared
