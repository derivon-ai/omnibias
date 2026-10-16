# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original nonlinear decisions under finite target and log-weight changes."""

from fractions import Fraction as F
from itertools import product

import pytest
from omnibias.core.verified.gaussian_entropy_bias_region import certify_gaussian_entropy_bias_region
from omnibias.core.verified.gaussian_entropy_face import certify_gaussian_entropy_face
from omnibias.core.verified.gaussian_entropy_joint_face import certify_gaussian_entropy_joint_face
from omnibias.core.verified.gaussian_entropy_joint_region import (
    certify_gaussian_entropy_joint_region,
)
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
    return tuple((x-a)/_mp(mp, model.time)-_mp(mp, v)
                 for x, a, v in zip(point, _mean(mp, model, point, offsets), target, strict=True))


def _root(mp, model, target, offsets):
    initial = tuple(_mp(mp, model.time*v+model.centers[0][j]) for j, v in enumerate(target))
    return tuple(mp.findroot(lambda *point: _gradient(mp, model, point, target, offsets), initial))


def _contains(mp, bounds, point):
    assert all(_mp(mp, lo) <= x <= _mp(mp, hi)
               for (lo, hi), x in zip(bounds, point, strict=True))


def _check_drift(mp, region, old, new, target_change, offsets):
    model, base_target = region.unique.domain.model, region.unique.domain.target
    dx2 = sum((x-y)**2 for x, y in zip(new, old, strict=True))
    old_mean = _mean(mp, model, old, (0,)*model.mode_count)
    new_mean = _mean(mp, model, new, offsets)
    dq = tuple(_mp(mp, model.time*dv)+a-b
               for dv, a, b in zip(target_change, new_mean, old_mean, strict=True))
    assert dx2 <= _mp(mp, region.displacement_squared_bound)
    assert sum(x*x for x in dq) <= _mp(mp, region.displacement_squared_bound)
    assert mp.sqrt(dx2) <= _mp(mp, region.displacement_bound)
    distance2 = sum(max(_mp(mp, lo)-x, mp.mpf(0), x-_mp(mp, hi))**2
                    for x, (lo, hi) in zip(new, region.unique.domain.physical_bounds, strict=True))
    assert distance2 <= _mp(mp, region.expansion_radius)**2
    _contains(mp, region.expanded_coordinate_bounds, new)
    assert region.contains_parameters(target_change, offsets)
    return tuple(_mp(mp, model.time*(v+dv))+a
                 for v, dv, a in zip(base_target, target_change, new_mean, strict=True))


def test_nonconvex_joint_region_roots_leave_old_box_and_covariance_stays_bounded():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
    unique = _case(model)
    region = certify_gaussian_entropy_joint_region(unique, F(1, 2), F(1, 10))
    assert region is not None and model.contraction_bound == 4
    stable = certify_gaussian_entropy_joint_face(_face(unique), region)
    assert stable is not None and stable.optimizer_bounds[0][0] < 0
    assert region.contains_physical((F(-3, 50), F(-2, 25)))
    assert not region.contains_physical((F(-1, 10), F(-1, 10)))
    assert not region.contains_parameters((F(1, 10), F(1, 10)), (0, 0, 0))
    assert not region.contains_parameters((0, 0), (0, 1, 0))
    cases = (((F(-1, 10), 0), (0, F(1, 2), 0)),
             ((0, F(-1, 10)), (0, 0, F(1, 2))),
             ((F(3, 50), F(2, 25)), (F(-1, 4), F(1, 4), 0)))
    with mp.workdps(90):
        old = _root(mp, model, (0, 0), (0, 0, 0))
        for dv, db in cases:
            new = _root(mp, model, dv, db)
            _check_drift(mp, region, old, new, dv, db)
            _contains(mp, stable.optimizer_bounds, new)
        assert _root(mp, model, cases[0][0], cases[0][1])[0] < 0
        # K plus a Euclidean ball: include exact rational points on its
        # rounded corner, not the unsupported outer-box corners.
        points = tuple(product((0, 2, 4), repeat=2))+((F(-3, 50), F(-2, 25)),
                                                     (4+F(3, 50), 4+F(2, 25)))
        for point in points:
            assert region.contains_physical(point)
            for offsets in ((0, 0, 0), (0, F(1, 2), 0), (F(-1, 4), F(1, 4), 0)):
                x = tuple(_mp(mp, value) for value in point)
                p = _posterior(mp, model, x, offsets)
                mean = _mean(mp, model, x, offsets)
                second = sum(w*sum(_mp(mp, a-c)**2 for a, c in zip(center, unique.reference, strict=True))
                             for w, center in zip(p, model.centers, strict=True))
                covariance = mp.matrix([[sum(w*(_mp(mp, a[i])-mean[i])*(_mp(mp, a[j])-mean[j])
                                             for w, a in zip(p, model.centers, strict=True))
                                         for j in range(2)] for i in range(2)])
                maximum = max(mp.eigsy(covariance, eigvals_only=True))
                bound = region.spatial_variance_upper_bound if not any(offsets) else region.variance_upper_bound
                assert maximum <= second <= _mp(mp, bound)
                assert 1-maximum/2 >= _mp(mp, region.strong_convexity_bound)
        diameter = mp.sqrt(32)
        spatial = diameter/20
        b, d = _mp(mp, unique.variance_upper_bound), mp.mpf(16)
        spatial_rho = d*b/(b+(d-b)*mp.exp(-spatial))/2
        needed = (mp.mpf(1)/10+diameter*mp.tanh(mp.mpf(1)/8))**2/(1-spatial_rho)**2
        assert _mp(mp, region.displacement_squared_bound) >= needed
        # Using rho only on old K would claim a smaller unsupported radius.
        assert region.spatial_contraction_bound > unique.variance_upper_bound/2


def test_redundant_nonconvex_face_survives_finite_joint_changes():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    unique = _case(model, ((1, 0), (2, 0)), (0, 0), (1, 0))
    region = certify_gaussian_entropy_joint_region(unique, F(1, 16), F(1, 100))
    assert region is not None
    stable = certify_gaussian_entropy_joint_face(_face(unique), region)
    assert stable is not None and stable.optimizer_bounds[0] == (0, 0)
    with mp.workdps(90):
        def root(dv, db):
            target = (1+F(dv[0]), F(dv[1]))
            y = mp.findroot(lambda y: _gradient(mp, model, (0, y), target, db)[1], _mp(mp, dv[1]))
            assert _gradient(mp, model, (0, y), target, db)[0] < 0
            return mp.mpf(0), y
        old = root((0, 0), (0, 0, 0))
        for dv, db in (((F(3, 500), F(4, 500)), (0, F(1, 16), 0)),
                       ((F(-1, 100), 0), (0, 0, F(1, 16)))):
            new = root(dv, db)
            _contains(mp, stable.optimizer_bounds, new)
            q = _check_drift(mp, region, old, new, dv, db)
            _contains(mp, stable.projection_input_bounds, q)


def test_oblique_equality_joint_region_and_huge_common_bias():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
    unique = _case(model, ((1, 1), (-1, -1), (2, 2)), (0, 0, 0))
    region = certify_gaussian_entropy_joint_region(unique, F(1, 16), F(1, 100))
    assert region is not None
    stable = certify_gaussian_entropy_joint_face(_face(unique), region)
    assert stable is not None
    with mp.workdps(90):
        def root(dv, db):
            def tangent(z):
                gx, gy = _gradient(mp, model, (z, -z), dv, db)
                return gx-gy
            z = mp.findroot(tangent, 0)
            return z, -z
        old = root((0, 0), (0, 0, 0))
        dv, db = (F(3, 500), F(4, 500)), (0, F(1, 16), 0)
        new = root(dv, db)
        _contains(mp, stable.optimizer_bounds, new)
        q = _check_drift(mp, region, old, new, dv, db)
        _contains(mp, stable.projection_input_bounds, q)
        shifted = tuple(F(value)+10**400 for value in db)
        assert region.contains_parameters(dv, shifted)
        assert root(dv, shifted) == new


def test_normal_target_motion_keeps_optimizer_fixed_but_moves_projection_input():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    unique = _case(model, ((1,),), (0,), (F(1, 4),), reference=(0,))
    region = certify_gaussian_entropy_joint_region(unique, F(1, 10), F(1, 20))
    assert region is not None
    stable = certify_gaussian_entropy_joint_face(_face(unique), region)
    assert stable is not None and stable.optimizer_bounds == ((0, 0),)
    with mp.workdps(90):
        dv, db = (F(1, 20),), (0, F(1, 10))
        new = old = (mp.mpf(0),)
        q = _check_drift(mp, region, old, new, dv, db)
        assert q[0] > _mp(mp, F(1, 4))
        _contains(mp, stable.projection_input_bounds, q)
        assert _gradient(mp, model, new, (F(3, 10),), db)[0] < 0


def test_finite_target_switch_refuses_face_but_preserves_uniqueness():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    unique = _case(model, ((1,),), (0,), (F(1, 4),), reference=(0,))
    region = certify_gaussian_entropy_joint_region(unique, 0, F(1, 2))
    assert region is not None
    assert certify_gaussian_entropy_joint_face(_face(unique), region) is None
    with mp.workdps(90):
        new = mp.findroot(lambda x: x-mp.tanh(x/2)+mp.mpf(1)/4, -1)
        assert new < 0
        _check_drift(mp, region, (mp.mpf(0),), (new,), (F(-1, 2),), (0, 0))


def test_nonunit_scale_and_zero_target_radius_recover_bias_only_guarantee():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (0, F(1, 3), F(-2, 5)),
                            viscosity=F(3, 2), time=F(5, 3))
    target = (F(1, 5), F(-1, 7))
    unique = _case(model, target=target)
    region = certify_gaussian_entropy_joint_region(unique, 1, F(1, 5))
    bias_only = certify_gaussian_entropy_bias_region(unique, 1)
    zero_target = certify_gaussian_entropy_joint_region(unique, 1, 0)
    assert region is not None and bias_only is not None and zero_target is not None
    assert region.expansion_radius == F(1, 3)
    assert zero_target.expansion_radius == 0 and zero_target.spatial_oscillation == 0
    assert zero_target.spatial_variance_upper_bound == unique.variance_upper_bound
    assert zero_target.contains_parameters((0, 0), (0, 1, 0))
    assert not zero_target.contains_parameters((F(1, 10**10), 0), (0, 1, 0))
    with mp.workdps(90):
        old = _root(mp, model, target, (0, 0, 0))
        dv, db = (F(3, 25), F(4, 25)), (0, 1, 0)
        new_target = tuple(v+change for v, change in zip(target, dv, strict=True))
        new = _root(mp, model, new_target, db)
        _check_drift(mp, region, old, new, dv, db)
        fixed_target_new = _root(mp, model, target, db)
        _check_drift(mp, zero_target, old, fixed_target_new, (0, 0), db)
        squared = sum((x-y)**2 for x, y in zip(fixed_target_new, old, strict=True))
        assert squared <= _mp(mp, bias_only.displacement_squared_bound)
