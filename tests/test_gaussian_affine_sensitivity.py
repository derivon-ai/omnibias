# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""True constrained target VJPs from untrusted forward/autodiff proposals.

The high-precision oracle solves the original Gaussian KKT equations and
differentiates their original ambient Hessian. It never uses the transformed
mixture, its interval action, or the finite approximation's derivative.
"""

import math
from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_sensitivity import (
    certify_gaussian_affine_target_vjp,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.jax.gaussian_constrained_approximation import (
    gaussian_heat_inverse_constrained_approximation as jax_approximation,
)
from omnibias.torch.gaussian_constrained_approximation import (
    gaussian_heat_inverse_constrained_approximation as torch_approximation,
)
from test_gaussian_affine_constraints import _kkt_root, _mp, _velocity

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _family(index=0):
    if index == 0:
        source = GaussianHeatMap(((-1, -2), (1, 2), (2, -1)), (0, F(1, 3), F(-1, 4)), time=4)
        return source, ((1, 2),), (1,), (F(1, 7), F(-1, 11))
    source = GaussianHeatMap(((0, 0, 0), (1, 0, 2), (-1, 2, 1), (2, -1, 0)),
                             (0, F(1, 3), F(-1, 4), F(2, 5)), viscosity=F(3, 2), time=8)
    return source, ((1, 2, -1),), (2,), (F(1, 7), F(-1, 11), F(1, 13))


def _project(restriction, vector):
    return tuple(sum((p*F(value) for p, value in zip(row, vector, strict=True)), F(0))
                 for row in restriction.projector)


def _backend_proposal(backend, source, constraints, rhs, target, upstream, order):
    data = (target, source.centers, source.biases, constraints, rhs, source.viscosity, source.time)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in data)
    if backend == "torch":
        query = torch.tensor(arrays[0], dtype=torch.float64, requires_grad=True)
        parameters = tuple(torch.tensor(value, dtype=torch.float64) for value in arrays[1:])
        result = torch_approximation(query, *parameters, order=order)
        assert bool(result.valid)
        derivative, = torch.autograd.grad(torch.dot(result.value, torch.tensor(upstream, dtype=torch.float64)), query)
        return tuple(result.value.detach().tolist()), tuple(derivative.detach().tolist())
    parameters_j = tuple(jnp.asarray(value) for value in arrays[1:])

    def objective(query):
        result = jax_approximation(query, *parameters_j, order=order)
        return jnp.dot(result.value, jnp.asarray(upstream)), result

    (_, result), derivative = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.asarray(arrays[0]))
    assert bool(result.valid)
    return tuple(np.asarray(result.value).tolist()), tuple(np.asarray(derivative).tolist())


def _original_kkt_vjp(mp, source, constraints, rhs, target, upstream):
    """Differentiate original stationarity, with exact source metadata."""
    point = _kkt_root(mp, source, constraints, rhs, target)
    dimension, rows = source.dimension, len(constraints)
    age, nu = _mp(mp, source.time), _mp(mp, source.viscosity)
    centers = tuple(tuple(_mp(mp, value) for value in row) for row in source.centers)
    scores = tuple(_mp(mp, bias)-sum((x-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
                   for bias, row in zip(source.biases, centers, strict=True))
    weights = tuple(mp.exp(score-max(scores)) for score in scores)
    posterior = tuple(weight/sum(weights) for weight in weights)
    mean = tuple(sum(p*row[i] for p, row in zip(posterior, centers, strict=True)) for i in range(dimension))
    covariance = tuple(tuple(sum(p*(row[i]-mean[i])*(row[j]-mean[j])
                                for p, row in zip(posterior, centers, strict=True))
                            for j in range(dimension)) for i in range(dimension))
    kkt = mp.matrix(dimension+rows)
    for i in range(dimension):
        for j in range(dimension):
            kkt[i, j] = ((1 if i == j else 0)-covariance[i][j]/(2*nu*age))/age
        for j, row in enumerate(constraints):
            kkt[i, dimension+j] = kkt[dimension+j, i] = _mp(mp, row[i])
    solved = mp.lu_solve(kkt, mp.matrix([_mp(mp, value) for value in upstream]+[mp.mpf(0)]*rows))
    return tuple(solved[:dimension])


def _assert_true_derivative_enclosed(mp, certificate, expected):
    assert certificate is not None
    for value, interval in zip(expected, certificate.enclosure, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    center_error = mp.sqrt(sum((_mp(mp, center)-value)**2
                               for center, value in zip(certificate.value, expected, strict=True)))
    assert center_error <= _mp(mp, certificate.error_bound)
    for row in certificate.restriction.constraints:
        assert sum(a*x for a, x in zip(row, certificate.value, strict=True)) == 0
    assert certificate.error_bound == (certificate.base_error_bound+certificate.upstream_rounding_error_bound
                                       +certificate.proposal_rounding_error_bound)
    return center_error


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("index", [0, 1], ids=("oblique_2d", "coupled_3d"))
@pytest.mark.parametrize("order", [0, 1, 8])
def test_finite_approximation_ad_proposals_certify_true_original_kkt_vjp(backend, index, order):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target = _family(index)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    upstream = (0.75, -0.5, 0.25)[:source.dimension]
    point, proposal = _backend_proposal(backend, source, constraints, rhs, target, upstream, order)
    scaled_point = tuple(x/float(source.time) for x in point)
    certificate = certify_gaussian_affine_target_vjp(restriction, target, scaled_point, upstream, proposal)
    assert certificate is not None and certificate.base_certificate is not None
    assert certificate.projected_proposal == certificate.value == _project(restriction, proposal)
    with mp.workdps(100):
        expected = _original_kkt_vjp(mp, source, constraints, rhs, target, upstream)
        error = _assert_true_derivative_enclosed(mp, certificate, expected)
        # The low-order AD value is deliberately not the true derivative;
        # containment must result from residual certification, not equality.
        if order == 0:
            assert error > mp.mpf("1e-3")


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_projected_convex_certificate_accepts_inexact_ad_in_nonconvex_ambient_model(backend):
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)))
    constraints, rhs, target, upstream = ((0, 1),), (1,), (0, 0), (1.0, 0.0)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert source.contraction_bound == F(5, 2) and restriction.model.contraction_bound == F(1, 2)
    point, proposal = _backend_proposal(backend, source, constraints, rhs, target, upstream, 1)
    np.testing.assert_allclose(point, (0, 1), rtol=0, atol=1e-14)
    np.testing.assert_allclose(proposal, (1.75, 0), rtol=0, atol=1e-14)
    certificate = certify_gaussian_affine_target_vjp(restriction, target, point, upstream, proposal)
    assert certificate is not None and certificate.enclosure[0].contains(2)
    assert certificate.enclosure[1].lo == certificate.enclosure[1].hi == 0
    assert certificate.error_bound >= F(1, 4)
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        expected = _original_kkt_vjp(mp, source, constraints, rhs, target, upstream)
        assert max(abs(value-known) for value, known in zip(expected, (2, 0), strict=True)) < mp.mpf("1e-75")
        # Differentiate the original ambient velocity as a second independent
        # check of its negative normal and positive tangent curvatures.
        assert abs(mp.diff(lambda y: _velocity(mp, source, (mp.mpf(0), y), mp.mpf(1))[1], 1)+1) < mp.mpf("1e-75")
        assert abs(mp.diff(lambda x: _velocity(mp, source, (x, mp.mpf(1)), mp.mpf(1))[0], 0)-mp.mpf("0.5")) < mp.mpf("1e-75")


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_exact_projection_retains_small_tangent_part_of_huge_normal_upstream(backend):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target = _family()
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    upstream = (float(2**53), float(2**54+4))
    projected = _project(restriction, upstream)
    assert projected == (F(-8, 5), F(4, 5))
    naive = np.asarray(restriction.projector, dtype=np.float64) @ np.asarray(upstream)
    assert not np.allclose(naive, np.asarray(projected, dtype=np.float64), rtol=0, atol=0.1)
    # Exact projection precedes the backend AD proposal. The certificate
    # remains bound to the original, large declared upstream vector.
    point, proposal = _backend_proposal(backend, source, constraints, rhs, target, tuple(map(float, projected)), 8)
    proposal = (proposal[0], math.nextafter(proposal[1], math.inf))
    certificate = certify_gaussian_affine_target_vjp(
        restriction, target, tuple(x/float(source.time) for x in point), upstream, proposal,
    )
    assert certificate is not None and certificate.projected_upstream == projected
    assert certificate.upstream_rounding_error_bound > 0
    assert certificate.proposal_rounding_error_bound > 0
    with mp.workdps(110):
        expected = _original_kkt_vjp(mp, source, constraints, rhs, target, upstream)
        _assert_true_derivative_enclosed(mp, certificate, expected)


def test_bad_forward_and_adjoint_proposals_widen_a_valid_true_derivative_bound():
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target = _family()
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    upstream = (0.75, -0.5)
    point, proposal = _backend_proposal("torch", source, constraints, rhs, target, upstream, 8)
    good = certify_gaussian_affine_target_vjp(restriction, target, tuple(x/4 for x in point), upstream, proposal)
    coarse = certify_gaussian_affine_target_vjp(restriction, target, (0.5, -0.5), upstream, (0.0, 0.0))
    assert good is not None and coarse is not None and coarse.error_bound > good.error_bound
    assert coarse.base_certificate.root_certificate.error_bound > good.base_certificate.root_certificate.error_bound
    with mp.workdps(100):
        expected = _original_kkt_vjp(mp, source, constraints, rhs, target, upstream)
        _assert_true_derivative_enclosed(mp, good, expected)
        _assert_true_derivative_enclosed(mp, coarse, expected)


def test_exact_normal_upstream_shortcut_ignores_bad_proposals():
    source, constraints, rhs, target = _family()
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    certificate = certify_gaussian_affine_target_vjp(
        restriction, target, (128.0, -64.0), (float(2**53), float(2**54)), (37.0, -11.0),
    )
    assert certificate is not None and certificate.base_certificate is None
    assert certificate.projected_upstream == certificate.value == (F(0), F(0))
    assert certificate.projected_proposal != certificate.value
    assert certificate.error_bound == 0 and certificate.coordinate_error_bounds == (0, 0)
    assert all(interval.lo == interval.hi == 0 for interval in certificate.enclosure)


def test_coincident_tangent_centers_have_exact_affine_target_vjp():
    source = GaussianHeatMap(((0, 0), (1, 2), (-2, -4)), (0, 1, -2), time=3)
    restriction = restrict_gaussian_heat(source, ((1, 2),), (1,))
    assert restriction.model.contraction_bound == 0
    upstream = (0.75, -0.5)
    certificate = certify_gaussian_affine_target_vjp(
        restriction, (F(1, 7), F(-1, 11)), (100.0, -20.0), upstream, (-77.0, 12.0),
    )
    assert certificate is not None and certificate.base_certificate is None
    assert certificate.value == tuple(3*value for value in _project(restriction, upstream))
    assert certificate.error_bound == 0
    for value, interval in zip(certificate.value, certificate.enclosure, strict=True):
        assert F(interval.lo) <= value <= F(interval.hi)
