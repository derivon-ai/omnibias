# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original constrained Gaussian parameter VJPs against independent KKT AD.

The mpmath oracle differentiates the original KKT residual with respect to
each scalar input. Backend finite formulas provide only untrusted root and
target-adjoint proposals; their parameter gradients are not the claim.
"""

from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_adjoint import compile_gaussian_affine_adjoint
from omnibias.core.verified.gaussian_affine_parameters import (
    certify_gaussian_affine_parameter_vjp,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.jax.gaussian_constrained import gaussian_heat_inverse_constrained as jax_inverse
from omnibias.torch.gaussian_constrained import gaussian_heat_inverse_constrained as torch_inverse
from test_gaussian_affine_constraints import _mp
from test_gaussian_affine_sensitivity import _backend_proposal

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _family(index=0):
    if index == 0:
        source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)))
        return source, ((0, 1),), (1,), (0, 0), (1.0, 3.0)
    source = GaussianHeatMap(((0, 0), (1, 2), (3, -1)), (F(1, 8), F(-1, 4), F(3, 8)),
                             viscosity=F(5, 4), time=8)
    return source, ((1, 2),), (F(3, 4),), (F(1, 4), F(-1, 8)), (0.75, -0.5)


def _data(source, constraints, rhs, target):
    return target, source.centers, source.biases, constraints, rhs, source.viscosity, source.time


def _flatten(data):
    arrays = tuple(np.asarray(value, dtype=object) for value in data)
    return tuple(value for array in arrays for value in array.reshape(-1)), tuple(array.shape for array in arrays)


def _unpack(flat, shapes):
    result, offset = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        values = flat[offset:offset+size]
        if not shape:
            result.append(values[0])
        elif len(shape) == 1:
            result.append(tuple(values))
        else:
            result.append(tuple(tuple(values[i*shape[1]:(i+1)*shape[1]]) for i in range(shape[0])))
        offset += size
    assert offset == len(flat)
    return tuple(result)


def _replace(values, index, value):
    return (*values[:index], value, *values[index+1:])


def _residual(mp, state, flat, shapes):
    target, centers, biases, constraints, rhs, nu, age = _unpack(flat, shapes)
    dimension, rows = len(target), len(constraints)
    point, multipliers = state[:dimension], state[dimension:]
    scores = tuple(bias-sum((x-a)**2 for x, a in zip(point, center, strict=True))/(4*nu*age)
                   for bias, center in zip(biases, centers, strict=True))
    weights = tuple(mp.exp(score-max(scores)) for score in scores)
    mean = tuple(sum(weight*center[i] for weight, center in zip(weights, centers, strict=True))/sum(weights)
                 for i in range(dimension))
    stationarity = tuple((point[i]-mean[i])/age-target[i]
                         +sum(constraints[j][i]*multipliers[j] for j in range(rows)) for i in range(dimension))
    feasibility = tuple(sum(a*x for a, x in zip(row, point, strict=True))-value
                        for row, value in zip(constraints, rhs, strict=True))
    return stationarity+feasibility


def _original_root(mp, flat, shapes):
    target, _, _, constraints, rhs, _, _ = _unpack(flat, shapes)
    matrix, right = mp.matrix(constraints), mp.matrix(rhs)
    anchor = matrix.T*mp.lu_solve(matrix*matrix.T, right)
    initial = tuple(anchor)+(mp.mpf(0),)*len(constraints)
    solved = mp.findroot(lambda *state: _residual(mp, state, flat, shapes), initial, tol=mp.eps*32, maxsteps=100)
    assert max(abs(value) for value in _residual(mp, solved, flat, shapes)) < mp.sqrt(mp.eps)
    return tuple(solved), len(target)


def _original_parameter_vjp(mp, data, upstream):
    """KKT implicit differentiation, without any closed-form VJP kernel."""
    flat, shapes = _flatten(data)
    exact = tuple(_mp(mp, value) for value in flat)
    state, dimension = _original_root(mp, exact, shapes)
    jacobian = mp.matrix([[mp.diff(lambda value, i=i, j=j: _residual(mp, _replace(state, j, value), exact, shapes)[i], state[j])
                           for j in range(len(state))] for i in range(len(state))])
    rhs = mp.matrix([_mp(mp, value) for value in upstream]+[mp.mpf(0)]*(len(state)-dimension))
    adjoint = mp.lu_solve(jacobian.T, rhs)
    derivatives = tuple(-sum(adjoint[i]*mp.diff(
        lambda value, i=i, j=j: _residual(mp, state, _replace(exact, j, value), shapes)[i], exact[j],
    ) for i in range(len(state))) for j in range(len(exact)))
    return _unpack(derivatives, shapes), state[:dimension]


def _blocks(certificate):
    return (certificate.target_gradient, certificate.center_gradient, certificate.bias_gradient,
            certificate.constraint_gradient, certificate.rhs_gradient, certificate.viscosity_gradient,
            certificate.time_gradient)


def _assert_enclosed(mp, certificate, expected):
    assert certificate is not None
    actual_flat, shapes = _flatten(_blocks(certificate))
    expected_flat, expected_shapes = _flatten(expected)
    assert shapes == expected_shapes
    for interval, value in zip(actual_flat, expected_flat, strict=True):
        assert _mp(mp, interval.lo) <= _mp(mp, value) <= _mp(mp, interval.hi)


def _backend_true_gradients(backend, data, upstream):
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in data)
    if backend == "torch":
        parameters = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in arrays)
        result = torch_inverse(*parameters, tolerance=1e-12, feasibility_tolerance=1e-12)
        assert bool(result.converged)
        gradients = torch.autograd.grad(torch.dot(result.value, torch.tensor(upstream, dtype=torch.float64)), parameters)
        return tuple(value.detach().numpy() for value in gradients)

    def objective(*parameters):
        result = jax_inverse(*parameters, tolerance=1e-12, feasibility_tolerance=1e-12)
        return jnp.dot(result.value, jnp.asarray(upstream)), result.converged

    (_, converged), gradients = jax.jit(jax.value_and_grad(objective, argnums=tuple(range(7)), has_aux=True))(
        *(jnp.asarray(value) for value in arrays),
    )
    assert bool(converged)
    return tuple(np.asarray(value) for value in gradients)


def _geometry(source, constraints, rhs):
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction is not None
    geometry = compile_gaussian_affine_adjoint(restriction)
    assert geometry is not None
    return geometry


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_original_nonconvex_fixture_all_seven_parameter_blocks(backend):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target, upstream = _family()
    geometry = _geometry(source, constraints, rhs)
    point, proposal = _backend_proposal(backend, source, constraints, rhs, target, upstream, 1)
    certificate = certify_gaussian_affine_parameter_vjp(geometry, target, point, upstream, proposal)
    expected = ((2, 0), ((F(1, 2), F(-3, 2)), (F(1, 2), F(-1, 2))), (-1, 1),
                ((2, -5),), (5,), -2, -2)
    assert source.contraction_bound > 1
    np.testing.assert_allclose(proposal, (1.75, 0), rtol=0, atol=1e-14)
    with mp.workdps(100):
        independent, root = _original_parameter_vjp(mp, _data(source, constraints, rhs, target), upstream)
        independent_flat, _ = _flatten(independent)
        exact_flat, _ = _flatten(expected)
        assert max(abs(value-_mp(mp, known)) for value, known in zip(independent_flat, exact_flat, strict=True)) < mp.mpf("1e-80")
        assert max(abs(value-known) for value, known in zip(root, (0, 1), strict=True)) < mp.mpf("1e-80")
        _assert_enclosed(mp, certificate, expected)
    gradients = _backend_true_gradients(backend, _data(source, constraints, rhs, target), upstream)
    for actual, known in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, np.asarray(known, dtype=np.float64), rtol=1e-10, atol=2e-11)


@pytest.fixture(scope="module")
def _oblique_oracle():
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target, upstream = _family(1)
    with mp.workdps(100):
        return _original_parameter_vjp(mp, _data(source, constraints, rhs, target), upstream)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("order", [0, 1, 8])
def test_oblique_all_parameter_certificates_from_forward_formula_proposals(backend, order, _oblique_oracle):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target, upstream = _family(1)
    geometry = _geometry(source, constraints, rhs)
    point, proposal = _backend_proposal(backend, source, constraints, rhs, target, upstream, order)
    certificate = certify_gaussian_affine_parameter_vjp(
        geometry, target, tuple(x/float(source.time) for x in point), upstream, proposal,
    )
    expected, root = _oblique_oracle
    with mp.workdps(100):
        _assert_enclosed(mp, certificate, expected)
        for interval, value in zip(certificate.root_enclosure, root, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    if order == 8:
        gradients = _backend_true_gradients(backend, _data(source, constraints, rhs, target), upstream)
        for actual, known in zip(gradients, expected, strict=True):
            np.testing.assert_allclose(actual, np.asarray(known, dtype=np.float64), rtol=2e-9, atol=2e-10)


def test_parameter_oracle_matches_direct_original_root_differentiation():
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target, upstream = _family(1)
    data = _data(source, constraints, rhs, target)
    flat, shapes = _flatten(data)
    with mp.workdps(90):
        exact = tuple(_mp(mp, value) for value in flat)
        expected, _ = _original_parameter_vjp(mp, data, upstream)
        wanted, _ = _flatten(expected)
        # Include one center coordinate, one constraint coordinate, rhs,
        # viscosity and age; each perturbation solves a fresh original KKT.
        for index in (2, 11, 13, 14, 15):
            def objective(value, index=index):
                root, dimension = _original_root(mp, _replace(exact, index, value), shapes)
                return sum(_mp(mp, g)*x for g, x in zip(upstream, root[:dimension], strict=True))
            actual = mp.diff(objective, exact[index])
            assert abs(actual-wanted[index]) < mp.mpf("1e-65")


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_normal_upstream_preserves_constraint_and_rhs_gradients(backend):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target, _ = _family()
    upstream = (0.0, 3.0)
    geometry = _geometry(source, constraints, rhs)
    # Neither root nor adjoint proposal is trusted. A zero target sensitivity
    # does not make the derivative with respect to the feasible plane zero.
    certificate = certify_gaussian_affine_parameter_vjp(geometry, target, (0.125, 0.875), upstream, (23.0, -17.0))
    expected = ((0, 0), ((0, 0), (0, 0)), (0, 0), ((0, -3),), (3,), 0, 0)
    assert certificate is not None and certificate.target_certificate.base_certificate is None
    assert certificate.root_certificate is not None
    assert certificate.primal_multiplier is None and certificate.posterior == ()
    with mp.workdps(90):
        _assert_enclosed(mp, certificate, expected)
    gradients = _backend_true_gradients(backend, _data(source, constraints, rhs, target), upstream)
    for actual, known in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, np.asarray(known, dtype=np.float64), rtol=1e-10, atol=2e-11)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_full_rank_constraints_have_only_constraint_and_rhs_sensitivities(backend):
    mp = pytest.importorskip("mpmath")
    source, _, _, target, _ = _family(1)
    constraints, rhs, upstream = ((1, 2), (2, -1)), (1, -2), (0.75, -0.5)
    geometry = _geometry(source, constraints, rhs)
    certificate = certify_gaussian_affine_parameter_vjp(geometry, target, (17.0, -29.0), upstream, (-3.0, 11.0))
    # x=C^-1 rhs=(-3/5,4/5); lambda=C^-T g=(-1/20,2/5).
    expected = ((0, 0), ((0, 0),)*3, (0, 0, 0),
                ((F(-3, 100), F(1, 25)), (F(6, 25), F(-8, 25))), (F(-1, 20), F(2, 5)), 0, 0)
    assert certificate is not None and certificate.target_certificate.base_certificate is None
    assert certificate.root_certificate is None
    with mp.workdps(90):
        _assert_enclosed(mp, certificate, expected)
        for interval, value in zip(certificate.root_enclosure, (F(-3, 5), F(4, 5)), strict=True):
            assert _mp(mp, interval.lo) <= _mp(mp, value) <= _mp(mp, interval.hi)
    gradients = _backend_true_gradients(backend, _data(source, constraints, rhs, target), upstream)
    for actual, known in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, np.asarray(known, dtype=np.float64), rtol=1e-10, atol=2e-11)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_affine_restricted_map_still_has_original_center_and_constraint_gradients(backend):
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((0, 0), (1, 2), (-2, -4)), (0, 1, -2), time=3)
    constraints, rhs, target = ((1, 2),), (1,), (F(1, 4), F(-1, 8))
    upstream = (0.75, -0.5)
    geometry = _geometry(source, constraints, rhs)
    assert geometry.restriction.model.contraction_bound == 0
    certificate = certify_gaussian_affine_parameter_vjp(geometry, target, (100.0, -23.0), upstream, (47.0, -9.0))
    assert certificate is not None and certificate.root_certificate is None
    assert len(certificate.posterior) == source.mode_count
    with mp.workdps(100):
        expected, root = _original_parameter_vjp(mp, _data(source, constraints, rhs, target), upstream)
        # These zeros follow from coincident projected centers exactly; the
        # high-precision KKT differentiation can leave sub-precision noise.
        assert max(abs(value) for value in expected[2]) < mp.mpf("1e-80")
        assert abs(expected[5]) < mp.mpf("1e-80")
        expected = (*expected[:2], (0, 0, 0), *expected[3:5], 0, expected[6])
        _assert_enclosed(mp, certificate, expected)
        assert abs(expected[1][0][0]) > mp.mpf("0.1")
        for interval, value in zip(certificate.root_enclosure, root, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    gradients = _backend_true_gradients(backend, _data(source, constraints, rhs, target), upstream)
    for actual, known in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, np.asarray(known, dtype=np.float64), rtol=1e-10, atol=2e-11)
