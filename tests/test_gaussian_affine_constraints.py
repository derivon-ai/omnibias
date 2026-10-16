# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained Gaussian decisions checked against original-family KKT roots.

The independent oracle uses the original Gaussian centers and biases at each
age. It never evaluates the compiled projected mixture, so it can detect a
missing time derivative of the normal-energy bias correction.
"""

from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_evolution import certify_gaussian_affine_evolution
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import (
    certify_gaussian_affine_inverse,
    inverse_gaussian_affine,
)
from omnibias.jax.gaussian_constrained import gaussian_heat_inverse_constrained as jax_constrained
from omnibias.torch.gaussian_constrained import (
    gaussian_heat_inverse_constrained as torch_constrained,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _mp(mp, value):
    if isinstance(value, (int, float, F)):
        exact = F(value)
        return mp.mpf(exact.numerator)/exact.denominator
    return value  # Preserve high-precision perturbations used by mp.diff.


def _velocity(mp, source, point, age):
    nu = _mp(mp, source.viscosity)
    centers = [tuple(_mp(mp, value) for value in row) for row in source.centers]
    logits = [_mp(mp, bias)-sum((x-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
              for bias, row in zip(source.biases, centers, strict=True)]
    weights = [mp.exp(value-max(logits)) for value in logits]
    mean = tuple(sum(p*row[axis] for p, row in zip(weights, centers, strict=True))/sum(weights)
                 for axis in range(source.dimension))
    return tuple((x-mu)/age for x, mu in zip(point, mean, strict=True))


def _kkt_root(mp, source, constraints, rhs, target, age=None):
    """Solve original stationarity plus Cx=d, with no projected-model calls."""
    age = _mp(mp, source.time if age is None else age)
    matrix = mp.matrix([[_mp(mp, value) for value in row] for row in constraints])
    right, wanted = mp.matrix([_mp(mp, value) for value in rhs]), tuple(_mp(mp, value) for value in target)
    dimension, rows = source.dimension, len(constraints)
    # Feasible initialization from constraint geometry alone, independent of
    # the Gaussian restriction compiler and its transformed biases.
    anchor = matrix.T*mp.lu_solve(matrix*matrix.T, right)

    def residual(*state):
        point, multipliers = state[:dimension], state[dimension:]
        velocity = _velocity(mp, source, point, age)
        stationarity = tuple(velocity[i]-wanted[i]+sum(matrix[j, i]*multipliers[j] for j in range(rows))
                             for i in range(dimension))
        feasibility = tuple(sum(matrix[j, i]*point[i] for i in range(dimension))-right[j] for j in range(rows))
        return stationarity+feasibility

    root = mp.findroot(residual, tuple(anchor)+(mp.mpf(0),)*rows, tol=mp.eps*32, maxsteps=100)
    assert max(abs(value) for value in residual(*root)) < mp.sqrt(mp.eps)
    return tuple(root[:dimension])


def _objective(mp, source, target, point):
    nu, age = _mp(mp, source.viscosity), _mp(mp, source.time)
    logits = [_mp(mp, bias)-sum((x-_mp(mp, a))**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
              for bias, row in zip(source.biases, source.centers, strict=True)]
    shift = max(logits)
    # The omitted age-only normalization cancels in the objective gap.
    return -2*nu*(shift+mp.log(sum(mp.exp(value-shift) for value in logits)))-sum(
        _mp(mp, value)*x for value, x in zip(target, point, strict=True))


def _true_target_jacobian(mp, source, constraints, rhs, target):
    def entry(axis, direction):
        def coordinate(t):
            wanted = tuple(_mp(mp, value)+(t if j == direction else 0) for j, value in enumerate(target))
            return _kkt_root(mp, source, constraints, rhs, wanted)[axis]
        return mp.diff(coordinate, 0)
    return tuple(tuple(entry(i, j) for j in range(source.dimension)) for i in range(source.dimension))


def _family(index):
    if index == 0:
        return (GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4))),
                ((0, 1),), (1,), (0, 0))
    return (GaussianHeatMap(((-1, -2), (1, 2), (2, -1)), (F(9, 4), F(1, 4), F(-1, 3)),
                            viscosity=F(3, 2), time=2),
            ((1, 2),), (3,), (F(2, 7), F(-1, 5)))


@pytest.mark.parametrize("index", [0, 1], ids=("nonconvex_axis", "off_axis_nonzero_target"))
def test_constrained_root_and_original_objective_gap_against_independent_kkt(index):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target = _family(index)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction is not None
    computed = inverse_gaussian_affine(restriction, target, absolute_tolerance=1e-11)
    assert computed.certificate is not None and computed.tolerance_met
    coarse = certify_gaussian_affine_inverse(restriction, target, (0.25, -0.125))
    assert coarse is not None
    with mp.workdps(90):
        root = _kkt_root(mp, source, constraints, rhs, target)
        for certificate in (computed.certificate, coarse):
            point = certificate.feasible_point
            assert tuple(sum(a*x for a, x in zip(row, point, strict=True)) for row in constraints) == rhs
            for expected, interval in zip(root, certificate.enclosure, strict=True):
                assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
            gap = _objective(mp, source, target, tuple(_mp(mp, value) for value in point))-_objective(mp, source, target, root)
            assert -mp.mpf("1e-75") <= gap <= _mp(mp, certificate.objective_gap_bound)
        assert coarse.objective_gap_bound > computed.certificate.objective_gap_bound
        # A normal target shift changes only an additive feasible-set constant.
        shifted = tuple(v+7*a for v, a in zip(target, constraints[0], strict=True))
        root_shifted = _kkt_root(mp, source, constraints, rhs, shifted)
        assert max(abs(a-b) for a, b in zip(root, root_shifted, strict=True)) < mp.mpf("1e-80")


def test_decisive_original_time_law_and_constraint_offset_derivative():
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target = _family(0)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    evolution = certify_gaussian_affine_evolution(restriction, (Interval.point(0.0),)*2)
    assert evolution is not None
    assert source.contraction_bound == F(5, 2) and restriction.model.contraction_bound == F(1, 2)
    with mp.workdps(90):
        root = _kkt_root(mp, source, constraints, rhs, target)
        assert max(abs(a-b) for a, b in zip(root, (0, 1), strict=True)) < mp.mpf("1e-80")
        age_derivative = tuple(mp.diff(lambda s, axis=axis: _kkt_root(mp, source, constraints, rhs, target, s)[axis], 1)
                               for axis in range(2))
        rhs_derivative = tuple(mp.diff(lambda d, axis=axis: _kkt_root(mp, source, constraints, (d,), target)[axis], 1)
                               for axis in range(2))
        for actual, expected in zip(age_derivative, (-2, 0), strict=True):
            assert abs(actual-expected) < mp.mpf("1e-70")
        for actual, expected in zip(rhs_derivative, (2, 1), strict=True):
            assert abs(actual-expected) < mp.mpf("1e-70")
        for value, interval in zip(age_derivative, evolution.time_derivative, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        assert not evolution.base_evolution.time_derivative[0].contains(-2)
        assert evolution.normal_forcing[0].contains(-2)
        assert evolution.diffusion[0].contains(0)
        # Original ambient Hessian, differentiated directly from its velocity.
        normal_curvature = mp.diff(lambda y: _velocity(mp, source, (mp.mpf(0), y), mp.mpf(1))[1], 1)
        tangent_curvature = mp.diff(lambda x: _velocity(mp, source, (x, mp.mpf(1)), mp.mpf(1))[0], 0)
        assert abs(normal_curvature+1) < mp.mpf("1e-80")
        assert abs(tangent_curvature-mp.mpf("0.5")) < mp.mpf("1e-80")


@pytest.mark.parametrize("index", [0, 1])
def test_target_jacobian_and_original_age_derivative_fit_uniform_certificates(index):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target = _family(index)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    box = tuple(Interval.from_rational(F(value)) for value in target)
    evolution = certify_gaussian_affine_evolution(restriction, box, absolute_tolerance=1e-13)
    assert evolution is not None
    with mp.workdps(90):
        jacobian = _true_target_jacobian(mp, source, constraints, rhs, target)
        age_derivative = tuple(mp.diff(lambda s, axis=axis: _kkt_root(mp, source, constraints, rhs, target, s)[axis],
                                      _mp(mp, source.time)) for axis in range(2))
        for row, bounds in zip(jacobian, evolution.target_jacobian, strict=True):
            for expected, interval in zip(row, bounds, strict=True):
                assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
        for expected, interval in zip(age_derivative, evolution.time_derivative, strict=True):
            assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
        # Ambient target sensitivity is singular in the normal direction.
        for row in jacobian:
            assert abs(sum(value*a for value, a in zip(row, constraints[0], strict=True))) < mp.mpf("1e-75")


@pytest.mark.parametrize("index", [0, 1])
def test_backend_target_and_age_gradients_match_original_kkt_and_core_certificate(index):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target = _family(index)
    data = tuple(np.asarray(values, dtype=np.float64) for values in
                 (target, source.centers, source.biases, constraints, rhs, source.viscosity, source.time))
    torch_data = tuple(torch.tensor(value, dtype=torch.float64) for value in data)
    jax_data = tuple(jnp.asarray(value) for value in data)
    torch_result = torch_constrained(*torch_data)
    jax_result = jax.jit(jax_constrained)(*jax_data)
    assert bool(torch_result.converged) and bool(jax_result.converged)
    np.testing.assert_allclose(torch_result.value.detach().numpy(), np.asarray(jax_result.value), rtol=2e-12, atol=2e-12)
    torch_jacobian = torch.autograd.functional.jacobian(lambda v: torch_constrained(v, *torch_data[1:]).value, torch_data[0])
    jax_jacobian = jax.jit(jax.jacfwd(lambda v: jax_constrained(v, *jax_data[1:]).value))(jax_data[0])
    torch_age = torch.autograd.functional.jacobian(lambda s: torch_constrained(*torch_data[:-1], s).value, torch_data[-1])
    jax_age = jax.jit(jax.jacfwd(lambda s: jax_constrained(*jax_data[:-1], s).value))(jax_data[-1])
    with mp.workdps(85):
        root = _kkt_root(mp, source, constraints, rhs, target)
        jacobian = np.asarray(_true_target_jacobian(mp, source, constraints, rhs, target), dtype=np.float64)
        age_derivative = np.asarray([mp.diff(lambda s, axis=axis: _kkt_root(mp, source, constraints, rhs, target, s)[axis],
                                            _mp(mp, source.time)) for axis in range(2)], dtype=np.float64)
        np.testing.assert_allclose(torch_result.value.detach().numpy(), np.asarray(root, dtype=np.float64), rtol=3e-12, atol=3e-12)
        for actual in (torch_jacobian.detach().numpy(), np.asarray(jax_jacobian)):
            np.testing.assert_allclose(actual, jacobian, rtol=2e-11, atol=2e-12)
        for actual in (torch_age.detach().numpy(), np.asarray(jax_age)):
            np.testing.assert_allclose(actual, age_derivative, rtol=3e-11, atol=3e-12)
        restriction = restrict_gaussian_heat(source, constraints, rhs)
        for proposed in (torch_result.value.detach().tolist(), np.asarray(jax_result.value).tolist()):
            certified = certify_gaussian_affine_inverse(restriction, target, tuple(x/float(source.time) for x in proposed))
            assert certified is not None
            for value, interval in zip(root, certified.enclosure, strict=True):
                assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_nonlinear_constraint_parameter_gradients_and_second_derivatives():
    mp = pytest.importorskip("mpmath")
    source, _, _, target = _family(1)
    values = tuple(np.asarray(value, dtype=np.float64) for value in
                   (target, source.centers, source.biases, source.viscosity, source.time))
    tv, ta, tb, tn, ts = tuple(torch.tensor(value, dtype=torch.float64) for value in values)
    jv, ja, jb, jn, js = tuple(jnp.asarray(value) for value in values)

    def torch_function(t):
        matrix = torch.stack((1+t, 2+t*t))[None, :]
        right = (3+torch.sin(t)/5)[None]
        return torch_constrained(tv, ta, tb, matrix, right, tn, ts).value

    def jax_function(t):
        return jax_constrained(jv, ja, jb, jnp.stack((1+t, 2+t*t))[None, :],
                               (3+jnp.sin(t)/5)[None], jn, js).value

    parameter = 0.25
    with mp.workdps(90):
        def true(t):
            return _kkt_root(mp, source, ((1+t, 2+t*t),), (3+mp.sin(t)/5,), target)

        for order in (1, 2):
            expected = np.asarray([mp.diff(lambda t, axis=axis: true(t)[axis], _mp(mp, parameter), order)
                                   for axis in range(2)], dtype=np.float64)
            previous = torch_function

            def torch_function(t, previous=previous):
                return torch.autograd.functional.jacobian(previous, t, create_graph=True)

            jax_function = jax.jacfwd(jax_function)
            actual_t = torch_function(torch.tensor(parameter, dtype=torch.float64, requires_grad=True)).detach().numpy()
            actual_j = np.asarray(jax.jit(jax_function)(jnp.asarray(parameter)))
            np.testing.assert_allclose(actual_t, expected, rtol=2e-10, atol=3e-11)
            np.testing.assert_allclose(actual_j, expected, rtol=2e-10, atol=3e-11)
