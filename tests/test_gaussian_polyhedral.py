# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Polyhedral Gaussian optima checked by independent active-set enumeration.

Original Gaussian KKT equations and objective values select the reference
face. Fixed-face Torch/JAX gradients are numerical oracles only after the
core certificate has established strict active-set stability.
"""

from fractions import Fraction as F
from itertools import combinations

import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import (
    certify_gaussian_polyhedral_minimum,
    certify_gaussian_polyhedral_parameter_vjp,
    compile_gaussian_polyhedral_face,
)
from test_gaussian_affine_constraints import _mp, _objective
from test_gaussian_affine_parameters import (
    _backend_true_gradients,
    _data,
    _flatten,
    _original_parameter_vjp,
    _residual,
)
from test_gaussian_affine_sensitivity import _backend_proposal


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _square():
    source = GaussianHeatMap(((-1, 0), (1, 0), (0, 1)), (0, F(1, 4), F(-1, 8)), time=4)
    inequalities = ((1, 0), (-1, 0), (0, 1), (0, -1))
    return source, inequalities, (1, 1, 1, 1)


def _active_kkt(mp, source, equalities, rhs, inequalities, upper, active, target):
    constraints = equalities+tuple(inequalities[i] for i in active)
    right = rhs+tuple(upper[i] for i in active)
    if constraints:
        matrix = mp.matrix(constraints)
        gram = matrix*matrix.T
        if len(constraints) > source.dimension or mp.det(gram) == 0:
            return None
        anchor = tuple(matrix.T*mp.lu_solve(gram, mp.matrix(right)))
    else:
        anchor = (mp.mpf(0),)*source.dimension
    flat, shapes = _flatten(_data(source, constraints, right, target))
    exact = tuple(_mp(mp, value) for value in flat)
    initial = anchor+(mp.mpf(0),)*len(constraints)
    state = tuple(mp.findroot(lambda *state: _residual(mp, state, exact, shapes), initial,
                             tol=mp.eps*32, maxsteps=100))
    assert max(abs(value) for value in _residual(mp, state, exact, shapes)) < mp.sqrt(mp.eps)
    point, multipliers = state[:source.dimension], state[source.dimension:]
    slacks = tuple(_mp(mp, bound)-sum(_mp(mp, a)*x for a, x in zip(row, point, strict=True))
                   for row, bound in zip(inequalities, upper, strict=True))
    return point, multipliers, slacks


def _enumerate_original_minimum(mp, source, equalities, rhs, inequalities, upper, target):
    feasible = []
    for size in range(source.dimension-len(equalities)+1):
        for active in combinations(range(len(inequalities)), size):
            result = _active_kkt(mp, source, equalities, rhs, inequalities, upper, active, target)
            if result is None:
                continue
            point, multipliers, slacks = result
            tolerance = mp.mpf("1e-75")
            if min(slacks, default=0) < -tolerance or min(multipliers[len(equalities):], default=0) < -tolerance:
                continue
            feasible.append((_objective(mp, source, target, point), active, result))
    assert feasible
    feasible.sort(key=lambda entry: entry[0])
    return feasible[0][1:]


@pytest.fixture(scope="module")
def _square_oracles():
    mp = pytest.importorskip("mpmath")
    source, inequalities, upper = _square()
    with mp.workdps(100):
        return {target: _enumerate_original_minimum(mp, source, (), (), inequalities, upper, target)
                for target in ((0, 0), (1, 0), (1, 1), (0, -1), (-1, 1))}


def _scaled(point, age):
    return tuple(float(value)/float(age) for value in point)


def _assert_contains(mp, intervals, expected):
    actual_flat, shape = _flatten(intervals)
    expected_flat, expected_shape = _flatten(expected)
    assert shape == expected_shape
    for interval, value in zip(actual_flat, expected_flat, strict=True):
        assert _mp(mp, interval.lo) <= _mp(mp, value) <= _mp(mp, interval.hi)


@pytest.mark.parametrize("target,expected_active", [((0, 0), ()), ((1, 0), (0,)), ((1, 1), (0, 2)),
                                                    ((0, -1), (3,)), ((-1, 1), (1, 2))])
def test_polytope_minimum_matches_independent_original_active_set_enumeration(target, expected_active, _square_oracles):
    mp = pytest.importorskip("mpmath")
    source, inequalities, upper = _square()
    active, (point, multipliers, slacks) = _square_oracles[target]
    assert active == expected_active
    base = restrict_gaussian_heat(source, (), ())
    # Compiler normalization must not confuse gradient/multiplier row slots.
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, active_indices=tuple(reversed(active)))
    assert face is not None and face.active_indices == active
    result = certify_gaussian_polyhedral_minimum(face, target, _scaled(point, source.time))
    assert result is not None and result.active_set_stable
    assert result.strong_convexity_bound == (1-base.model.contraction_bound)/source.time
    with mp.workdps(100):
        _assert_contains(mp, (result.root_enclosure,), (point,))
        for i, interval in enumerate(result.inequality_multipliers):
            if i in active:
                assert interval.lo > 0
                assert _mp(mp, interval.lo) <= multipliers[active.index(i)] <= _mp(mp, interval.hi)
                assert result.slacks[i].lo == result.slacks[i].hi == 0
            else:
                assert interval.lo == interval.hi == 0
                assert result.slacks[i].lo > 0
                assert _mp(mp, result.slacks[i].lo) <= slacks[i] <= _mp(mp, result.slacks[i].hi)
        optimum = _objective(mp, source, target, point)
        for candidate in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0), (F(1, 4), F(-1, 2))):
            feasible = tuple(_mp(mp, value) for value in candidate)
            gap = _objective(mp, source, target, feasible)-optimum
            promised = _mp(mp, result.strong_convexity_bound)*sum((a-b)**2 for a, b in zip(feasible, point, strict=True))/2
            assert gap+mp.mpf("1e-80") >= promised


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("target", [(1, 0), (1, 1)])
def test_stable_selected_face_parameter_gradients_and_inactive_zeros(backend, target, _square_oracles):
    mp = pytest.importorskip("mpmath")
    source, inequalities, upper = _square()
    active, _ = _square_oracles[target]
    constraints = tuple(inequalities[i] for i in active)
    rhs = tuple(upper[i] for i in active)
    upstream = (0.75, -0.5)
    point, proposal = _backend_proposal(backend, source, constraints, rhs, target, upstream, 8)
    face = compile_gaussian_polyhedral_face(restrict_gaussian_heat(source, (), ()), inequalities, upper, active_indices=active)
    minimum = certify_gaussian_polyhedral_minimum(face, target, _scaled(point, source.time))
    assert minimum is not None and minimum.active_set_stable
    derivative = certify_gaussian_polyhedral_parameter_vjp(minimum, upstream, proposal)
    assert derivative is not None
    assert derivative.equality_gradient == derivative.equality_rhs_gradient == ()
    blocks = (derivative.target_gradient, derivative.center_gradient, derivative.bias_gradient,
              tuple(derivative.inequality_gradient[i] for i in active),
              tuple(derivative.inequality_rhs_gradient[i] for i in active),
              derivative.viscosity_gradient, derivative.time_gradient)
    with mp.workdps(100):
        expected, _ = _original_parameter_vjp(mp, _data(source, constraints, rhs, target), upstream)
        # At a full-rank face, all model/target derivatives vanish exactly.
        if len(active) == source.dimension:
            expected = ((0, 0), ((0, 0),)*source.mode_count, (0,)*source.mode_count,
                        expected[3], expected[4], 0, 0)
        _assert_contains(mp, blocks, expected)
    for index in set(range(len(inequalities)))-set(active):
        assert all(value.lo == value.hi == 0 for value in derivative.inequality_gradient[index])
        assert derivative.inequality_rhs_gradient[index].lo == derivative.inequality_rhs_gradient[index].hi == 0
    gradients = _backend_true_gradients(backend, _data(source, constraints, rhs, target), upstream)
    for actual, known in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, np.asarray(known, dtype=np.float64), rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_ambient_indefinite_base_equality_positive_and_full_active_face_gradients(backend):
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)))
    equalities, rhs = ((0, 1),), (1,)
    inequalities, upper, target = ((1, 0), (-1, 0)), (0, 2), (1, 0)
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert source.contraction_bound == F(5, 2) and base.model.contraction_bound == F(1, 2)
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, active_indices=(0,))
    minimum = certify_gaussian_polyhedral_minimum(face, target, (100.0, -27.0))
    assert minimum is not None and minimum.active_set_stable
    assert minimum.strong_convexity_bound == F(1, 2)
    with mp.workdps(100):
        active, (point, multipliers, slacks) = _enumerate_original_minimum(
            mp, source, equalities, rhs, inequalities, upper, target,
        )
        assert active == (0,)
        assert max(abs(a-b) for a, b in zip(point, (0, 1), strict=True)) < mp.mpf("1e-80")
        assert max(abs(a-b) for a, b in zip(multipliers, (-1, 1), strict=True)) < mp.mpf("1e-80")
        assert abs(slacks[1]-2) < mp.mpf("1e-80")
        _assert_contains(mp, (minimum.equality_multipliers, minimum.inequality_multipliers), ((-1,), (1, 0)))
    upstream = (1.0, 3.0)
    derivative = certify_gaussian_polyhedral_parameter_vjp(minimum, upstream, (123.0, -321.0))
    assert derivative is not None
    expected = ((0, 0), ((0, 0),)*2, (0, 0), ((0, -3), (0, -1)), (3, 1), 0, 0)
    blocks = (derivative.target_gradient, derivative.center_gradient, derivative.bias_gradient,
              derivative.equality_gradient+(derivative.inequality_gradient[0],),
              derivative.equality_rhs_gradient+(derivative.inequality_rhs_gradient[0],),
              derivative.viscosity_gradient, derivative.time_gradient)
    with mp.workdps(100):
        _assert_contains(mp, blocks, expected)
    gradients = _backend_true_gradients(backend, _data(source, equalities+(inequalities[0],), rhs+(upper[0],), target), upstream)
    for actual, known in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, np.asarray(known, dtype=np.float64), rtol=1e-10, atol=2e-11)


@pytest.mark.parametrize("active", [(), (0,)])
def test_boundary_optimum_is_certified_but_active_set_derivative_is_refused(active):
    source = GaussianHeatMap(((0,),), (0,))
    base = restrict_gaussian_heat(source, (), ())
    face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), active_indices=active)
    result = certify_gaussian_polyhedral_minimum(face, (0,), (73.0,))
    assert result is not None and not result.active_set_stable
    assert result.root_enclosure[0].lo == result.root_enclosure[0].hi == 0
    assert result.inequality_multipliers[0].lo == result.inequality_multipliers[0].hi == 0
    assert result.slacks[0].lo == result.slacks[0].hi == 0
    assert certify_gaussian_polyhedral_parameter_vjp(result, (1.0,), (0.0,)) is None


def test_singleton_face_does_not_prove_global_optimality_in_nonconvex_base():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((-3,), (3,)), (0, 0))
    base = restrict_gaussian_heat(source, (), ())
    singleton = restrict_gaussian_heat(source, ((1,),), (0,))
    assert base.model.contraction_bound == F(9, 2) and singleton.model.contraction_bound == 0
    face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), active_indices=(0,))
    assert face is not None
    assert certify_gaussian_polyhedral_minimum(face, (1,), (0.0,)) is None
    with mp.workdps(100):
        # x=0 has a positive active multiplier 1 but is not a global optimum:
        # the feasible competitor x=-3 has a strictly smaller original cost.
        state = _active_kkt(mp, source, (), (), ((1,),), (0,), (0,), (1,))
        assert abs(state[1][0]-1) < mp.mpf("1e-80")
        assert _objective(mp, source, (1,), (mp.mpf(-3),)) < _objective(mp, source, (1,), (mp.mpf(0),))


@pytest.mark.parametrize("active", [(), (1,)])
def test_primal_infeasible_or_negative_multiplier_face_is_refused(active):
    mp = pytest.importorskip("mpmath")
    source, inequalities, upper = _square()
    target = (1, 0)
    face = compile_gaussian_polyhedral_face(restrict_gaussian_heat(source, (), ()), inequalities, upper, active_indices=active)
    with mp.workdps(100):
        point, multipliers, slacks = _active_kkt(mp, source, (), (), inequalities, upper, active, target)
        assert min(slacks) < 0 or min(multipliers) < 0
    assert certify_gaussian_polyhedral_minimum(face, target, _scaled(point, source.time)) is None
