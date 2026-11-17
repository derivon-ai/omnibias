# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained forward formulas versus exact Picard algebra and original KKT.

The oracle performs full formal fixed-point substitutions using finite powers
of exp and a geometric reciprocal. It retains the original limiting biases
and normal costs, independently of the implementation's coefficient recurrence.
Autodiff is compared with this finite formula, not with the true inverse.
"""

from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_aging import (
    certify_gaussian_affine_aging,
    certify_gaussian_affine_predictor,
)
from omnibias.core.verified.gaussian_aging import certify_gaussian_inverse_aging
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.jax.gaussian_constrained_aging import (
    gaussian_heat_inverse_constrained_aging as jax_aging,
)
from omnibias.jax.gaussian_constrained_aging import (
    gaussian_heat_inverse_constrained_predictor as jax_predictor,
)
from omnibias.torch.gaussian_constrained_aging import (
    gaussian_heat_inverse_constrained_aging as torch_aging,
)
from omnibias.torch.gaussian_constrained_aging import (
    gaussian_heat_inverse_constrained_predictor as torch_predictor,
)
from test_gaussian_affine_constraints import _kkt_root, _mp, _objective
from test_gaussian_inverse_aging import _exponential, _multiply, _reciprocal

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _one_row_geometry(centers, constraints, rhs, target):
    """Closed-form one-equation projection, independent of core elimination."""
    normal = constraints[0]
    squared = sum(value*value for value in normal)

    def project(vector):
        scale = sum(a*x for a, x in zip(normal, vector, strict=True))/squared
        return tuple(x-scale*a for x, a in zip(vector, normal, strict=True))

    anchor = tuple(a*rhs[0]/squared for a in normal)
    tangent = tuple(project(row) for row in centers)
    costs = tuple((sum(a*x for a, x in zip(normal, row, strict=True))-rhs[0])**2/squared for row in centers)
    return anchor, tangent, tuple(value-costs[0] for value in costs), project(target)


def _picard(tangent, posterior, costs, viscosity, time, order):
    """Formal substitution gains one coefficient because the exponent has t."""
    zero = viscosity*0
    dimension = len(tangent[0])
    mean = tuple(sum(p*row[axis] for p, row in zip(posterior, tangent, strict=True)) for axis in range(dimension))
    centered = tuple(tuple(a-mu for a, mu in zip(row, mean, strict=True)) for row in tangent)
    h = [[zero]*(order+1) for _ in range(dimension)]
    for _ in range(order):
        exponentials = []
        for row, cost in zip(centered, costs, strict=True):
            logit = [zero]+[sum(a*h[axis][n-1] for axis, a in enumerate(row))/(2*viscosity*time)
                            for n in range(1, order+1)]
            logit[1] -= (sum(a*a for a in row)+cost)/(4*viscosity*time)
            exponentials.append(_exponential(logit))
        denominator = [sum(p*e[n] for p, e in zip(posterior, exponentials, strict=True)) for n in range(order+1)]
        reciprocal = _reciprocal(denominator)
        for axis in range(dimension):
            numerator = [sum(p*a[axis]*e[n] for p, a, e in zip(posterior, centered, exponentials, strict=True))
                         for n in range(order+1)]
            h[axis] = _multiply(numerator, reciprocal)
            h[axis][0] = zero
    return tuple(tuple(h[axis][n] for axis in range(dimension)) for n in range(order+1)), mean


def _mp_formula(mp, data, order):
    target, centers, biases, constraints, rhs, viscosity, time = data
    anchor, tangent, costs, wanted = _one_row_geometry(centers, constraints, rhs, target)
    logits = [bias+sum(a*v for a, v in zip(row, wanted, strict=True))/(2*viscosity)
              for bias, row in zip(biases, tangent, strict=True)]
    weights = [mp.exp(value-max(logits)) for value in logits]
    posterior = tuple(value/sum(weights) for value in weights)
    coefficients, mean = _picard(tangent, posterior, costs, viscosity, time, order)
    value = tuple(x0+time*v+mu+sum(row[axis] for row in coefficients)
                  for axis, (x0, v, mu) in enumerate(zip(anchor, wanted, mean, strict=True)))
    return value, coefficients


def _data(source, constraints, rhs, target):
    return target, source.centers, source.biases, constraints, rhs, source.viscosity, source.time


def _mp_data(mp, data):
    def convert(value):
        if isinstance(value, (tuple, list)):
            return tuple(convert(entry) for entry in value)
        return _mp(mp, value)
    return tuple(convert(value) for value in data)


def _exact_family(index):
    if index == 0:
        source = GaussianHeatMap(((0, 0), (1, 2), (3, -1)), (0, 0, 0), time=16)
        return source, ((F(0), F(1)),), (F(1),), (F(0), F(0))
    source = GaussianHeatMap(((0, 0, 0), (1, 0, 2), (-1, 2, 1), (2, -1, 0)),
                             (0, 0, 0, 0), viscosity=F(3, 2), time=16)
    return source, ((F(1), F(2), F(-1)),), (F(2),), (F(0),)*3


@pytest.mark.parametrize("index", [0, 1], ids=("axis_2d", "oblique_3d"))
@pytest.mark.parametrize("order", [0, 1, 4, 8])
def test_exact_rational_picard_coefficients_match_core_and_both_backends(index, order):
    source, constraints, rhs, target = _exact_family(index)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction is not None
    anchor, tangent, costs, wanted = _one_row_geometry(source.centers, constraints, rhs, target)
    posterior = (F(1, source.mode_count),)*source.mode_count  # ORIGINAL biases and target are zero.
    coefficients, mean = _picard(tangent, posterior, costs, source.viscosity, source.time, order)
    expected = tuple(a+source.time*v+mu+sum(row[axis] for row in coefficients)
                     for axis, (a, v, mu) in enumerate(zip(anchor, wanted, mean, strict=True)))
    core = certify_gaussian_affine_aging(restriction, (Interval.point(0.0),)*source.dimension, order=order)
    assert core is not None and core.mode == "series"
    assert coefficients[0] == (0,)*source.dimension
    if index == 0 and order:
        assert coefficients[1] == (F(-65, 1728), F(0))
    for row, intervals in zip(coefficients, core.correction_coefficients, strict=True):
        for value, interval in zip(row, intervals, strict=True):
            assert F(interval.lo) <= value <= F(interval.hi)
    for value, interval in zip(expected, core.predictor_enclosure, strict=True):
        assert F(interval.lo) <= value <= F(interval.hi)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in _data(source, constraints, rhs, target))
    result_t = torch_aging(*(torch.tensor(value, dtype=torch.float64) for value in arrays), order=order)
    result_j = jax.jit(lambda *args: jax_aging(*args, order=order))(*(jnp.asarray(value) for value in arrays))
    assert bool(result_t.valid) and bool(result_j.valid)
    for values, rows in ((result_t.value.detach().numpy(), result_t.correction_coefficients.detach().numpy()),
                         (np.asarray(result_j.value), np.asarray(result_j.correction_coefficients))):
        np.testing.assert_allclose(rows, np.asarray(coefficients, dtype=np.float64), rtol=2e-11, atol=3e-14)
        np.testing.assert_allclose(values, np.asarray(expected, dtype=np.float64), rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("mode,order", [("predictor", 0), ("predictor", 1), ("series", 4), ("series", 8)])
def test_original_kkt_roots_and_objective_gaps_fit_uniform_target_box_certificates(mode, order):
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((0, 0), (1, 2), (3, -1)), (F(1, 8), F(-1, 4), F(3, 8)), time=16)
    constraints, rhs = ((F(1), F(2)),), (F(3, 4),)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    center = (F(1, 4), F(-1, 8))
    box = tuple(Interval(float(value-F(1, 64)), float(value+F(1, 64))) for value in center)
    factory = certify_gaussian_affine_predictor if mode == "predictor" else certify_gaussian_affine_aging
    certificate = factory(restriction, box, order=order)
    assert certificate is not None and certificate.mode == mode
    assert tuple(sum(a*x for a, x in zip(row, certificate.feasible_point, strict=True)) for row in constraints) == rhs
    with mp.workdps(90):
        for target in (center, (F(box[0].lo), F(box[1].hi)), (F(box[0].hi), F(box[1].lo))):
            root = _kkt_root(mp, source, constraints, rhs, target)
            formula, _ = _mp_formula(mp, _mp_data(mp, _data(source, constraints, rhs, target)), order)
            assert mp.sqrt(sum((a-b)**2 for a, b in zip(formula, root, strict=True))) <= _mp(mp, certificate.approximation_error_bound)
            point = tuple(_mp(mp, value) for value in certificate.feasible_point)
            assert mp.sqrt(sum((a-b)**2 for a, b in zip(point, root, strict=True))) <= _mp(mp, certificate.feasible_point_error_bound)
            for expected, interval in zip(root, certificate.enclosure, strict=True):
                assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
            for expected, interval in zip(formula, certificate.predictor_enclosure, strict=True):
                assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
            minimum = _objective(mp, source, target, root)
            formula_gap = _objective(mp, source, target, formula)-minimum
            feasible_gap = _objective(mp, source, target, point)-minimum
            assert -mp.mpf("1e-75") <= formula_gap <= _mp(mp, certificate.objective_gap_bound)
            assert -mp.mpf("1e-75") <= feasible_gap <= _mp(mp, certificate.feasible_point_objective_gap_bound)


def test_fixed_transformed_biases_have_different_original_age_coefficients():
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, target = _exact_family(0)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    correct = certify_gaussian_affine_aging(restriction, (Interval.point(0.0),)*2, order=1)
    frozen = certify_gaussian_inverse_aging(restriction.model, (Interval.point(0.0),)*2, order=1)
    assert correct is not None and frozen is not None
    assert correct.correction_coefficients[1][0].contains(float(F(-65, 1728)))
    assert not frozen.correction_coefficients[1][0].contains(float(F(-65, 1728)))
    # At the original age both families represent the same constrained
    # objective. Reusing the transformed biases at a different age does not.
    with mp.workdps(90):
        original_at_new_age = _kkt_root(mp, source, constraints, rhs, target, 2*source.time)
        frozen_at_new_age = _kkt_root(mp, restriction.model, constraints, rhs, target, 2*source.time)
        assert abs(original_at_new_age[0]-frozen_at_new_age[0]) > mp.mpf("1e-3")


def test_predictor_uses_q_tangent_while_series_requires_the_larger_aging_parameter():
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)))
    constraints, rhs, target = ((0, 1),), (1,), (0, 0)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction.model.contraction_bound == F(1, 2)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in _data(source, constraints, rhs, target))
    data_t = tuple(torch.tensor(value, dtype=torch.float64) for value in arrays)
    data_j = tuple(jnp.asarray(value) for value in arrays)
    for order in (0, 1):
        core = certify_gaussian_affine_predictor(restriction, (Interval.point(0.0),)*2, order=order)
        assert core is not None and core.aging_parameter == F(3, 2)
        assert bool(torch_predictor(*data_t, order=order).valid)
        assert bool(jax_predictor(*data_j, order=order).valid)
        assert certify_gaussian_affine_aging(restriction, (Interval.point(0.0),)*2, order=order) is None
        assert not bool(torch_aging(*data_t, order=order).valid)
        assert not bool(jax_aging(*data_j, order=order).valid)


def test_identical_tangent_centers_are_exact_despite_huge_normal_cost_variation():
    huge = 2**1000
    source = GaussianHeatMap(((2, huge), (2, huge+1)), (0, 0))
    constraints, rhs, target = ((0, 1),), (0,), (F(1, 4), F(-1, 2))
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction is not None and restriction.model.diameter_squared == 0
    assert max(restriction.normal_cost_differences) == 2*huge+1
    for factory in (certify_gaussian_affine_predictor, certify_gaussian_affine_aging):
        result = factory(restriction, tuple(Interval.from_rational(value) for value in target), order=1)
        assert result is not None and result.approximation_error_bound == 0
        assert result.feasible_point == (F(9, 4), F(0))
        assert all(interval.lo <= expected <= interval.hi
                   for interval, expected in zip(result.enclosure, (F(9, 4), F(0)), strict=True))
    # An exactly representable large FLOAT family is a separate numerical
    # instance: tangent collapse still removes all normal-weight dependence.
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in _data(source, constraints, rhs, target))
    for function in (torch_predictor, torch_aging):
        numerical = function(*(torch.tensor(value, dtype=torch.float64) for value in arrays), order=1)
        assert bool(numerical.valid)
        np.testing.assert_array_equal(numerical.value.detach().numpy(), (2.25, 0.0))
    for function in (jax_predictor, jax_aging):
        numerical = function(*(jnp.asarray(value) for value in arrays), order=1)
        assert bool(numerical.valid)
        np.testing.assert_array_equal(np.asarray(numerical.value), (2.25, 0.0))


@pytest.mark.parametrize("mode,order", [("predictor", 1), ("series", 4)])
def test_all_parameter_autodiff_matches_independent_finite_formula(mode, order):
    mp = pytest.importorskip("mpmath")
    data = (np.array([0.25, -0.125]), np.array([[0., 0.], [1., 2.], [3., -1.]]),
            np.array([0.125, -0.25, 0.375]), np.array([[1., 2.]]), np.array([0.75]), np.array(1.25), np.array(16.))
    function_t = torch_predictor if mode == "predictor" else torch_aging
    function_j = jax_predictor if mode == "predictor" else jax_aging
    tensors = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in data)
    upstream = (0.75, -0.5)
    result = function_t(*tensors, order=order)
    assert bool(result.valid)
    loss = (result.value*torch.tensor(upstream, dtype=torch.float64)).sum()
    gradients_t = torch.autograd.grad(loss, tensors)
    gradients_j = jax.jit(jax.grad(lambda *args: jnp.dot(function_j(*args, order=order).value, jnp.asarray(upstream)),
                                  argnums=tuple(range(7))))(*(jnp.asarray(value) for value in data))
    flat = np.concatenate([value.reshape(-1) for value in data]).tolist()

    def unpack(values):
        result, offset = [], 0
        for array in data:
            size = array.size
            part = values[offset:offset+size]
            result.append(part[0] if array.ndim == 0 else np.asarray(part, dtype=object).reshape(array.shape).tolist())
            offset += size
        return result

    with mp.workdps(90):
        exact = [_mp(mp, value) for value in flat]

        def scalar(values):
            value, _ = _mp_formula(mp, unpack(values), order)
            return sum(_mp(mp, u)*x for u, x in zip(upstream, value, strict=True))

        expected = [mp.diff(lambda value, i=i: scalar(exact[:i]+[value]+exact[i+1:]), original)
                    for i, original in enumerate(exact)]
        reference = np.asarray(expected, dtype=np.float64)
    actual_t = np.concatenate([value.detach().numpy().reshape(-1) for value in gradients_t])
    actual_j = np.concatenate([np.asarray(value).reshape(-1) for value in gradients_j])
    np.testing.assert_allclose(actual_t, reference, rtol=3e-10, atol=2e-12)
    np.testing.assert_allclose(actual_j, reference, rtol=3e-10, atol=2e-12)
