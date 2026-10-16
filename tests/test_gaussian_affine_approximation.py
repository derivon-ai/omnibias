# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed-age constrained approximations against independent algebra and KKT.

The oracle absorbs the normal energies into the limiting posterior, then
performs formal Picard substitutions without an extra normal-cost term.
Parameter derivatives concern this finite formula, not the true inverse.
"""

from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_aging import certify_gaussian_affine_aging
from omnibias.core.verified.gaussian_affine_approximation import (
    certify_gaussian_affine_approximation,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.jax.gaussian_constrained_approximation import (
    gaussian_heat_inverse_constrained_approximation as jax_approximation,
)
from omnibias.torch.gaussian_constrained_approximation import (
    gaussian_heat_inverse_constrained_approximation as torch_approximation,
)
from test_gaussian_affine_aging import _data, _mp_data, _one_row_geometry, _picard
from test_gaussian_affine_constraints import _kkt_root, _mp, _objective

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _resummed_formula(mp, data, order, *, frozen_bias_age=None):
    target, centers, biases, constraints, rhs, viscosity, time = data
    anchor, tangent, costs, wanted = _one_row_geometry(centers, constraints, rhs, target)
    # Freezing this age is deliberately a DIFFERENT parameterized family and
    # is used only by the regression demonstrating the missing chain rule.
    bias_age = time if frozen_bias_age is None else frozen_bias_age
    logits = [bias-cost/(4*viscosity*bias_age)+sum(a*v for a, v in zip(row, wanted, strict=True))/(2*viscosity)
              for bias, cost, row in zip(biases, costs, tangent, strict=True)]
    weights = [mp.exp(value-max(logits)) for value in logits]
    posterior = tuple(value/sum(weights) for value in weights)
    zero_costs = (viscosity*0,)*len(centers)
    coefficients, mean = _picard(tangent, posterior, zero_costs, viscosity, time, order)
    value = tuple(a+time*v+mu+sum(row[axis] for row in coefficients)
                  for axis, (a, v, mu) in enumerate(zip(anchor, wanted, mean, strict=True)))
    return value, coefficients


def _exact_family(index):
    if index == 0:
        centers, constraints, rhs, nu, time = (((F(0), F(0)), (F(1), F(100)), (F(3), F(101))),
                                              ((F(0), F(1)),), (F(0),), F(1), F(16))
    else:
        centers = tuple(tuple(map(F, row)) for row in ((0, 0, 0), (1, 0, 2), (-1, 2, 1), (2, -1, 0)))
        constraints, rhs, nu, time = ((F(1), F(2), F(-1)),), (F(2),), F(3, 2), F(16)
    target = (F(0),)*len(centers[0])
    _, _, costs, _ = _one_row_geometry(centers, constraints, rhs, target)
    # Exactly cancel each relative normal energy at this selected age. The
    # transformed limiting posterior is uniform although original biases vary.
    biases = tuple(cost/(4*nu*time) for cost in costs)
    return GaussianHeatMap(centers, biases, viscosity=nu, time=time), constraints, rhs, target


@pytest.mark.parametrize("index", [0, 1], ids=("large_normal_cost", "oblique_3d"))
@pytest.mark.parametrize("order", [0, 1, 2, 8])
def test_exact_fixed_age_picard_coefficients_match_core_and_both_backends(index, order):
    source, constraints, rhs, target = _exact_family(index)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction is not None
    anchor, tangent, costs, wanted = _one_row_geometry(source.centers, constraints, rhs, target)
    posterior = (F(1, source.mode_count),)*source.mode_count
    coefficients, mean = _picard(tangent, posterior, (F(0),)*source.mode_count, source.viscosity, source.time, order)
    expected = tuple(a+source.time*v+mu+sum(row[axis] for row in coefficients)
                     for axis, (a, v, mu) in enumerate(zip(anchor, wanted, mean, strict=True)))
    box = (Interval.point(0.0),)*source.dimension
    core = certify_gaussian_affine_approximation(restriction, box, order=order)
    assert core is not None and core.base_approximation is not None
    if index == 0:
        original_aging_parameter = (restriction.model.diameter_squared+max(costs)-min(costs))/(8*source.viscosity*source.time)
        assert original_aging_parameter == F(5105, 64) > F(1, 3)
        assert restriction.model.contraction_bound == F(9, 128) < F(1, 3)
        assert certify_gaussian_affine_aging(restriction, box, order=order) is None
        if order:
            assert coefficients[1] == (F(-5, 432), F(0))
    for row, intervals in zip(coefficients, core.correction_coefficients, strict=True):
        for value, interval in zip(row, intervals, strict=True):
            assert F(interval.lo) <= value <= F(interval.hi)
    for value, interval in zip(expected, core.predictor_enclosure, strict=True):
        assert F(interval.lo) <= value <= F(interval.hi)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in _data(source, constraints, rhs, target))
    result_t = torch_approximation(*(torch.tensor(value, dtype=torch.float64) for value in arrays), order=order)
    result_j = jax.jit(lambda *args: jax_approximation(*args, order=order))(*(jnp.asarray(value) for value in arrays))
    assert bool(result_t.valid) and bool(result_j.valid)
    for value, rows in ((result_t.value.detach().numpy(), result_t.correction_coefficients.detach().numpy()),
                        (np.asarray(result_j.value), np.asarray(result_j.correction_coefficients))):
        np.testing.assert_allclose(rows, np.asarray(coefficients, dtype=np.float64), rtol=3e-10, atol=3e-13)
        np.testing.assert_allclose(value, np.asarray(expected, dtype=np.float64), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("order", [0, 1, 4, 8])
def test_original_kkt_roots_and_objective_gaps_at_large_normal_cost(order):
    mp = pytest.importorskip("mpmath")
    source, constraints, rhs, _ = _exact_family(0)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    box = (Interval(0.125, 0.15625), Interval(-1.0, 2.0))
    result = certify_gaussian_affine_approximation(restriction, box, order=order)
    assert result is not None
    assert result.projected_target_box[1].lo == result.projected_target_box[1].hi == 0
    assert result.feasible_point[1] == 0
    with mp.workdps(90):
        for target in ((F(1, 8), F(-1)), (F(9, 64), F(0)), (F(5, 32), F(2))):
            root = _kkt_root(mp, source, constraints, rhs, target)
            formula, _ = _resummed_formula(mp, _mp_data(mp, _data(source, constraints, rhs, target)), order)
            assert mp.sqrt(sum((a-b)**2 for a, b in zip(formula, root, strict=True))) <= _mp(mp, result.approximation_error_bound)
            feasible = tuple(_mp(mp, value) for value in result.feasible_point)
            assert mp.sqrt(sum((a-b)**2 for a, b in zip(feasible, root, strict=True))) <= _mp(mp, result.feasible_point_error_bound)
            for value, interval in zip(root, result.enclosure, strict=True):
                assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
            for value, interval in zip(formula, result.predictor_enclosure, strict=True):
                assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
            optimum = _objective(mp, source, target, root)
            assert -mp.mpf("1e-75") <= _objective(mp, source, target, formula)-optimum <= _mp(mp, result.objective_gap_bound)
            assert -mp.mpf("1e-75") <= _objective(mp, source, target, feasible)-optimum <= _mp(mp, result.feasible_point_objective_gap_bound)


def test_low_orders_use_real_q_bounds_beyond_series_convergence_disk():
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)))
    constraints, rhs, target = ((0, 1),), (1,), (0, 0)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction.model.contraction_bound == F(1, 2)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in _data(source, constraints, rhs, target))
    data_t = tuple(torch.tensor(value, dtype=torch.float64) for value in arrays)
    data_j = tuple(jnp.asarray(value) for value in arrays)
    for order in (0, 1):
        result = certify_gaussian_affine_approximation(restriction, (Interval.point(0.0),)*2, order=order)
        assert result is not None and result.approximation_error_bound == F(1, 2)
        assert bool(torch_approximation(*data_t, order=order).valid)
        assert bool(jax_approximation(*data_j, order=order).valid)
    for order in (2, 8):
        assert certify_gaussian_affine_approximation(restriction, (Interval.point(0.0),)*2, order=order) is None
        assert not bool(torch_approximation(*data_t, order=order).valid)
        assert not bool(jax_approximation(*data_j, order=order).valid)


def test_physical_age_chain_rule_and_second_gradient_differ_from_freezing_biases():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)))
    constraints, rhs, target = ((0, 1),), (1,), (0, 0)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in _data(source, constraints, rhs, target))
    data_t = tuple(torch.tensor(value, dtype=torch.float64) for value in arrays)
    data_j = tuple(jnp.asarray(value) for value in arrays)

    def function_t(s):
        return torch_approximation(*data_t[:-1], s, order=1).value

    def function_j(s):
        return jax_approximation(*data_j[:-1], s, order=1).value

    with mp.workdps(90):
        data = _mp_data(mp, _data(source, constraints, rhs, target))
        for order, exact in ((1, (F(-3, 2), F(0))), (2, (F(4), F(0)))):
            expected = [mp.diff(lambda s, axis=axis: _resummed_formula(mp, (*data[:-1], s), 1)[0][axis], 1, order)
                        for axis in range(2)]
            for value, known in zip(expected, exact, strict=True):
                assert abs(value-_mp(mp, known)) < mp.mpf("1e-75")
            previous = function_t

            def function_t(s, previous=previous):
                return torch.autograd.functional.jacobian(previous, s, create_graph=True)

            function_j = jax.jacfwd(function_j)
            actual_t = function_t(torch.tensor(1.0, dtype=torch.float64, requires_grad=True)).detach().numpy()
            actual_j = np.asarray(jax.jit(function_j)(jnp.asarray(1.0)))
            np.testing.assert_allclose(actual_t, np.asarray(exact, dtype=np.float64), rtol=2e-11, atol=2e-12)
            np.testing.assert_allclose(actual_j, np.asarray(exact, dtype=np.float64), rtol=2e-11, atol=2e-12)
            frozen = mp.diff(lambda s: _resummed_formula(mp, (*data[:-1], s), 1, frozen_bias_age=mp.mpf(1))[0][0], 1, order)
            assert abs(frozen) < mp.mpf("1e-75")
        true_age_derivative = mp.diff(lambda s: _kkt_root(mp, source, constraints, rhs, target, s)[0], 1)
        assert abs(true_age_derivative+2) < mp.mpf("1e-75")
        assert abs(true_age_derivative+mp.mpf(3)/2) > mp.mpf("0.4")


@pytest.mark.parametrize("order", [1, 4])
def test_all_parameter_autodiff_matches_independent_resummed_finite_formula(order):
    mp = pytest.importorskip("mpmath")
    data = (np.array([0.25, -0.125]), np.array([[0., 0.], [1., 2.], [3., -1.]]),
            np.array([0.125, -0.25, 0.375]), np.array([[1., 2.]]), np.array([0.75]), np.array(1.25), np.array(16.))
    tensors = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in data)
    upstream = (0.75, -0.5)
    result = torch_approximation(*tensors, order=order)
    assert bool(result.valid)
    gradients_t = torch.autograd.grad((result.value*torch.tensor(upstream, dtype=torch.float64)).sum(), tensors)
    gradients_j = jax.jit(jax.grad(lambda *args: jnp.dot(jax_approximation(*args, order=order).value, jnp.asarray(upstream)),
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
            value, _ = _resummed_formula(mp, unpack(values), order)
            return sum(_mp(mp, u)*x for u, x in zip(upstream, value, strict=True))

        expected = [mp.diff(lambda value, i=i: scalar(exact[:i]+[value]+exact[i+1:]), original)
                    for i, original in enumerate(exact)]
        reference = np.asarray(expected, dtype=np.float64)
    actual_t = np.concatenate([value.detach().numpy().reshape(-1) for value in gradients_t])
    actual_j = np.concatenate([np.asarray(value).reshape(-1) for value in gradients_j])
    np.testing.assert_allclose(actual_t, reference, rtol=3e-10, atol=2e-12)
    np.testing.assert_allclose(actual_j, reference, rtol=3e-10, atol=2e-12)
