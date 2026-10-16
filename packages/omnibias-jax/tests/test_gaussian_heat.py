# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent roots, implicit parameter derivatives, tracing and refusal."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.gaussian_heat import gaussian_heat_inverse

jax.config.update("jax_enable_x64", True)


def _data():
    return (jnp.array([0.2, -0.15]), jnp.array([[0.1, -0.2], [0.8, 0.3], [-0.2, 0.6]]),
            jnp.array([0.2, -0.1, 0.3]), jnp.array(0.8), jnp.array(1.3))


def _reference(target, centers, biases, viscosity, time):
    """Independent high-precision physical-coordinate root, not our iteration."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(70):
        v = [mp.mpf(float(value)) for value in target]
        a = [[mp.mpf(float(value)) for value in row] for row in centers]
        b = [mp.mpf(float(value)) for value in biases]
        nu, s = mp.mpf(float(viscosity)), mp.mpf(float(time))

        def residual(*point):
            weights = [mp.exp(bias - sum((x - center)**2 for x, center in zip(point, row, strict=True))/(4*nu*s))
                       for row, bias in zip(a, b, strict=True)]
            total = sum(weights)
            return tuple(x - s*goal - sum(w*row[axis] for w, row in zip(weights, a, strict=True))/total
                         for axis, (x, goal) in enumerate(zip(point, v, strict=True)))

        root = mp.findroot(residual, tuple(s*goal for goal in v), tol=mp.mpf("1e-55"))
        return np.array([float(value) for value in root])


@pytest.mark.parametrize("adjoint", ["auto", "dense", "low_rank"])
def test_jitted_inverse_matches_independent_root_and_live_parameter_gradients(adjoint):
    args = _data()
    solve = jax.jit(lambda *p: gaussian_heat_inverse(*p, adjoint=adjoint))
    result = solve(*args)
    assert bool(result.converged)
    np.testing.assert_allclose(result.value, _reference(*args), rtol=2e-13, atol=2e-13)
    weights = jnp.array([0.7, -0.4])
    def objective(*p):
        return jnp.vdot(weights, gaussian_heat_inverse(*p, adjoint=adjoint).value)
    gradients = jax.jit(jax.grad(objective, argnums=(0, 1, 2, 3, 4)))(*args)
    step = 2e-5
    for parameter, gradient in enumerate(gradients):
        expected = np.empty(args[parameter].shape)
        for index in np.ndindex(expected.shape):
            plus, minus = [np.array(value) for value in args], [np.array(value) for value in args]
            plus[parameter][index] += step
            minus[parameter][index] -= step
            expected[index] = np.dot(np.asarray(weights), _reference(*plus) - _reference(*minus))/(2*step)
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(gradient, expected, rtol=2e-7, atol=2e-9)
    # Common bias shifts change no probabilities; common center translations
    # translate the physical inverse by exactly the same vector.
    np.testing.assert_allclose(jnp.sum(gradients[2]), 0.0, atol=3e-15)
    np.testing.assert_allclose(jnp.sum(gradients[1], axis=0), weights, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("adjoint", ["dense", "low_rank"])
def test_second_derivative_matches_independent_implicit_scalar_formula(adjoint):
    # Centers ±a give x−s*v−a*tanh(a*x/(2nu*s))=0.
    a, nu, time, target = 0.7, 0.8, 1.2, 0.3
    centers, biases = jnp.array([[-a], [a]]), jnp.zeros(2)

    def value(v):
        return gaussian_heat_inverse(jnp.array([v]), centers, biases, nu, time, adjoint=adjoint).value[0]

    x = float(value(target))
    c = a/(2*nu*time)
    tanh = np.tanh(c*x)
    h = 1 - a*c*(1 - tanh*tanh)
    first = time/h
    second = -2*a*c*c*(1 - tanh*tanh)*tanh*first*first/h
    np.testing.assert_allclose(jax.jit(jax.grad(value))(target), first, rtol=3e-13)
    np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(value)))(target), second, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(jax.jacfwd(jax.jacrev(value))(target), second, rtol=3e-12, atol=1e-13)


def test_mixed_second_parameter_derivatives_are_symmetric_and_nonzero():
    target, centers, biases, nu, time = _data()

    def value(parameters):
        return jnp.sum(gaussian_heat_inverse(target, centers, biases, parameters[0], parameters[1],
                                             adjoint="low_rank").value**2)

    hessian = jax.jit(jax.hessian(value))(jnp.array([nu, time]))
    assert np.isfinite(hessian).all() and abs(float(hessian[0, 1])) > 1e-4
    np.testing.assert_allclose(hessian, hessian.T, rtol=3e-12, atol=3e-12)
    delta = 1e-4
    center = np.array([nu, time], dtype=float)
    fpp, fpm, fmp, fmm = [float(value(jnp.array(center + delta*np.array(sign))))
                         for sign in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    np.testing.assert_allclose(hessian[0, 1], (fpp-fpm-fmp+fmm)/(4*delta**2), rtol=2e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_multibatch_jit_vmap_dtypes_and_shared_gradients(dtype):
    target, centers, biases, nu, time = [value.astype(dtype) for value in _data()]
    batch = jnp.stack((target, 2*target, -target, target/2)).reshape(2, 2, 2)
    function = jax.jit(gaussian_heat_inverse)
    result = function(batch, centers, biases, nu, time)
    assert result.value.shape == batch.shape and result.value.dtype == dtype
    assert result.converged.shape == (2, 2) and bool(jnp.all(result.converged))
    assert result.residual_bound.shape == (2, 2) and result.contraction_bound.shape == ()
    mapped = jax.jit(jax.vmap(jax.vmap(lambda v: gaussian_heat_inverse(v, centers, biases, nu, time))))(batch)
    np.testing.assert_allclose(result.value, mapped.value, rtol=1e-6, atol=1e-6)
    grad_batch = jax.grad(lambda a: jnp.sum(gaussian_heat_inverse(batch, a, biases, nu, time).value))(centers)
    grad_rows = sum(jax.grad(lambda a, row=row: jnp.sum(gaussian_heat_inverse(row, a, biases, nu, time).value))(centers)
                    for row in batch.reshape(-1, 2))
    np.testing.assert_allclose(grad_batch, grad_rows, rtol=2e-6, atol=2e-6)
    empty = gaussian_heat_inverse(jnp.empty((0, 2), dtype=dtype), centers, biases, nu, time)
    assert empty.value.shape == (0, 2) and empty.converged.shape == (0,)


def _jaxpr_shapes(closed):
    """Walk nested control-flow/custom-JVP jaxprs and collect output shapes."""
    if hasattr(closed, "jaxpr"):
        yield from _jaxpr_shapes(closed.jaxpr)
    if hasattr(closed, "eqns"):
        for equation in closed.eqns:
            for variable in equation.outvars:
                if hasattr(variable.aval, "shape"):
                    yield tuple(variable.aval.shape)
            for value in equation.params.values():
                yield from _jaxpr_shapes(value)
    if isinstance(closed, (tuple, list)):
        for value in closed:
            yield from _jaxpr_shapes(value)
    if isinstance(closed, dict):
        for value in closed.values():
            yield from _jaxpr_shapes(value)


@pytest.mark.parametrize("adjoint", ["auto", "low_rank"])
def test_low_rank_gradient_never_materializes_dimension_squared_array(adjoint):
    dimension, count = 257, 3
    target = jnp.full(dimension, 0.001)
    centers = jnp.arange(count*dimension, dtype=jnp.float64).reshape(count, dimension) * 1e-5
    biases = jnp.array([0.0, 0.1, -0.1])
    gradient = jax.grad(lambda v: jnp.sum(gaussian_heat_inverse(v, centers, biases, adjoint=adjoint).value))
    shapes = list(_jaxpr_shapes(jax.make_jaxpr(gradient)(target)))
    assert not any(shape[-2:] == (dimension, dimension) for shape in shapes)
    assert any(shape[-2:] == (count, count) for shape in shapes)
    assert np.isfinite(jax.jit(gradient)(target)).all()


def test_low_rank_underflowed_probabilities_have_finite_higher_derivatives():
    centers = jnp.array([[0.0], [0.5], [-0.5]])
    biases = jnp.array([0.0, -1000.0, -1000.0])

    def value(v):
        return gaussian_heat_inverse(jnp.array([v]), centers, biases, adjoint="low_rank").value[0]

    assert float(value(0.2)) == 0.2
    np.testing.assert_allclose(jax.grad(value)(0.2), 1.0)
    np.testing.assert_allclose(jax.grad(jax.grad(value))(0.2), 0.0, atol=1e-14)


def test_single_center_and_redundant_centers_need_no_affine_rank():
    target = jnp.array([0.2, -0.3])
    centers = jnp.array([[4.0, -2.0]])
    result = gaussian_heat_inverse(target, centers, [100.0], time=3.0, max_iterations=0)
    assert bool(result.converged) and float(result.contraction_bound) == 0.0
    np.testing.assert_allclose(result.value, centers[0] + 3*target)
    repeated = gaussian_heat_inverse(target, jnp.repeat(centers, 4, axis=0), jnp.zeros(4), time=3.0)
    np.testing.assert_allclose(repeated.value, result.value)


@pytest.mark.parametrize("parameter,value", [(1, jnp.inf), (2, jnp.nan), (3, 0.0), (3, -1.0),
                                              (4, 0.0), (4, jnp.inf), (3, 1e308)])
def test_invalid_shared_numeric_data_refuses_without_claiming_infeasibility(parameter, value):
    args = list(_data())
    args[parameter] = jnp.full_like(args[parameter], value)
    result = jax.jit(gaussian_heat_inverse)(*args)
    assert not bool(result.converged)
    assert np.isnan(result.value).all() and np.isinf(result.residual_bound)


def test_invalid_targets_only_refuse_their_own_batch():
    target, centers, biases, nu, time = _data()
    targets = jnp.stack((target, jnp.array([jnp.nan, 0.0]), jnp.array([jnp.inf, 0.0])))
    result = jax.jit(gaussian_heat_inverse)(targets, centers, biases, nu, time)
    np.testing.assert_array_equal(result.converged, [True, False, False])
    assert np.isfinite(result.value[0]).all() and np.isnan(result.value[1:]).all()


def test_noncontractive_and_unconverged_cases_are_explicit():
    bad = gaussian_heat_inverse(jnp.array([0.2]), [[-2.0], [2.0]], [0.0, 0.0])
    assert float(bad.contraction_bound) >= 1.0
    assert not bool(bad.converged) and np.isnan(bad.value).all()
    early = gaussian_heat_inverse(*_data(), max_iterations=0)
    assert not bool(early.converged) and np.isnan(early.value).all()
    assert np.isfinite(early.residual_bound) and float(early.residual_bound) > 1e-4


@pytest.mark.parametrize("option,value,error", [("max_iterations", True, TypeError), ("max_iterations", -1, ValueError),
                                              ("max_iterations", 1.0, TypeError), ("adjoint", "other", ValueError),
                                              ("tolerance", 0.0, ValueError), ("tolerance", jnp.inf, ValueError),
                                              ("tolerance", True, TypeError)])
def test_static_control_validation(option, value, error):
    with pytest.raises(error):
        gaussian_heat_inverse(*_data(), **{option: value})


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, jnp.float16, jnp.complex64])
def test_target_dtype_validation(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        gaussian_heat_inverse(jnp.zeros(2, dtype=dtype), [[0.0, 0.0]], [0.0])


def test_structural_shapes_and_nonreal_parameter_validation():
    target, centers, biases, nu, time = _data()
    for args in ((target, centers[:, :1], biases, nu, time), (target, centers, biases[:1], nu, time),
                 (target, centers, biases, jnp.ones(2), time), (target, centers[:0], biases[:0], nu, time),
                 (jnp.array(1.0), centers, biases, nu, time)):
        with pytest.raises(ValueError):
            gaussian_heat_inverse(*args)
    for value in (True, 1j):
        with pytest.raises(TypeError, match="exclude bool"):
            gaussian_heat_inverse(target, centers, biases, value, time)


def test_residual_l2_scaling_prevents_false_convergence_from_squared_underflow():
    target = jnp.array([0.0], dtype=jnp.float32)
    centers = jnp.array([[0.0], [1e-30]], dtype=jnp.float32)
    result = gaussian_heat_inverse(target, centers, jnp.zeros(2, dtype=jnp.float32),
                                   max_iterations=0, tolerance=1e-35)
    assert not bool(result.converged)
    assert float(result.residual_bound) > 1e-31


def test_jitted_vmap_can_batch_live_mixture_parameters():
    target, centers, biases, nu, time = _data()
    mixtures = jnp.stack((centers, centers + jnp.array([0.2, -0.3])))
    mapped = jax.jit(jax.vmap(lambda a: gaussian_heat_inverse(target, a, biases, nu, time)))(mixtures)
    assert bool(jnp.all(mapped.converged))
    np.testing.assert_allclose(mapped.value[1] - mapped.value[0], [0.2, -0.3], rtol=2e-13, atol=2e-13)
    jacobian = jax.jit(jax.vmap(jax.jacfwd(lambda a: gaussian_heat_inverse(target, a, biases, nu, time).value)))(mixtures)
    assert np.isfinite(jacobian).all()
    assert jacobian.shape == (2, 2, 3, 2)


def test_overflowed_negative_logit_is_not_accepted_by_softmax_saturation():
    # The true numerical softmax of (0,-inf) is finite. Nevertheless the
    # -inf here came from overflow of finite inputs and is an arithmetic refusal.
    result = gaussian_heat_inverse(jnp.array([1e308]), [[0.0], [-10.0]], [0.0, 0.0], viscosity=100.0)
    assert not bool(result.converged)
    assert np.isnan(result.value).all() and np.isinf(result.residual_bound)
