# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT, vmap, analytic and parameter-gradient checks of PDE ridge fields."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.ridge_system import RidgeSystem, RidgeSyzygy
from omnibias.jax.ridge_system import ridge_solution

jax.config.update("jax_enable_x64", True)


def _wave_pack(sign: int = 1) -> RidgeSyzygy:
    system = RidgeSystem.from_operators(
        [
            [OperatorSpec.polynomial({(1, 0): -1}), OperatorSpec.mixed((0, 0)), None],
            [OperatorSpec.polynomial({(0, 1): -1}), None, OperatorSpec.mixed((0, 0))],
            [None, OperatorSpec.mixed((1, 0)), OperatorSpec.polynomial({(0, 1): -4})],
        ],
        (2 * sign, 1),
    )
    pack, = system.kernel_basis(max_degree=1)
    return pack


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "softplus", "gaussian", "exp",
                                         "sin", "cos", "sinh", "cosh"])
def test_jit_vmap_and_spatial_equations(activation: str) -> None:
    pack = _wave_pack()
    x = jnp.array([[0.1, -0.4], [-0.2, 0.7]])

    def field(point, bias):
        return ridge_solution(point, pack, bias, activation=activation)

    if activation == "gaussian":
        def base(z):
            return jnp.exp(-z**2 / 2)
    elif activation in ("sigmoid", "softplus"):
        base = getattr(jax.nn, activation)
    else:
        base = getattr(jnp, activation)
    z = 2 * x[:, 0] + x[:, 1] + 0.13
    derivative = jax.vmap(jax.grad(base))(z)
    expected = jnp.stack((base(z), 2 * derivative, derivative), axis=-1)
    actual = jax.jit(field)(x, jnp.array(0.13))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.vmap(field, in_axes=(0, None)))(x, 0.13), actual,
                               rtol=1e-12, atol=1e-12)
    jac = jax.jit(jax.vmap(jax.jacfwd(field), in_axes=(0, None)))(x, 0.13)
    residual = jnp.stack((actual[:, 1] - jac[:, 0, 0], actual[:, 2] - jac[:, 0, 1],
                          jac[:, 1, 0] - 4 * jac[:, 2, 1]), axis=-1)
    np.testing.assert_allclose(residual, 0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("activation", ["tanh", "sin", "exp"])
def test_jit_trainable_bias_and_scalar_readout(activation: str) -> None:
    x = jnp.array([[0.1, -0.4], [-0.2, 0.7]])
    packs = (_wave_pack(1), _wave_pack(-1))

    def actual_loss(bias, weights):
        value = sum(weights[i] * ridge_solution(x, pack, bias[i], activation=activation)
                    for i, pack in enumerate(packs))
        return jnp.sum(value**2)

    def reference_loss(bias, weights):
        value = jnp.zeros((2, 3))
        base = getattr(jnp, activation)
        for i, sign in enumerate((1, -1)):
            z = 2 * sign * x[:, 0] + x[:, 1] + bias[i]
            derivative = jax.vmap(jax.grad(base))(z)
            value = value + weights[i] * jnp.stack((base(z), 2 * sign * derivative, derivative), axis=-1)
        return jnp.sum(value**2)

    bias, weights = jnp.array([0.13, -0.17]), jnp.array([0.7, -0.4])
    actual = jax.jit(jax.grad(actual_loss, argnums=(0, 1)))(bias, weights)
    expected = jax.grad(reference_loss, argnums=(0, 1))(bias, weights)
    for value, reference in zip(actual, expected, strict=True):
        assert np.isfinite(value).all() and np.count_nonzero(value) == value.size
        np.testing.assert_allclose(value, reference, rtol=1e-11, atol=1e-12)


def test_third_derivative_uses_raw_coefficients() -> None:
    system = RidgeSystem.from_operators(
        [[OperatorSpec.mixed((3,)), OperatorSpec.mixed((0,))]], (Fraction(3, 2),)
    )
    pack, = system.kernel_basis(max_degree=3)
    def field(point):
        return ridge_solution(point, pack, 0.1, activation="sin")
    x = jnp.array([[-0.2], [0.1], [0.4]])
    result = jax.jit(field)(x)
    z = 1.5 * x[:, 0] + 0.1
    np.testing.assert_allclose(result, jnp.stack((-8 / 27 * jnp.sin(z), -jnp.cos(z)), axis=-1))
    third = jax.jacfwd(jax.jacfwd(jax.jacfwd(lambda point: field(point)[0])))
    derivative = jax.jit(jax.vmap(third))(x)[:, 0, 0, 0]
    np.testing.assert_allclose(derivative + result[:, 1], 0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_shapes_dtype_and_scalar_batch(dtype) -> None:
    pack = _wave_pack()
    x = jnp.zeros((2, 3, 2), dtype=dtype)
    result = ridge_solution(x, pack)
    assert result.shape == (2, 3, 3) and result.dtype == dtype
    assert ridge_solution(x[0, 0], pack).shape == (3,)
    assert ridge_solution(x[:0], pack).shape == (0, 3, 3)


def test_invalid_inputs_and_nonsmooth_profiles() -> None:
    pack = _wave_pack()
    for x in (jnp.zeros(()), jnp.zeros(3)):
        with pytest.raises(ValueError, match="shape"):
            ridge_solution(x, pack)
    with pytest.raises(TypeError, match="real floating"):
        ridge_solution(jnp.zeros(2, dtype=jnp.int64), pack)
    with pytest.raises(ValueError, match="scalar"):
        ridge_solution(jnp.zeros((2, 2)), pack, jnp.zeros(2))
    for activation in ("relu", "elu", "abs", "softsign"):
        with pytest.raises(ValueError, match="globally smooth"):
            ridge_solution(jnp.zeros(2), pack, activation=activation)
