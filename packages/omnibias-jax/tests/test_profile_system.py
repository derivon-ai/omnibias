# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT, batching and independent analytic checks of annihilator profiles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.profile_system import ProfilePack, ProfileSystem
from omnibias.jax.profile_system import profile_solution

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("equation", ["heat", "laplace"])
def test_compiled_sine_profile_solves_pde(equation: str) -> None:
    operator = (OperatorSpec.polynomial({(1, 0): 1, (0, 2): -1})
                if equation == "heat" else OperatorSpec.laplacian())
    carrier = (-1, 0) if equation == "heat" else (1, 0)
    system = ProfileSystem.from_operators([[operator]], (0, 1), annihilator=(1, 0, 1), carrier=carrier)
    result = system.solve()
    assert result.consistent
    pack = result.homogeneous_basis[0]
    x = jnp.array([[0.2, -0.4], [0.3, 0.7]])

    def field(point):
        return profile_solution(point, pack, [0.0, 1.0])[..., 0]

    value = jax.jit(field)(x)
    np.testing.assert_allclose(value, jnp.exp(carrier[0] * x[:, 0]) * jnp.sin(x[:, 1]), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(jax.jit(jax.vmap(field))(x), value, rtol=2e-12, atol=2e-12)
    gradient = jax.jit(jax.vmap(jax.jacfwd(field)))(x)
    hessian = jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(field))))(x)
    residual = (gradient[:, 0] - hessian[:, 1, 1] if equation == "heat"
                else hessian[:, 0, 0] + hessian[:, 1, 1])
    np.testing.assert_allclose(residual, 0, rtol=0, atol=2e-12)


def test_repeated_root_and_live_initial_derivatives() -> None:
    pack = ProfilePack((1,), (0,), (1, -2, 1), ((1, 0), (0, 1)))
    x = jnp.array([[-0.4], [0.0], [0.6]])

    def actual(initial, bias, weight):
        return weight * profile_solution(x, pack, initial, bias)

    def expected(initial, bias, weight):
        z = x[:, 0] + bias
        polynomial = initial[0] + (initial[1] - initial[0]) * z
        return weight * jnp.stack((polynomial * jnp.exp(z),
                                   (polynomial + initial[1] - initial[0]) * jnp.exp(z)), axis=-1)

    args = jnp.array([0.2, 0.7]), jnp.array(0.13), jnp.array(0.8)
    np.testing.assert_allclose(jax.jit(actual)(*args), expected(*args), rtol=2e-12, atol=2e-12)
    actual_grad = jax.jit(jax.grad(lambda *p: jnp.sum(actual(*p)**2), argnums=(0, 1, 2)))(*args)
    expected_grad = jax.grad(lambda *p: jnp.sum(expected(*p)**2), argnums=(0, 1, 2))(*args)
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        assert np.isfinite(value).all() and np.count_nonzero(value) == value.size
        np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)
    canonical = profile_solution(x, pack, [0.0, 1.0])
    np.testing.assert_allclose(canonical[:, 0], x[:, 0] * jnp.exp(x[:, 0]), rtol=2e-12, atol=2e-12)


def test_repeated_oscillatory_root_forced_poisson() -> None:
    pack = ProfilePack((0, 1), (1, 0), (1, 0, 2, 0, 1), ((1,), (0,), (0,), (0,)))
    x = jnp.array([[0.2, 0.0], [-0.2, 0.4], [0.4, -0.3]])

    def field(point):
        return profile_solution(point, pack, [0.0, 0.0, 0.0, 1.0])[..., 0]

    value = jax.jit(field)(x)
    y = x[:, 1]
    expected = jnp.exp(x[:, 0]) * (jnp.sin(y) - y * jnp.cos(y)) / 2
    np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
    hessian = jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(field))))(x)
    np.testing.assert_allclose(jnp.trace(hessian, axis1=-2, axis2=-1), jnp.exp(x[:, 0]) * jnp.sin(y),
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(value[0], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(jax.jacfwd(field)(x[0])[1], 0, rtol=0, atol=1e-14)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_exponential_profile_shapes_dtype(dtype) -> None:
    pack = ProfilePack((1,), (0,), (-1, 1), ((1,),))
    x = jnp.full((2, 3, 1), 0.2, dtype=dtype)
    value = profile_solution(x, pack, [2.0])
    assert value.shape == (2, 3, 1) and value.dtype == dtype
    np.testing.assert_allclose(value, 2 * jnp.exp(x), rtol=2e-6)
    assert profile_solution(x[0, 0], pack, [2.0]).shape == (1,)
    assert profile_solution(x[:0], pack, [2.0]).shape == (0, 3, 1)


def test_invalid_inputs() -> None:
    pack = ProfilePack((1,), (0,), (1, 0, 1), ((1,), (0,)))
    for x in (jnp.zeros(()), jnp.zeros(2)):
        with pytest.raises(ValueError, match="shape"):
            profile_solution(x, pack, [0.0, 1.0])
    for dtype in (jnp.int64, jnp.float16, jnp.complex64):
        with pytest.raises(TypeError, match="float32 or float64"):
            profile_solution(jnp.zeros(1, dtype=dtype), pack, [0.0, 1.0])
    for initial in ([1.0], [[0.0, 1.0]]):
        with pytest.raises(ValueError, match="initial_jet"):
            profile_solution(jnp.zeros(1), pack, initial)
    with pytest.raises(ValueError, match="scalar"):
        profile_solution(jnp.zeros(1), pack, [0.0, 1.0], jnp.zeros(1))
