# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Both backends obey the weak derivative identity and retain gradients."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.ridge_distribution import RidgeDistribution
from omnibias.core.ridge_system import RidgeSystem
from omnibias.jax.ridge_distribution import gaussian_weak as jweak
from omnibias.torch.ridge_distribution import gaussian_weak as tweak


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixed_regular_and_singular_profiles_parameter_parity(dtype) -> None:
    field = RidgeDistribution((2, -1), ((-5, (1, 2)), (-2, (3, -1)),
                                        (0, (1, 0)), (3, (-1, 2))), outputs=2)
    mean = np.array([[-0.3], [0.5]], dtype=dtype)
    variance = np.array([[0.8, 1.2]], dtype=dtype)
    tm = torch.tensor(mean, requires_grad=True)
    tv = torch.tensor(variance, requires_grad=True)
    tb = torch.tensor(0.1, dtype=tm.dtype, requires_grad=True)
    value = tweak(field, tm, tv, tb)
    jforward = jax.jit(lambda m, v, b: jweak(field, m, v, b))
    jm, jv, jb = jnp.asarray(mean), jnp.asarray(variance), jnp.asarray(0.1, dtype=dtype)
    tol = 5e-5 if dtype is np.float32 else 3e-12
    np.testing.assert_allclose(value.detach(), jforward(jm, jv, jb), rtol=tol, atol=tol)
    tg = torch.autograd.grad(value.square().sum(), (tm, tv, tb))
    jg = jax.jit(jax.grad(lambda m, v, b: jnp.sum(jforward(m, v, b)**2),
                          argnums=(0, 1, 2)))(jm, jv, jb)
    for a, b in zip(tg, jg, strict=True):
        np.testing.assert_allclose(a.detach(), b, rtol=tol, atol=tol)


@pytest.mark.parametrize("index", [-4, -1, 0, 1, 4])
def test_weak_derivative_equals_bias_derivative(index) -> None:
    field = RidgeDistribution((2,), ((index, (1,)),))
    dx = RidgeSystem.from_operators(((OperatorSpec.mixed((1,)),),), (2,))
    derivative = field.apply(dx)
    mean = torch.tensor(0.3, dtype=torch.float64)
    bias = torch.tensor(-0.1, dtype=mean.dtype, requires_grad=True)
    expected = 2 * torch.autograd.grad(tweak(field, mean, 0.7, bias).sum(), bias)[0]
    actual = tweak(derivative, mean, 0.7, bias).sum()
    assert actual.item() == pytest.approx(expected.item(), rel=2e-11, abs=2e-12)
    jb = jnp.asarray(-0.1)
    expected_j = 2 * jax.grad(lambda b: jweak(field, jnp.asarray(0.3), 0.7, b).sum())(jb)
    np.testing.assert_allclose(jweak(derivative, jnp.asarray(0.3), 0.7, jb).sum(), expected_j,
                               rtol=2e-11, atol=2e-12)
