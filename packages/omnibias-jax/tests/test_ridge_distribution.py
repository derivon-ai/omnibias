# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Compiled weak profiles with bias/variance gradients and batching."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.core.ridge_distribution import RidgeDistribution
from omnibias.jax.ridge_distribution import gaussian_weak

jax.config.update("jax_enable_x64", True)


def test_jit_vmap_and_gaussian_density_gradient_oracle() -> None:
    field = RidgeDistribution((1,), ((-1, (1,)),))
    m = jnp.array([-0.4, 0.2, 0.9])
    variance, bias = jnp.array(0.7), jnp.array(0.13)
    forward = jax.jit(lambda mean, v, b: gaussian_weak(field, mean, v, b)[..., 0])
    actual = forward(m, variance, bias)
    expected = jnp.exp(-((m+bias)**2)/(2*variance))/jnp.sqrt(2*jnp.pi*variance)
    np.testing.assert_allclose(actual, expected, rtol=2e-14)
    np.testing.assert_allclose(jax.jit(jax.vmap(forward, in_axes=(0, None, None)))(m, variance, bias), actual)
    gradient = jax.jit(jax.grad(lambda v, b: forward(m, v, b).sum(), argnums=(0, 1)))(variance, bias)
    np.testing.assert_allclose(gradient[0], jnp.sum(expected*((m+bias)**2-variance)/(2*variance**2)))
    np.testing.assert_allclose(gradient[1], jnp.sum(-expected*(m+bias)/variance))


def test_invalid_variance_and_scalar_metadata() -> None:
    field = RidgeDistribution((1,), ((-1, (1,)),))
    values = jax.jit(lambda v: gaussian_weak(field, jnp.zeros(4), v))(
        jnp.array([0., -1., jnp.inf, 1.]))
    assert np.isnan(values[:3]).all() and np.isfinite(values[3]).all()
    with pytest.raises(TypeError):
        gaussian_weak(field, jnp.array([1]), 1.)
    with pytest.raises(ValueError, match="scalar"):
        gaussian_weak(field, jnp.zeros(2), 1., jnp.zeros(2))
