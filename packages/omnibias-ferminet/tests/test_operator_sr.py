# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator QGT on live real pytrees, complex amplitudes and dense equivalence."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from omnibias.ferminet.operator_sr import matrixfree_sr_step, qgt_operator


def logpsi(p, x):
    return p["b"] + jnp.dot(p["w"], x) + 1j * jnp.dot(p["phase"], x * x)


def test_complex_qgt_actions_and_sr_match_dense():
    params = {"b": jnp.asarray(0.2), "w": jnp.array([0.3, 0.4]), "phase": jnp.array([0.1, -0.2])}
    xs = jnp.array([[1.0, 2.0], [2.0, -1.0], [-1.0, 0.5], [0.0, 1.0]])
    weights = jnp.array([1.0, 2.0, 3.0, 4.0])
    w = weights / weights.sum()
    flat, unravel = ravel_pytree(params)

    def fn(t):
        return jax.vmap(lambda x: logpsi(unravel(t), x))(xs)

    jr = jax.jacrev(lambda t: jnp.real(fn(t)))(flat)
    ji = jax.jacrev(lambda t: jnp.imag(fn(t)))(flat)
    cr = jr - jnp.sum(w[:, None] * jr, axis=0)
    ci = ji - jnp.sum(w[:, None] * ji, axis=0)
    dense = cr.T @ (w[:, None] * cr) + ci.T @ (w[:, None] * ci)
    op = qgt_operator(
        logpsi, params, xs, weights=weights, chunk_size=2, sampling_kind="exact_enumeration"
    )
    v = jnp.arange(1.0, flat.size + 1)
    np.testing.assert_allclose(op.matvec(v), dense @ v, atol=1e-11)
    energies = jnp.array([1.0, 2.0, 0.7, 1.4])
    gradient = 2 * cr.T @ (w * energies)
    np.testing.assert_allclose(op.energy_gradient(energies), gradient, atol=1e-12)
    step = matrixfree_sr_step(
        logpsi,
        params,
        xs,
        energies,
        weights=weights,
        damping=0.2,
        learning_rate=0.1,
        chunk_size=2,
        rtol=1e-12,
    )
    assert step.accepted
    updated, _ = ravel_pytree(step.params)
    np.testing.assert_allclose(
        updated, flat - 0.1 * np.linalg.solve(dense + 0.2 * np.eye(flat.size), gradient), atol=1e-11
    )
    assert float(step.params["b"]) == pytest.approx(float(params["b"]))


def test_operator_jit_failure_nonupdate_and_large_dimension():
    p = {"w": jnp.arange(16, dtype=float) / 100, "b": jnp.asarray(0.1)}
    xs = jnp.arange(128, dtype=float).reshape(8, 16) / 100

    def f(p, x):
        return p["b"] + jnp.dot(p["w"], jnp.sin(x))

    energies = jnp.arange(8, dtype=float)
    step = jax.jit(lambda p: matrixfree_sr_step(f, p, xs, energies, rtol=1e-12, max_iterations=1))(
        p
    )
    if not step.accepted:
        np.testing.assert_array_equal(step.params["w"], p["w"])
    # 10k parameters, 256 scalar observations, no parameter-square allocation.
    large = jnp.zeros(10000)
    data = jnp.arange(256, dtype=float)[:, None] / 256

    def model(p, x):
        return jnp.dot(p, jnp.sin(jnp.arange(p.size, dtype=p.dtype) * 0.0001 + x[0]))

    op = qgt_operator(model, large, data, chunk_size=32)
    product = op.matvec(jnp.ones_like(large))
    assert product.shape == large.shape and jnp.all(jnp.isfinite(product))


def test_invalid_weights_and_parameter_contract():
    with pytest.raises(ValueError, match="weights"):
        qgt_operator(
            lambda p, x: p[0] * x[0], jnp.ones(1), jnp.ones((2, 1)), weights=jnp.array([1.0, -1.0])
        )
