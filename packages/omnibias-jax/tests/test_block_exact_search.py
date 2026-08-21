# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Block exact search, JAX (theory 08-07)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.jax.optim_block_search import block_exact_search, last_linear_block

jax.config.update("jax_enable_x64", True)


def _section5_loss(params: Array) -> Array:
    p = jnp.reshape(params, (-1,))
    return (p[0] * 1.0 + p[1] * 0.5 - 1.0) ** 2


def test_g2_exact_quadratic() -> None:
    v = jnp.array([0.0, 0.0], dtype=jnp.float64)
    new, result = block_exact_search(
        _section5_loss,
        v,
        mask=(True, False),
        exact_quadratic=True,
    )
    assert result.fell_back is False
    assert abs(result.step - 1.0) <= 1e-12
    assert abs(float(new[0]) - 1.0) <= 1e-12
    assert abs(float(new[1])) <= 1e-12
    assert result.actual_value is not None
    assert abs(result.actual_value) <= 1e-12


def test_last_linear_block_decreases() -> None:
    hidden = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    target = jnp.array([1.0, -1.0], dtype=jnp.float64)

    def loss(params: Array) -> Array:
        r = hidden @ jnp.reshape(params, (-1,)) - target
        return 0.5 * jnp.dot(r, r)

    v0 = jnp.zeros(2, dtype=jnp.float64)
    new, result = block_exact_search(
        loss, v0, spec=last_linear_block(2), exact_quadratic=True
    )
    assert float(loss(new)) <= float(loss(v0))
    assert result.fell_back is False
