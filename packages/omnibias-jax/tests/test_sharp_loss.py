# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX sharpness-loss twin (09-23)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.jax.optim_sharp_loss import sharpness_augmented_loss, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1() -> None:
    ex = worked_example()
    assert ex["hvp"] == 10.0
    assert sharpness_augmented_loss(None, jnp.asarray(0.0)) == 1.0

    def loss_fn(params: jax.Array) -> jax.Array:
        return 5.0 * jnp.reshape(params, (-1,))[0] ** 2

    aug = sharpness_augmented_loss(loss_fn, jnp.asarray([0.0]))
    assert abs(aug - 1.0) < 1e-12
