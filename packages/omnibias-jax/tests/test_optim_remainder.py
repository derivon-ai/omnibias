# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX remainder-training twin (09-18)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.jax.optim_remainder import remainder_loss, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1() -> None:
    ex = worked_example()
    assert ex["R2_err"] < 1e-12
    report = remainder_loss(
        jnp.asarray([ex["exp"]]),
        jnp.asarray([1.0, 1.0, 1.0]),
        jnp.asarray([0.2]),
    )
    assert abs(float(report["max_abs"]) - ex["R2"]) < 1e-12
