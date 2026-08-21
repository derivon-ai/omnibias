# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX Pack-MoE twin (09-07)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.core.pack_moe import ExpertWindow, PackMoEConfig
from omnibias.jax.architectures.pack_moe import pack_moe_forward, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1_worked() -> None:
    y = pack_moe_forward(
        jnp.asarray(0.0),
        (1.0, 3.0),
        (ExpertWindow(-0.2, 0.0), ExpertWindow(0.0, 0.2)),
        config=PackMoEConfig(router="band"),
    )
    assert abs(float(y) - 2.0) < 1e-12
    assert worked_example()["g_a_err"] < 1e-12
