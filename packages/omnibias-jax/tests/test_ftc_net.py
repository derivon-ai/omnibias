# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX FTC-Net twin (theory 09-03 / 09-17)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.core.ftc import worked_example
from omnibias.jax.architectures.ftc_net import FTCNet, FTCNetConfig, ftc_block

jax.config.update("jax_enable_x64", True)


def test_g1_matches_core() -> None:
    x = jnp.asarray(0.0)
    w = jnp.asarray(1.0)
    lo = jnp.asarray(-0.1)
    hi = jnp.asarray(0.1)
    integral, deriv, collapse = ftc_block(x, w, lo, hi)
    ex = worked_example()
    assert float(integral) == ex["I"]
    assert float(deriv) == ex["dI"]
    assert float(collapse) == ex["collapse"]
    assert FTCNet(FTCNetConfig()).config.window == 0.6
