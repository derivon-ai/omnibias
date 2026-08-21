# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX Collapse-Net twin (09-11)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.core.collapse_net import CollapseNetConfig
from omnibias.jax.architectures.collapse_net import (
    collapse_net_forward,
    collapse_remainder,
    worked_example,
)

jax.config.update("jax_enable_x64", True)


def test_g1() -> None:
    ex = worked_example()
    assert ex["rel_err"] < 1e-6
    cfg = CollapseNetConfig(order=1, delta=0.1, mode="collapsed")
    y = collapse_net_forward(jnp.asarray(0.0), config=cfg)
    assert abs(float(y) - 0.25) < 1e-12
    rem = collapse_remainder(jnp.asarray(0.0), config=cfg)
    assert abs(float(rem) - ex["remainder"]) < 1e-15
