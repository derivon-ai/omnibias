# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX Riccati-flow twin (09-10)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.core.riccati_flow import RiccatiFlowConfig
from omnibias.jax.architectures.riccati_flow import riccati_flow, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1() -> None:
    ex = worked_example()
    assert ex["err"] < 1e-12
    y = riccati_flow(jnp.asarray(0.25), config=RiccatiFlowConfig(t=1.0))
    assert abs(float(y) - ex["s"]) < 1e-12
