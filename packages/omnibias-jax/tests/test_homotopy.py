# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX homotopy twin (09-20)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.jax.optim_homotopy import homotopy_step, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1() -> None:
    ex = worked_example()
    assert ex["accepted"] is True
    trial, residual, decision = homotopy_step(jnp.asarray(1.0), 0.1)
    assert decision.accepted is True
    assert residual < 1e-10
    assert abs(trial - float(ex["theta"])) < 1e-12
