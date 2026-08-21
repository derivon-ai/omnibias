# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX integral-kernel twin (09-14)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.core.integral_kernel import integral_cell as core_cell
from omnibias.jax.architectures.integral_kernel import integral_cell, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1() -> None:
    ex = worked_example()
    assert ex["g1_err"] < 1e-12
    z = jnp.asarray([0.5])
    cell = float(integral_cell(z, -0.5, 0.5)[0])
    assert abs(cell - core_cell(0.5, -0.5, 0.5)) < 1e-12
