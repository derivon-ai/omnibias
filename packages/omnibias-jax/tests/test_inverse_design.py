# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX inverse-design twin (09-22)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.jax.optim_inverse import invert_input, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1() -> None:
    ex = worked_example()
    assert ex["residual"] < 1e-12
    report = invert_input(None, jnp.asarray(0.5), jnp.asarray(0.0))
    assert abs(report.residual) < 1e-12
    assert abs(report.x - ex["x"]) < 1e-12
