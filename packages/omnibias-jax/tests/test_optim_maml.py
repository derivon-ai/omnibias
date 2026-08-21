# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX exact-MAML twin (09-16)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.jax.optim_maml import (
    ift_dtheta_dalpha,
    inner_newton_quadratic,
    quadratic_worked_example,
)

jax.config.update("jax_enable_x64", True)


def test_g1_quadratic() -> None:
    theta = jnp.asarray(2.5)
    alpha = jnp.asarray(-0.3)
    theta_p = inner_newton_quadratic(theta, alpha)
    assert abs(float(theta_p - alpha)) < 1e-12
    ift = ift_dtheta_dalpha(theta, alpha)
    assert abs(float(ift) - 1.0) < 1e-12
    assert quadratic_worked_example()["ift_err"] < 1e-10
