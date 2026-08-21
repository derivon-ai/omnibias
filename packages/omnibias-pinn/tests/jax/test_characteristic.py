# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX Characteristic-Net twin (09-08)."""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.characteristic import constant_v, gaussian_u0  # noqa: E402
from omnibias.pinn.jax.characteristic import characteristic_eval, worked_example  # noqa: E402


def test_g1() -> None:
    ex = worked_example()
    assert ex["err"] < 1e-12
    u, crossed = characteristic_eval(jnp.asarray(0.0), 0.2, constant_v, gaussian_u0)
    assert abs(float(u) - math.exp(-0.04)) < 1e-12
    assert float(crossed) == 0.0
