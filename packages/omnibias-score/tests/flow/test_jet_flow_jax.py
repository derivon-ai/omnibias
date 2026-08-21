# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX twin for theory 09-06."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from omnibias.score.flow.jax.jet_flow import jet_flow_forward, jet_flow_inverse

jax.config.update("jax_enable_x64", True)


def test_g1_jax_log_det_and_inverse() -> None:
    y, log_det = jet_flow_forward(jnp.asarray(0.3), jnp.asarray(2.0))
    closed = math.log(2.0 * (1.0 - math.tanh(0.6) ** 2))
    assert abs(float(log_det) - closed) < 1e-12
    z = jet_flow_inverse(y, jnp.asarray(2.0))
    assert abs(float(z) - 0.3) < 1e-10
