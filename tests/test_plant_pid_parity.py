# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX bit-parity for theory 09-29 plant PID."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.core.pid_layer import PlantPIDConfig
from omnibias.core.pid_layer import plant_pid as core_pid
from omnibias.jax.plant_pid import plant_pid as jax_pid
from omnibias.jax.plant_pid import worked_example as jax_ex
from omnibias.torch.plant_pid import plant_pid as torch_pid
from omnibias.torch.plant_pid import worked_example as torch_ex


def test_worked_example_is_shared() -> None:
    assert torch_ex() == jax_ex()
    assert float(torch_ex()["ftc_residual"]) < 1e-8


def test_scalar_parity() -> None:
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = PlantPIDConfig(
        kp=0.5, ki=1.0, kd=0.25, setpoint=0.4, alpha=1.25, beta=-0.2, t0=-0.5
    )
    core = core_pid(0.0, config=cfg)
    t_rep = torch_pid(torch.tensor(0.0), config=cfg)
    j_rep = jax_pid(jnp.asarray(0.0), config=cfg)
    assert t_rep.control == j_rep.control == core.control
    assert t_rep.integral == j_rep.integral == core.integral
    assert t_rep.derivative == j_rep.derivative == core.derivative
