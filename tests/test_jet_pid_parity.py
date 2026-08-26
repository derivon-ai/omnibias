# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX bit-parity for theory 08-10 jet-PID."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_pid import JetPIDConfig
from omnibias.jax.optim_pid import apply_pid_step as jax_apply
from omnibias.jax.optim_pid import jet_pid_step as jax_pid
from omnibias.jax.optim_pid import worked_example as jax_ex
from omnibias.torch.optim_pid import apply_pid_step as torch_apply
from omnibias.torch.optim_pid import jet_pid_step as torch_pid
from omnibias.torch.optim_pid import worked_example as torch_ex


def test_worked_example_is_shared() -> None:
    assert torch_ex() == jax_ex()
    assert torch_ex()["abs_step_err"] < 1e-15


def test_quadratic_restriction_parity() -> None:
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetPIDConfig(kp=0.5, ki=0.0, kd=0.0, s_max=2.0)

    def torch_loss(params: torch.Tensor) -> torch.Tensor:
        return (params[0] - 1.0) ** 2

    def jax_loss(params: jax.Array) -> jax.Array:
        return (params[0] - 1.0) ** 2

    t_params = torch.zeros(1)
    t_dir = torch.ones(1)
    j_params = jnp.zeros(1)
    j_dir = jnp.ones(1)
    t_rep = torch_pid(torch_loss, t_params, t_dir, config=cfg, order=2)
    j_rep = jax_pid(jax_loss, j_params, j_dir, config=cfg, order=2)
    assert t_rep.step == j_rep.step
    assert t_rep.proportional == j_rep.proportional
    assert t_rep.integral == j_rep.integral
    assert t_rep.derivative == j_rep.derivative
    assert t_rep.control == j_rep.control
    t_next = torch_apply(t_params, t_dir, t_rep)
    j_next = jax_apply(j_params, j_dir, j_rep)
    assert float(t_next[0]) == float(j_next[0])
    assert abs(float(t_next[0]) - 1.0) < 1e-15


def test_damped_pid_matches_newton_on_bowl() -> None:
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetPIDConfig(kp=1.0, ki=0.0, kd=0.5, s_max=2.0)

    def torch_loss(params: torch.Tensor) -> torch.Tensor:
        return (params[0] - 1.0) ** 2

    def jax_loss(params: jax.Array) -> jax.Array:
        return (params[0] - 1.0) ** 2

    t_rep = torch_pid(torch_loss, torch.zeros(1), torch.ones(1), config=cfg, order=2)
    j_rep = jax_pid(jax_loss, jnp.zeros(1), jnp.ones(1), config=cfg, order=2)
    assert t_rep.step == j_rep.step
    assert t_rep.step == 1.0
