# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX bit-parity for theory 08-12 jet-MPC."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_mpc import JetMPCConfig
from omnibias.jax.optim_mpc import apply_mpc_step as jax_apply
from omnibias.jax.optim_mpc import jet_mpc_step as jax_mpc
from omnibias.jax.optim_mpc import worked_example as jax_ex
from omnibias.torch.optim_mpc import apply_mpc_step as torch_apply
from omnibias.torch.optim_mpc import jet_mpc_step as torch_mpc
from omnibias.torch.optim_mpc import worked_example as torch_ex


def test_worked_example_is_shared() -> None:
    assert torch_ex() == jax_ex()
    assert torch_ex()["newton_recovered"] is True


def test_quadratic_restriction_parity() -> None:
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetMPCConfig(q=0.0, r=0.0, qf=1.0, horizon=1, u_max=4.0)

    def torch_loss(params: torch.Tensor) -> torch.Tensor:
        return (params[0] - 1.0) ** 2

    def jax_loss(params: jax.Array) -> jax.Array:
        return (params[0] - 1.0) ** 2

    t_params = torch.zeros(1)
    t_dir = torch.ones(1)
    j_params = jnp.zeros(1)
    j_dir = jnp.ones(1)
    t_rep = torch_mpc(torch_loss, t_params, t_dir, config=cfg)
    j_rep = jax_mpc(jax_loss, j_params, j_dir, config=cfg)
    assert t_rep.step == j_rep.step
    assert t_rep.unconstrained_step == j_rep.unconstrained_step
    assert t_rep.gain == j_rep.gain
    assert t_rep.receding is True
    t_next = torch_apply(t_params, t_dir, t_rep)
    j_next = jax_apply(j_params, j_dir, j_rep)
    assert float(t_next[0]) == float(j_next[0])
    assert abs(float(t_next[0]) - 1.0) < 1e-15
