# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch/JAX parity for sharpness-scheduled step (theory 08-06 G3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_sharpness import SharpnessSchedule
from omnibias.jax.optim_sharpness import sharpness_lambda_max as jax_lambda_max
from omnibias.jax.optim_sharpness import sharpness_scheduled_step as jax_step
from omnibias.torch.optim_sharpness import sharpness_lambda_max as torch_lambda_max
from omnibias.torch.optim_sharpness import sharpness_scheduled_step as torch_step

jax.config.update("jax_enable_x64", True)

STIFF = 1.0e4
SCHEDULE = SharpnessSchedule(n_lanczos=4, c=1e-3, ell_min=1e-6)


def _torch_loss(params: torch.Tensor) -> torch.Tensor:
    p = params.reshape(-1)
    return 0.5 * (p[0] * p[0] + STIFF * p[1] * p[1])


def _jax_loss(params: jax.Array) -> jax.Array:
    p = jnp.reshape(params, (-1,))
    return 0.5 * (p[0] * p[0] + STIFF * p[1] * p[1])


def test_lambda_max_parity() -> None:
    torch.set_default_dtype(torch.float64)
    t_ell = torch_lambda_max(_torch_loss, torch.tensor([1.0, 1.0]), n_lanczos=4)
    j_ell = jax_lambda_max(
        _jax_loss, jnp.array([1.0, 1.0], dtype=jnp.float64), n_lanczos=4
    )
    rel = abs(t_ell - j_ell) / max(abs(t_ell), abs(j_ell), 1.0)
    assert rel <= 1e-10
    assert abs(t_ell - STIFF) / STIFF <= 1e-10


def test_scheduled_step_parity() -> None:
    torch.set_default_dtype(torch.float64)
    t_new, t_rep = torch_step(
        _torch_loss, torch.tensor([1.0, 1.0]), schedule=SCHEDULE
    )
    j_new, j_rep = jax_step(
        _jax_loss, jnp.array([1.0, 1.0], dtype=jnp.float64), schedule=SCHEDULE
    )
    rel_ell = abs(t_rep.ell_k - j_rep.ell_k) / max(abs(t_rep.ell_k), 1.0)
    assert rel_ell <= 1e-10
    assert abs(t_rep.scheduled - j_rep.scheduled) <= 1e-10 * max(
        abs(t_rep.scheduled), 1.0
    )
    gap = torch.max(torch.abs(t_new - torch.tensor([float(v) for v in j_new])))
    assert float(gap) <= 1e-10
