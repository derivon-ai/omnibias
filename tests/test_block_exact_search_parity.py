# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch/JAX parity for block exact search (theory 08-07 G4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_block_search import (
    block_exact_search as jax_search,
)
from omnibias.torch.optim_block_search import (
    block_exact_search as torch_search,
)

jax.config.update("jax_enable_x64", True)


def _torch_loss(params: torch.Tensor) -> torch.Tensor:
    p = params.reshape(-1)
    return (p[0] * 1.0 + p[1] * 0.5 - 1.0) ** 2


def _jax_loss(params: jax.Array) -> jax.Array:
    p = jnp.reshape(params, (-1,))
    return (p[0] * 1.0 + p[1] * 0.5 - 1.0) ** 2


def test_g2_parity() -> None:
    torch.set_default_dtype(torch.float64)
    t_new, t_res = torch_search(
        _torch_loss, torch.tensor([0.0, 0.0]), mask=(True, False), exact_quadratic=True
    )
    j_new, j_res = jax_search(
        _jax_loss,
        jnp.array([0.0, 0.0], dtype=jnp.float64),
        mask=(True, False),
        exact_quadratic=True,
    )
    assert abs(t_res.step - j_res.step) <= 1e-12
    assert abs(float(t_new[0]) - float(j_new[0])) <= 1e-12
    assert t_res.fell_back is False
    assert j_res.fell_back is False
    assert abs(t_res.step - 1.0) <= 1e-12
