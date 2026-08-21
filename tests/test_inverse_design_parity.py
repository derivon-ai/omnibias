# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax inverse-design G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_inverse import invert_input as jax_inv
from omnibias.jax.optim_inverse import worked_example as jax_ex
from omnibias.torch.optim_inverse import invert_input as torch_inv
from omnibias.torch.optim_inverse import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["x"] == jax_ex()["x"]
    t_rep = torch_inv(None, torch.tensor(0.5), torch.tensor(0.0))
    j_rep = jax_inv(None, jnp.asarray(0.5), jnp.asarray(0.0))
    assert t_rep.x == j_rep.x
    assert t_rep.residual == j_rep.residual
