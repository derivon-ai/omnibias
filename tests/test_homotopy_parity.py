# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax homotopy G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_homotopy import homotopy_step as jax_step
from omnibias.jax.optim_homotopy import worked_example as jax_ex
from omnibias.torch.optim_homotopy import homotopy_step as torch_step
from omnibias.torch.optim_homotopy import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["theta"] == jax_ex()["theta"]
    t_trial, t_res, t_dec = torch_step(torch.tensor(1.0), 0.1)
    j_trial, j_res, j_dec = jax_step(jnp.asarray(1.0), 0.1)
    assert t_trial == j_trial
    assert t_res == j_res
    assert t_dec.accepted == j_dec.accepted
