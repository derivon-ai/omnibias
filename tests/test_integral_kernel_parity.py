# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax integral-kernel G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.architectures.integral_kernel import integral_cell as jax_cell
from omnibias.jax.architectures.integral_kernel import worked_example as jax_ex
from omnibias.torch.architectures.integral_kernel import integral_cell as torch_cell
from omnibias.torch.architectures.integral_kernel import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["g1_cell"] == jax_ex()["g1_cell"]
    t = float(torch_cell(torch.tensor([0.5]), -0.5, 0.5)[0])
    j = float(jax_cell(jnp.asarray([0.5]), -0.5, 0.5)[0])
    assert t == j
