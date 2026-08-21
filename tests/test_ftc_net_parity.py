# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax FTC-Net are bit-identical on the 09-03 worked cell."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.architectures.ftc_net import ftc_block as jax_ftc_block
from omnibias.torch.architectures.ftc_net import ftc_block as torch_ftc_block

jax.config.update("jax_enable_x64", True)


def test_g4_worked_example_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    ti, td, tc = torch_ftc_block(
        torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-0.1), torch.tensor(0.1)
    )
    ji, jd, jc = jax_ftc_block(
        jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(-0.1), jnp.asarray(0.1)
    )
    assert float(ti) == float(ji)
    assert float(td) == float(jd)
    assert float(tc) == float(jc)
