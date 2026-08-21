# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax Pack-MoE gates are bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.core.pack_moe import ExpertWindow, PackMoEConfig
from omnibias.jax.architectures.pack_moe import pack_moe_forward as jax_fwd
from omnibias.torch.architectures.pack_moe import pack_moe_forward as torch_fwd

jax.config.update("jax_enable_x64", True)


def test_g4_worked_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    windows = (ExpertWindow(-0.2, 0.0), ExpertWindow(0.0, 0.2))
    cfg = PackMoEConfig(router="band")
    t = torch_fwd(torch.tensor(0.0), (1.0, 3.0), windows, config=cfg)
    j = jax_fwd(jnp.asarray(0.0), (1.0, 3.0), windows, config=cfg)
    assert float(t) == float(j)
