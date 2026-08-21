# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax Collapse-Net G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.core.collapse_net import CollapseNetConfig
from omnibias.jax.architectures.collapse_net import collapse_net_forward as jax_fwd
from omnibias.jax.architectures.collapse_net import collapse_remainder as jax_rem
from omnibias.jax.architectures.collapse_net import worked_example as jax_ex
from omnibias.torch.architectures.collapse_net import collapse_net_forward as torch_fwd
from omnibias.torch.architectures.collapse_net import collapse_remainder as torch_rem
from omnibias.torch.architectures.collapse_net import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["remainder"] == jax_ex()["remainder"]
    cfg = CollapseNetConfig(order=1, delta=0.1, mode="stencil")
    t = torch_fwd(torch.tensor(0.0), config=cfg)
    j = jax_fwd(jnp.asarray(0.0), config=cfg)
    assert float(t) == float(j)
    tr = torch_rem(torch.tensor(0.0), config=cfg)
    jr = jax_rem(jnp.asarray(0.0), config=cfg)
    assert float(tr) == float(jr)
