# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax Frame-UNet G1 values are bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.architectures.frame_unet import frame_unet_forward as jax_fwd
from omnibias.jax.architectures.frame_unet import worked_example as jax_ex
from omnibias.torch.architectures.frame_unet import frame_unet_forward as torch_fwd
from omnibias.torch.architectures.frame_unet import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_worked_and_forward() -> None:
    torch.set_default_dtype(torch.float64)
    t = torch_ex()
    j = jax_ex()
    assert t["band"] == j["band"]
    assert t["collapse"] == j["collapse"]
    ty, ts = torch_fwd(torch.tensor(0.0))
    jy, js = jax_fwd(jnp.asarray(0.0))
    assert float(ty) == float(jy)
    assert ts["band"] == js["band"]
    assert ts["collapse"] == js["collapse"]
