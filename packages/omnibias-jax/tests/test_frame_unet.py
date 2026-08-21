# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX Frame-UNet twin (09-04)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from omnibias.jax.architectures.frame_unet import frame_unet_forward, worked_example

jax.config.update("jax_enable_x64", True)


def test_g1() -> None:
    ex = worked_example()
    assert ex["band_err"] < 1e-12
    assert ex["skip_gap"] > 1e-3
    y, skips = frame_unet_forward(jnp.asarray(0.0))
    assert y.shape == ()
    assert skips["kinds"] == ("band", "collapse")
