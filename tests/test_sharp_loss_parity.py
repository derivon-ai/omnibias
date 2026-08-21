# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax sharpness-loss G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_sharp_loss import sharpness_augmented_loss as jax_aug
from omnibias.jax.optim_sharp_loss import worked_example as jax_ex
from omnibias.torch.optim_sharp_loss import sharpness_augmented_loss as torch_aug
from omnibias.torch.optim_sharp_loss import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex() == jax_ex()
    assert torch_aug(None, torch.tensor(0.0)) == jax_aug(None, jnp.asarray(0.0))
