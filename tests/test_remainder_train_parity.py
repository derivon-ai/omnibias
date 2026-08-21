# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax remainder of exp is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_remainder import remainder_loss as jax_loss
from omnibias.torch.optim_remainder import remainder_loss as torch_loss

jax.config.update("jax_enable_x64", True)


def test_g4_exp_r2_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    xs = [0.2]
    jet = [1.0, 1.0, 1.0]
    import math

    vals = [math.exp(0.2)]
    t = torch_loss(torch.tensor(vals), torch.tensor(jet), torch.tensor(xs))
    j = jax_loss(jnp.asarray(vals), jnp.asarray(jet), jnp.asarray(xs))
    assert float(t["max_abs"]) == float(j["max_abs"])
    assert float(t["loss"]) == float(j["loss"])
