# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax exact-MAML quadratic step is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.jax.optim_maml import inner_newton_quadratic as jax_step
from omnibias.torch.optim_maml import inner_newton_quadratic as torch_step

jax.config.update("jax_enable_x64", True)


def test_g4_quadratic_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    t = torch_step(torch.tensor(2.5), torch.tensor(-0.3))
    j = jax_step(jnp.asarray(2.5), jnp.asarray(-0.3))
    assert float(t) == float(j)
