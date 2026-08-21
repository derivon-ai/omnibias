# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax q-OMBU G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.qcalculus.jax.hybrid import q_ombu_forward as jax_fwd
from omnibias.qcalculus.jax.hybrid import worked_example as jax_ex
from omnibias.qcalculus.torch.hybrid import q_ombu_forward as torch_fwd
from omnibias.qcalculus.torch.hybrid import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["y"] == jax_ex()["y"]
    t_y, t_r = torch_fwd(torch.tensor(2.0))
    j_y, j_r = jax_fwd(jnp.asarray(2.0))
    assert float(t_y) == float(j_y)
    assert float(t_r) == float(j_r)
