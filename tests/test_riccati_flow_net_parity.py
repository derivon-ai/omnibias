# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax Riccati flow G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.core.riccati_flow import RiccatiFlowConfig
from omnibias.jax.architectures.riccati_flow import riccati_flow as jax_flow
from omnibias.jax.architectures.riccati_flow import worked_example as jax_ex
from omnibias.torch.architectures.riccati_flow import riccati_flow as torch_flow
from omnibias.torch.architectures.riccati_flow import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["s"] == jax_ex()["s"]
    t = torch_flow(torch.tensor(0.25), config=RiccatiFlowConfig(t=1.0))
    j = jax_flow(jnp.asarray(0.25), config=RiccatiFlowConfig(t=1.0))
    assert float(t) == float(j)
