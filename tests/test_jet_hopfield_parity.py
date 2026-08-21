# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax Jet-Hopfield G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.core.jet_hopfield import JetHopfieldConfig
from omnibias.jax.architectures.jet_hopfield import jet_hopfield_retrieve as jax_ret
from omnibias.jax.architectures.jet_hopfield import worked_example as jax_ex
from omnibias.torch.architectures.jet_hopfield import jet_hopfield_retrieve as torch_ret
from omnibias.torch.architectures.jet_hopfield import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["retrieved_value"] == jax_ex()["retrieved_value"]
    cfg = JetHopfieldConfig(lam=1.0, beta=10.0)
    t = torch_ret(torch.tensor([1.0, 0.01]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]), config=cfg)
    j = jax_ret(jnp.asarray([1.0, 0.01]), jnp.asarray([[1.0, 0.0], [0.0, 1.0]]), config=cfg)
    assert float(t[0]) == float(j[0])
    assert float(t[1]) == float(j[1])
