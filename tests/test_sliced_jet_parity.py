# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G5: torch / jax sliced-jet G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.core.sliced_jet import worked_example as core_ex
from omnibias.jax.architectures.sliced_jet import SlicedJetEncoder as JaxEnc
from omnibias.jax.architectures.sliced_jet import worked_example as jax_ex
from omnibias.torch.architectures.sliced_jet import SlicedJetEncoder as TorchEnc
from omnibias.torch.architectures.sliced_jet import worked_example as torch_ex

jax.config.update("jax_enable_x64", True)


def test_g5_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert core_ex() == torch_ex() == jax_ex()
    image_t = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    image_j = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    t_rec = TorchEnc().reconstruct(image_t).detach().cpu().numpy()
    j_rec = JaxEnc().reconstruct(image_j)
    assert t_rec.tolist() == j_rec.tolist()
