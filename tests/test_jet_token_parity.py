# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax jet-token worked mix is bit-identical."""

from __future__ import annotations

import jax
import torch
from omnibias.jax.architectures.jet_token import worked_compose_jet as jax_worked
from omnibias.torch.architectures.jet_token import worked_compose_jet as torch_worked

jax.config.update("jax_enable_x64", True)


def test_g4_worked_example_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    t = torch_worked()
    j = jax_worked()
    assert float(t[0]) == float(j[0])
    assert float(t[1]) == float(j[1])
