# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G5: torch / jax parameter-jet G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import torch
from omnibias.core.parameter_jets import worked_example as core_ex

jax.config.update("jax_enable_x64", True)

pytest.importorskip("omnibias.pinn.operator.torch.parameter_jets")
from omnibias.pinn.operator.jax.parameter_jets import mixed_jet as jax_jet
from omnibias.pinn.operator.jax.parameter_jets import worked_example as jax_ex
from omnibias.pinn.operator.torch.parameter_jets import mixed_jet as torch_jet
from omnibias.pinn.operator.torch.parameter_jets import worked_example as torch_ex


def test_g5_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert core_ex() == torch_ex() == jax_ex()
    ex = core_ex()
    t_du = torch_jet(None, torch.tensor([ex["x"], ex["t"]]), torch.tensor([ex["mu"]]))
    j_du = jax_jet(None, jnp.asarray([ex["x"], ex["t"]]), jnp.asarray([ex["mu"]]))
    assert t_du == j_du
