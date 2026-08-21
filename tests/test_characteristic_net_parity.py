# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax Characteristic-Net G1 is bit-identical."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.characteristic import constant_v, gaussian_u0  # noqa: E402
from omnibias.pinn.jax.characteristic import characteristic_eval as jax_eval  # noqa: E402
from omnibias.pinn.jax.characteristic import worked_example as jax_ex  # noqa: E402
from omnibias.pinn.torch.characteristic import characteristic_eval as torch_eval  # noqa: E402
from omnibias.pinn.torch.characteristic import worked_example as torch_ex  # noqa: E402


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["u"] == jax_ex()["u"]
    tu, tc = torch_eval(torch.tensor(0.0), 0.2, constant_v, gaussian_u0)
    ju, jc = jax_eval(jnp.asarray(0.0), 0.2, constant_v, gaussian_u0)
    assert float(tu) == float(ju)
    assert float(tc) == float(jc)
