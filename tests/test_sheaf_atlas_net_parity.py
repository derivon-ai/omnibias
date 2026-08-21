# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax sheaf-atlas G1 is bit-identical."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
pytest.importorskip("omnibias.partition")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.geometry.atlas.cocycle import AffineChart  # noqa: E402
from omnibias.geometry.atlas.jax import cocycle_residual as jax_res  # noqa: E402
from omnibias.geometry.atlas.jax import worked_example as jax_ex  # noqa: E402
from omnibias.geometry.atlas.torch import cocycle_residual as torch_res  # noqa: E402
from omnibias.geometry.atlas.torch import worked_example as torch_ex  # noqa: E402


def test_g4_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    assert torch_ex()["residual"] == jax_ex()["residual"]
    transitions = {
        (2, 1): AffineChart(2.0),
        (3, 2): AffineChart(0.5),
        (3, 1): AffineChart(1.0),
    }
    t = torch_res(transitions, torch.tensor(0.5), ((1, 2, 3),))
    j = jax_res(transitions, jnp.asarray(0.5), ((1, 2, 3),))
    assert t["residual"] == j["residual"]
