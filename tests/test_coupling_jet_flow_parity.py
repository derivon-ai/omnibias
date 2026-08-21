# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax coupling jet-flow log-det is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import torch

jax.config.update("jax_enable_x64", True)

pytest.importorskip("omnibias.score.flow.torch.jet_flow")
from omnibias.score.flow.jax.jet_flow import jet_flow_forward as jax_fwd
from omnibias.score.flow.torch.jet_flow import jet_flow_forward as torch_fwd


def test_g4_log_det_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    yt, lt = torch_fwd(torch.tensor(0.3), torch.tensor(2.0))
    yj, lj = jax_fwd(jnp.asarray(0.3), jnp.asarray(2.0))
    assert float(yt) == float(yj)
    assert float(lt) == float(lj)
