# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G4: torch / jax exact-div G1 is bit-identical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import torch
from omnibias.core.score_matching import exact_div_neg_id as core_div
from omnibias.core.score_matching import worked_example as core_ex

jax.config.update("jax_enable_x64", True)

pytest.importorskip("omnibias.score.torch.score_matching")
from omnibias.score.jax.score_matching import exact_div_neg_id as jax_div
from omnibias.score.jax.score_matching import worked_example as jax_ex
from omnibias.score.torch.score_matching import exact_div_neg_id as torch_div
from omnibias.score.torch.score_matching import worked_example as torch_ex


def test_g4_core_is_the_twin() -> None:
    torch.set_default_dtype(torch.float64)

    assert core_ex() == torch_ex() == jax_ex()
    assert core_div(1) == torch_div(1) == jax_div(1)
    assert torch_div(torch.tensor(1.0)) == jax_div(jnp.asarray(1.0))
