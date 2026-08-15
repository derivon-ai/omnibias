# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hermite ladder torch/jax parity (theory 02-10 G6)."""

from __future__ import annotations

import pytest
import torch
from omnibias.core.ladder import Normalization
from omnibias.torch.architectures.ladder import HermiteBasis


def test_ladder_basis_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.jax.architectures.ladder import hermite_basis

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    x = torch.tensor([-0.4, 0.0, 0.7], dtype=torch.float64)
    basis = HermiteBasis(6, normalization=Normalization.TOWER, learnable_scale=False)
    y_t = basis(x)
    y_j = hermite_basis(
        jnp.asarray(x.numpy()), 6, normalization=Normalization.TOWER
    )
    import numpy as np

    assert np.asarray(y_t.detach().cpu()) == pytest.approx(np.asarray(y_j), rel=0, abs=0)
