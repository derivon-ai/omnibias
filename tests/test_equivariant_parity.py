# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equivariant scan torch/jax parity (theory 02-08 G4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.core.scan import BankSpec
from omnibias.torch.scan_equivariant import EquivariantScan, OrientationBank, steerable_basis


def test_equivariant_scan_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.jax.scan_equivariant import equivariant_scan_apply
    from omnibias.jax.scan_equivariant import steerable_basis as steerable_jax

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    assert steerable_basis(1, 2, base="tanh") is None
    assert steerable_jax(1, 2, base="tanh") is None
    angles = (0.0, math.pi / 2)
    offsets = BankSpec.uniform(-0.4, 0.4, 5)
    net = EquivariantScan(
        2,
        OrientationBank(angles),
        offsets,
        base="gaussian",
        dtype=torch.float64,
    )
    x = torch.tensor([[0.2, -0.15], [0.05, 0.3]], dtype=torch.float64)
    y_t = net(x)
    y_j = equivariant_scan_apply(
        jnp.asarray(x.numpy()), angles, offsets, base="gaussian"
    )
    assert np.asarray(y_t.detach().cpu()) == pytest.approx(
        np.asarray(y_j), rel=1e-12, abs=1e-12
    )
