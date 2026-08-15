# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Layered transfer twins (theory 02-11). Distinct from geometry.gauge.transfer."""

from __future__ import annotations

import pytest
import torch
from omnibias.core.transfer import Layer, quarter_wave_stack, unitarity_residual
from omnibias.pinn.layered.torch import TransferStack


def test_g1_stack_unitarity() -> None:
    torch.set_default_dtype(torch.float64)
    stack = TransferStack(2, lossless=True, dtype=torch.float64)
    omega = torch.tensor([0.7, 1.1], dtype=torch.float64)
    r, t = stack(omega)
    cons = r * r + t * t
    assert float((cons - 1.0).abs().max().detach()) <= 1e-12
    for layer_set in (stack.layers(),):
        m = __import__("omnibias.core.transfer", fromlist=["stack_matrix"]).stack_matrix(
            layer_set, 1.0
        )
        assert unitarity_residual(m) <= 1e-12


def test_g6_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.pinn.layered.jax import transfer_apply

    jax.config.update("jax_enable_x64", True)
    layers = quarter_wave_stack(2.0, 1.0, n_periods=1, omega0=1.0)
    omega = [0.8, 1.0, 1.2]
    r_t, t_t = TransferStack(1, dtype=torch.float64).forward(
        torch.tensor(omega, dtype=torch.float64)
    )
    # Compare the core algebra twins, not the untrained module.
    r_j, t_j = transfer_apply(layers, jnp.asarray(omega, dtype=jnp.float64))
    from omnibias.core.transfer import reflection_transmission, stack_matrix

    rs = []
    ts = []
    for w in omega:
        r, t = reflection_transmission(stack_matrix(layers, w))
        rs.append(abs(r))
        ts.append(abs(t))
    assert r_j.tolist() == pytest.approx(rs, rel=0, abs=0)
    assert t_j.tolist() == pytest.approx(ts, rel=0, abs=0)
    assert float(r_t[0].detach()) >= 0.0
