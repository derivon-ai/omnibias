# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX parity for a tiny 1-D Poisson depth-causal sweep."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import torch
from omnibias.pinn.train import DepthResidualConfig
from omnibias.pinn.train.jax.depth_residual import (
    depth_residual_sweep as jax_sweep,
)
from omnibias.pinn.train.jax.depth_residual import (
    poisson_1d_residual as jax_poisson,
)
from omnibias.pinn.train.torch.depth_residual import (
    depth_residual_sweep as torch_sweep,
)
from omnibias.pinn.train.torch.depth_residual import (
    poisson_1d_residual as torch_poisson,
)


def test_tiny_poisson_parity() -> None:
    torch.set_default_dtype(torch.float64)
    xs_np = np.linspace(0.15, 0.85, 5)
    w = np.array([[0.4], [-0.25]])
    b = np.array([0.1, -0.05])
    v = np.array([[0.3, -0.2]])
    c = np.array([0.0])
    cfg = DepthResidualConfig(n_directions=1, jet_order=2, damping=1e-4)
    force_np = (np.pi**2) * np.sin(np.pi * xs_np)

    t_layers = [(torch.tensor(w), torch.tensor(b), "tanh")]
    t_decode = (torch.tensor(v), torch.tensor(c))
    t_xs = torch.tensor(xs_np)
    t_new, t_dec, t_rep = torch_sweep(
        t_layers, t_decode, torch_poisson(torch.tensor(force_np)), t_xs, config=cfg
    )

    j_layers = [(jnp.asarray(w), jnp.asarray(b), "tanh")]
    j_decode = (jnp.asarray(v), jnp.asarray(c))
    j_xs = jnp.asarray(xs_np)
    j_new, j_dec, j_rep = jax_sweep(
        j_layers, j_decode, jax_poisson(jnp.asarray(force_np)), j_xs, config=cfg
    )

    gap_w = abs(float(t_new[0][0][0, 0]) - float(j_new[0][0][0, 0].reshape(-1)[0]))
    gap_v = abs(float(t_dec[0][0][0, 0]) - float(j_dec[0][0][0, 0].reshape(-1)[0]))
    gap_r = abs(t_rep.residual_norms[-1] - j_rep.residual_norms[-1])
    assert gap_w <= 1e-10
    assert gap_v <= 1e-10
    assert gap_r <= 1e-10
    assert t_rep.greedy_only_claimed_optimal is False
    assert j_rep.hilbert_not_in_scope is True
