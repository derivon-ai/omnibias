# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch/JAX parity for depth-causal local jet (theory 08-03 G4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.core.local_jet import LocalJetConfig
from omnibias.jax.train_local import local_jet_step as jax_step
from omnibias.torch.train_local import local_jet_step as torch_step

jax.config.update("jax_enable_x64", True)


def test_g4_section5_parity() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = LocalJetConfig(n_directions=1, damping=0.0, variant="readout")
    t_layers, t_report = torch_step(
        [(torch.tensor([[0.5]]), None, "tanh"), (torch.tensor([[0.2]]), None, None)],
        torch.tensor([1.0]),
        config=cfg,
        target=torch.tensor([1.0]),
    )
    j_layers, j_report = jax_step(
        [(jnp.asarray([[0.5]]), None, "tanh"), (jnp.asarray([[0.2]]), None, None)],
        jnp.asarray([1.0]),
        config=cfg,
        target=jnp.asarray([1.0]),
    )
    assert abs(float(t_layers[0][0].reshape(-1)[0]) - float(j_layers[0][0].reshape(-1)[0])) <= 1e-12
    assert abs(float(t_layers[1][0].reshape(-1)[0]) - float(j_layers[1][0].reshape(-1)[0])) <= 1e-12
    assert t_report.n_params == j_report.n_params
    assert t_report.greedy_only_claimed_optimal is False
    assert j_report.greedy_only_claimed_optimal is False
