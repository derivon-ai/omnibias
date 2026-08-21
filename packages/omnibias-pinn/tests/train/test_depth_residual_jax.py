# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Depth-causal residual sweep, JAX (theory 08-05)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
from omnibias.pinn.train import DepthResidualConfig, DepthResidualForbidden
from omnibias.pinn.train.jax.depth_residual import (
    depth_residual_sweep,
    field_tower,
    poisson_1d_residual,
)


def test_manufactured_constant_field_is_exact() -> None:
    xs = jnp.linspace(0.1, 0.9, 9)
    layers = [(jnp.zeros((1, 1)), jnp.ones((1,)), None)]
    decode = (jnp.ones((1, 1)), None)
    cfg = DepthResidualConfig(n_directions=1, jet_order=2, damping=0.0)
    tower = field_tower(layers, decode, xs, config=cfg)
    residual = poisson_1d_residual(jnp.full_like(xs, 2.0))(tower)
    assert float(jnp.max(jnp.abs(residual))) <= 1e-12
    assert jnp.allclose(tower[2], jnp.full_like(xs, -2.0))


def test_g2_honesty_keys_sealed() -> None:
    xs = jnp.linspace(0.1, 0.9, 5)
    layers = [(0.2 * jnp.ones((2, 1)), jnp.zeros((2,)), "tanh")]
    decode = (0.1 * jnp.ones((1, 2)), jnp.zeros((1,)))
    force = (jnp.pi**2) * jnp.sin(jnp.pi * xs)
    cfg = DepthResidualConfig(n_directions=1, jet_order=2, damping=1e-3)
    _layers, _dec, report = depth_residual_sweep(
        layers, decode, poisson_1d_residual(force), xs, config=cfg
    )
    assert report.navier_stokes_proof_claim is False
    assert report.stretch_cleared is False
    assert report.hilbert_not_in_scope is True
    assert report.greedy_only_claimed_optimal is False


def test_flood_forbid() -> None:
    layers = [(jnp.ones((1, 1)), None, None)]
    decode = (jnp.ones((1, 1)), None)
    xs = jnp.asarray([0.3, 0.7])
    cfg = DepthResidualConfig(n_directions=2, jet_order=2, damping=0.0)
    with pytest.raises(DepthResidualForbidden, match="allow_full"):
        depth_residual_sweep(
            layers,
            decode,
            poisson_1d_residual(jnp.ones_like(xs)),
            xs,
            config=cfg,
        )
