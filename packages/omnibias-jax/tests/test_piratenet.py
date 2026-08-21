# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PirateNet identity skip and a tiny 1-D residual smoke."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from omnibias.jax.architectures.piratenet import (  # noqa: E402
    PirateNetConfig,
    init_pirate_params,
    pirate_apply,
    pirate_features,
)


def test_alpha_zero_is_embedding() -> None:
    cfg = PirateNetConfig(in_dim=2, hidden=6, n_layers=3, seed=1)
    params = init_pirate_params(cfg)
    assert float(jnp.max(jnp.abs(params["alpha"]))) == 0.0
    x = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
    coords = jnp.stack([x, x * x], axis=-1)
    feat = pirate_features(params, coords)
    emb = jnp.tanh(coords @ params["We"].T + params["be"])
    assert float(jnp.max(jnp.abs(feat - emb))) < 1e-14


def test_one_d_residual_smoke_decreases() -> None:
    cfg = PirateNetConfig(in_dim=1, hidden=8, n_layers=1, seed=2)
    params = init_pirate_params(cfg)
    x = jnp.linspace(-1.0, 1.0, 32, dtype=jnp.float64)[:, None]
    target = jnp.sin(jnp.pi * x[:, 0])

    def loss(th: dict) -> jnp.ndarray:
        pred = pirate_apply(th, x)
        return jnp.mean((pred - target) ** 2)

    l0 = float(loss(params))
    grad = jax.grad(loss)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - 0.05 * g, params, grad)
    l1 = float(loss(params))
    assert l1 < l0
