# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Depth-causal local jet, JAX (theory 08-03)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from omnibias.core.local_jet import LocalJetConfig, LocalJetForbidden
from omnibias.jax.train_local import local_jet_step

jax.config.update("jax_enable_x64", True)


def _section5_layers() -> list[tuple[Array, None, str | None]]:
    w = jnp.asarray([[0.5]])
    v = jnp.asarray([[0.2]])
    return [(w, None, "tanh"), (v, None, None)]


def test_section5_readout_then_hidden() -> None:
    cfg = LocalJetConfig(n_directions=1, damping=0.0, variant="readout")
    layers, report = local_jet_step(
        _section5_layers(),
        jnp.asarray([1.0]),
        config=cfg,
        target=jnp.asarray([1.0]),
    )
    v = jnp.reshape(layers[1][0], (-1,))
    w = jnp.reshape(layers[0][0], (-1,))
    h = jnp.tanh(w * 1.0)
    yhat = jnp.reshape(v * h, (-1,))
    assert abs(float(yhat[0]) - 1.0) <= 1e-12
    assert report.greedy_only_claimed_optimal is False
    assert report.n_params == 2


def test_g1_forbid_flood() -> None:
    with pytest.raises(LocalJetForbidden, match="allow_full"):
        local_jet_step(
            _section5_layers(),
            jnp.asarray([1.0]),
            config=LocalJetConfig(n_directions=2, damping=0.0),
            target=jnp.asarray([1.0]),
        )


def test_invert_and_match_tanh() -> None:
    z_star = 0.3
    target = jnp.asarray([math.tanh(z_star)])
    w0 = jnp.asarray([[0.5]])
    layers, _report = local_jet_step(
        [(w0, None, "tanh")],
        jnp.asarray([1.0]),
        config=LocalJetConfig(
            n_directions=1, damping=0.0, variant="invert", allow_full=True
        ),
        target=target,
    )
    u = float(jnp.reshape(layers[0][0], (-1,))[0])
    assert abs(u - z_star) <= 1e-12


def test_predcode_linear_to_eps() -> None:
    w0 = jnp.asarray([[0.0]])
    layers, report = local_jet_step(
        [(w0, None, None)],
        jnp.asarray([1.0]),
        config=LocalJetConfig(
            n_directions=1, damping=0.0, variant="predcode", allow_full=True
        ),
        target=jnp.asarray([2.0]),
    )
    yhat = float(jnp.reshape(layers[0][0], (-1,))[0])
    assert abs(yhat - 2.0) <= 1e-12
    assert report.residual_norms[0] <= 1e-12
