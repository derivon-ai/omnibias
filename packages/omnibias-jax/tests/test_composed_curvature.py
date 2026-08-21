# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX composed curvature (theory 08-02): G1 FD and G4."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.core.composed_curvature import ComposedCurvatureConfig
from omnibias.jax.optim_composed import composed_block_hessian, scalar_nest_hessian


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def test_g1_compose_jet_and_hvp_match_fd() -> None:
    w = jnp.asarray(0.2)
    v = jnp.asarray(0.1)
    closed = scalar_nest_hessian(w, v, 1.0)

    def residual(prev: jax.Array, curr: jax.Array) -> jax.Array:
        h = jnp.tanh(prev * 1.0)
        return jnp.reshape(curr * h - 1.0, (1,))

    cfg = ComposedCurvatureConfig(n_directions=2, allow_full=True)
    h_slice, h_joint, cross = composed_block_hessian(
        residual,
        w,
        v,
        directions=(
            (jnp.asarray(1.0), jnp.asarray(0.0)),
            (jnp.asarray(0.0), jnp.asarray(1.0)),
        ),
        config=cfg,
    )
    assert _rel(float(h_joint[0, 0]), closed[0]) <= 1e-10
    assert _rel(float(h_joint[0, 1]), closed[1]) <= 1e-10
    assert _rel(float(h_joint[1, 1]), closed[2]) <= 1e-10
    assert _rel(float(h_slice[0, 0]), closed[2]) <= 1e-10
    assert _rel(float(cross[0, 0]), closed[1]) <= 1e-10


def test_g4_raises_without_allow_full() -> None:
    def residual(prev: jax.Array, curr: jax.Array) -> jax.Array:
        return jnp.reshape(curr * jnp.tanh(prev) - 1.0, (1,))

    with pytest.raises(ValueError, match="allow_full"):
        composed_block_hessian(
            residual,
            jnp.asarray(0.2),
            jnp.asarray(0.1),
            config=ComposedCurvatureConfig(n_directions=2, allow_full=False),
        )
