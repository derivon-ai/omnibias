# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Optional Equinox wrappers: filter_grad reaches encoder and W; filter_jit matches eager."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

_ON_CI = os.environ.get("CI", "").lower() in ("1", "true", "yes")
if _ON_CI:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
else:
    eqx = pytest.importorskip("equinox")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

from omnibias.tab import SoftTreeConfig, init_params  # noqa: E402
from omnibias.tab.jax.equinox_head import ArrangementHead, BoostedHead, SoftTreeHead  # noqa: E402


class _Encoder(eqx.Module):
    W: jax.Array
    b: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(x @ self.W + self.b)


class _Composed(eqx.Module):
    encoder: _Encoder
    head: eqx.Module

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(self.encoder(x))


def _loss(model: _Composed, x: jax.Array) -> jax.Array:
    return jnp.mean(model(x))


def test_equinox_arrangement_filter_grad_and_jit() -> None:
    rng = np.random.default_rng(0)
    encoder = _Encoder(
        W=jnp.asarray(rng.normal(size=(8, 16)) * 0.3),
        b=jnp.asarray(rng.normal(size=(16,)) * 0.1),
    )
    head = ArrangementHead(
        W=jnp.asarray(rng.normal(size=(2, 16)) * 0.3),
        t=jnp.asarray(rng.normal(size=(2,)) * 0.1),
        cell_logits=jnp.asarray(rng.normal(size=(4,))),
        beta=jnp.asarray(4.0),
    )
    model = _Composed(encoder=encoder, head=head)
    x = jnp.asarray(rng.normal(size=(32, 8)))
    grads = eqx.filter_grad(_loss)(model, x)
    assert float(jnp.max(jnp.abs(grads.encoder.W))) > 0.0
    assert float(jnp.max(jnp.abs(grads.head.W))) > 0.0
    eager = model(x)
    jitted = eqx.filter_jit(model)(x)
    assert float(jnp.max(jnp.abs(eager - jitted))) < 1e-9


def test_equinox_softtree_filter_grad_and_jit() -> None:
    rng = np.random.default_rng(1)
    cfg = SoftTreeConfig(
        n_features=16, n_trees=3, depth=2, task="binary", beta_final=4.0, seed=1
    )
    p = init_params(cfg, 1)
    encoder = _Encoder(
        W=jnp.asarray(rng.normal(size=(8, 16)) * 0.3),
        b=jnp.asarray(rng.normal(size=(16,)) * 0.1),
    )
    head = SoftTreeHead(
        W=jnp.asarray(p.W),
        t=jnp.asarray(p.t),
        leaves=jnp.asarray(p.leaves),
        b0=jnp.asarray(p.b0),
        beta=jnp.asarray(float(cfg.beta_final)),
        depth=int(cfg.depth),
    )
    model = _Composed(encoder=encoder, head=head)
    x = jnp.asarray(rng.normal(size=(24, 8)))
    grads = eqx.filter_grad(_loss)(model, x)
    assert float(jnp.max(jnp.abs(grads.encoder.W))) > 0.0
    assert float(jnp.max(jnp.abs(grads.head.W))) > 0.0
    eager = model(x)
    jitted = eqx.filter_jit(model)(x)
    assert float(jnp.max(jnp.abs(eager - jitted))) < 1e-9


def test_equinox_boosted_filter_grad_and_jit() -> None:
    rng = np.random.default_rng(2)
    encoder = _Encoder(
        W=jnp.asarray(rng.normal(size=(8, 16)) * 0.3),
        b=jnp.asarray(rng.normal(size=(16,)) * 0.1),
    )
    head = BoostedHead(
        W_stack=jnp.asarray(rng.normal(size=(3, 2, 16)) * 0.3),
        t_stack=jnp.asarray(rng.normal(size=(3, 2)) * 0.1),
        logits_stack=jnp.asarray(rng.normal(size=(3, 4))),
        beta=jnp.asarray(4.0),
        learning_rate=jnp.asarray(0.3),
        base=jnp.asarray(0.0),
    )
    model = _Composed(encoder=encoder, head=head)
    x = jnp.asarray(rng.normal(size=(24, 8)))
    grads = eqx.filter_grad(_loss)(model, x)
    assert float(jnp.max(jnp.abs(grads.encoder.W))) > 0.0
    assert float(jnp.max(jnp.abs(grads.head.W_stack))) > 0.0
    eager = model(x)
    jitted = eqx.filter_jit(model)(x)
    assert float(jnp.max(jnp.abs(eager - jitted))) < 1e-9
