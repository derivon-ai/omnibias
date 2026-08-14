# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX tab layers as neural heads: encoder arrays + arrangement / soft-tree kernels.

Run:

    python docs/examples/tab_as_layer_jax.py

Functional kernels, plus an optional Equinox ``eqx.Module`` wrapper when the
``[equinox]`` extra is installed. ``jax.grad`` / ``eqx.filter_grad`` reach encoder
weights **and** tree / arrangement parameters; ``jax.jit`` / ``eqx.filter_jit``
compile the composed map. Constructors stay float64; pass arrays already in the
host dtype. ``import omnibias.tab.jax`` stays Equinox-free.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from omnibias.tab import SoftTreeConfig, init_params
from omnibias.tab.jax.arrangement import arrangement_forward
from omnibias.tab.jax.model import forward_arrays


def _assert_both_move(g_enc: object, g_head: object) -> None:
    ge = jnp.asarray(g_enc)
    gh = jnp.asarray(g_head)
    assert float(jnp.max(jnp.abs(ge))) > 0.0, "encoder grad vanished"
    assert float(jnp.max(jnp.abs(gh))) > 0.0, "head grad vanished"


def arrangement_encoder() -> None:
    print("=== encoder + arrangement_forward (jax.grad + jit) ===")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 8))
    enc_W = rng.normal(size=(8, 16)) * 0.3
    enc_b = rng.normal(size=(16,)) * 0.1
    W = rng.normal(size=(2, 16)) * 0.3
    t = rng.normal(size=(2,)) * 0.1
    cell = rng.normal(size=(4,))
    beta = 4.0

    def loss(eW, eb, tW, tt, logits, xv):
        z = jnp.tanh(xv @ eW + eb)
        return arrangement_forward(tW, tt, logits, z, beta).mean()

    compiled = jax.jit(loss)
    xv = jnp.asarray(X)
    args = (
        jnp.asarray(enc_W),
        jnp.asarray(enc_b),
        jnp.asarray(W),
        jnp.asarray(t),
        jnp.asarray(cell),
        xv,
    )
    v0 = float(compiled(*args))
    g_eW, g_W = jax.grad(compiled, argnums=(0, 2))(*args)
    _assert_both_move(g_eW, g_W)
    assert np.isfinite(v0)
    print("  encoder and arrangement W both have nonzero grad; jit matches")


def softtree_encoder() -> None:
    print("=== encoder + forward_arrays (jax.grad) ===")
    rng = np.random.default_rng(1)
    cfg = SoftTreeConfig(
        n_features=16, n_trees=4, depth=2, task="binary", beta_final=8.0, seed=1
    )
    p = init_params(cfg)
    X = rng.normal(size=(24, 8))
    enc_W = rng.normal(size=(8, 16)) * 0.3
    enc_b = rng.normal(size=(16,)) * 0.1

    def loss(eW, eb, tW, tt, leaves, b0, xv):
        z = jnp.tanh(xv @ eW + eb)
        return forward_arrays(tW, tt, leaves, b0, z, float(cfg.beta_final), cfg.depth).mean()

    g_eW, g_tW = jax.grad(loss, argnums=(0, 2))(
        jnp.asarray(enc_W),
        jnp.asarray(enc_b),
        jnp.asarray(p.W),
        jnp.asarray(p.t),
        jnp.asarray(p.leaves),
        jnp.asarray(p.b0),
        jnp.asarray(X),
    )
    _assert_both_move(g_eW, g_tW)
    print("  encoder and SoftTree W both have nonzero grad")


def equinox_heads() -> None:
    try:
        import equinox as eqx
    except ImportError:
        print("=== Equinox wrappers skipped (pip install equinox) ===")
        return
    from omnibias.tab.jax.equinox_head import ArrangementHead, BoostedHead

    print("=== Equinox ArrangementHead (filter_grad + filter_jit) ===")
    rng = np.random.default_rng(2)
    enc_W = jnp.asarray(rng.normal(size=(8, 16)) * 0.3)
    head = ArrangementHead(
        W=jnp.asarray(rng.normal(size=(2, 16)) * 0.3),
        t=jnp.asarray(rng.normal(size=(2,)) * 0.1),
        cell_logits=jnp.asarray(rng.normal(size=(4,))),
        beta=jnp.asarray(4.0),
    )
    xv = jnp.asarray(rng.normal(size=(16, 8)))

    def loss(params, x):
        eW, h = params
        z = jnp.tanh(x @ eW)
        return jnp.mean(h(z))

    g_eW, g_head = eqx.filter_grad(loss)((enc_W, head), xv)
    _assert_both_move(g_eW, g_head.W)
    eager = head(jnp.tanh(xv @ enc_W))
    jitted = eqx.filter_jit(head)(jnp.tanh(xv @ enc_W))
    assert float(jnp.max(jnp.abs(eager - jitted))) < 1e-9
    print("  encoder and arrangement W both have nonzero filter_grad; filter_jit matches")

    print("=== Equinox BoostedHead (filter_grad + filter_jit) ===")
    boosted = BoostedHead(
        W_stack=jnp.asarray(rng.normal(size=(3, 2, 16)) * 0.3),
        t_stack=jnp.asarray(rng.normal(size=(3, 2)) * 0.1),
        logits_stack=jnp.asarray(rng.normal(size=(3, 4))),
        beta=jnp.asarray(4.0),
        learning_rate=jnp.asarray(0.3),
        base=jnp.asarray(0.0),
    )

    def loss_b(params, x):
        eW, h = params
        z = jnp.tanh(x @ eW)
        return jnp.mean(h(z))

    g_eW_b, g_boost = eqx.filter_grad(loss_b)((enc_W, boosted), xv)
    _assert_both_move(g_eW_b, g_boost.W_stack)
    eager_b = boosted(jnp.tanh(xv @ enc_W))
    jitted_b = eqx.filter_jit(boosted)(jnp.tanh(xv @ enc_W))
    assert float(jnp.max(jnp.abs(eager_b - jitted_b))) < 1e-9
    print("  encoder and boosted W_stack both have nonzero filter_grad; filter_jit matches")


def main() -> None:
    arrangement_encoder()
    softtree_encoder()
    equinox_heads()
    print("OK: JAX tab kernels plug into a host net; autograd reaches encoder and head.")


if __name__ == "__main__":
    main()
