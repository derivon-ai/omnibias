# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Face-Net (JAX twin; theory 02-02)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.graph.arrangement._core import ArrangementGraph, node_features
from omnibias.partition.arrangement import Arrangement


def facenet_apply(
    graph: ArrangementGraph,
    arr: Arrangement,
    *,
    lift_w: Array,
    lift_b: Array,
    msg_w: Array,
    msg_b: Array,
    out_w: Array,
    out_b: Array,
    rounds: int,
    beta: float,
) -> Array:
    feats = jnp.asarray(node_features(arr, graph, beta=beta), dtype=lift_w.dtype)
    h = jnp.tanh(feats @ lift_w.T + lift_b)
    n = h.shape[0]
    adj = jnp.zeros((n, n), dtype=h.dtype)
    crossed = jnp.zeros((n, n), dtype=h.dtype)
    for u, v, k in graph.edges:
        adj = adj.at[u, v].set(1.0).at[v, u].set(1.0)
        crossed = crossed.at[u, v].set(float(k)).at[v, u].set(float(k))
    for _ in range(int(rounds)):
        msgs = []
        for i in range(n):
            acc = jnp.zeros_like(h[i])
            deg = 0.0
            for j in range(n):
                if float(adj[i, j]) <= 0.0:
                    continue
                cat = jnp.concatenate([h[j], crossed[i, j].reshape(1)])
                acc = acc + jnp.tanh(cat @ msg_w.T + msg_b)
                deg += 1.0
            msgs.append(acc / max(deg, 1.0))
        h = h + jnp.stack(msgs)
    return (h @ out_w.T + out_b).reshape(-1)


__all__ = ["facenet_apply"]
