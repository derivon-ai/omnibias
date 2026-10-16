# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Face-Net (JAX twin; theory 02-02)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.ops import segment_sum
from omnibias.graph.arrangement._core import ArrangementGraph, _message_topology, node_features
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
    """Apply trainable weights to a static arrangement and graph.

    Capture ``graph`` and ``arr`` when using :func:`jax.jit`; feature discovery
    remains host preprocessing. Message arithmetic is batched on the device.
    """
    feats = jnp.asarray(node_features(arr, graph, beta=beta), dtype=lift_w.dtype)
    h = jnp.tanh(feats @ lift_w.T + lift_b)
    n = h.shape[0]
    receivers, senders, labels, degrees = _message_topology(graph, n)
    if receivers:
        dst = jnp.asarray(receivers, dtype=jnp.int32)
        src = jnp.asarray(senders, dtype=jnp.int32)
        crossed = jnp.asarray(labels, dtype=h.dtype).reshape(-1, 1)
        degree = jnp.asarray(degrees, dtype=h.dtype).reshape(-1, 1)
        for _ in range(int(rounds)):
            cat = jnp.concatenate([h[src], crossed], axis=-1)
            messages = jnp.tanh(cat @ msg_w.T + msg_b)
            aggregated = segment_sum(messages, dst, num_segments=n, indices_are_sorted=True)
            h = h + aggregated / degree
    return (h @ out_w.T + out_b).reshape(-1)


__all__ = ["facenet_apply"]
