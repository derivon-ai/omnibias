# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Keras 3 soft partition-of-unity weights (``keras.ops`` twin of numpy / torch / jax).

Parity with :func:`omnibias.partition._core.weights.partition_weights` is
``~1e-9`` in float64. Lives here (not in ``omnibias-keras``) so the keras core
package never depends on partition / tab.

Terminology: the split gate ``sigmoid(beta (W.x - t))`` hardens as ``beta -> inf``
(the feasibility / temperature sense of collapse), distinct from the founding
``delta -> 0`` bias collapse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from keras import ops
from omnibias.partition._core.params import PartitionParams, region_code_matrix


def prod_last_axis(x: Any, size: int) -> Any:
    """Product over the last axis, preserving operand dtype.

    ``keras.ops.prod`` on the torch backend returns float32 even when ``x`` is
    float64, which breaks float64 parity. Sequential multiply keeps dtype.
    """
    n = int(size)
    if n < 1:
        raise ValueError("size must be >= 1")
    acc = x[..., 0]
    for i in range(1, n):
        acc = acc * x[..., i]
    return acc


def partition_weights_arrays(W: Any, t: Any, X: Any, beta: float, depth: int) -> Any:
    r"""Region weights ``(n, 2**depth)`` from raw tensors -- the ``keras.ops`` kernel."""
    codes = ops.convert_to_tensor(region_code_matrix(int(depth)), dtype=X.dtype)
    z = ops.matmul(X, ops.transpose(W)) - ops.expand_dims(t, 0)
    g = ops.sigmoid(ops.cast(beta, X.dtype) * z)
    gexp = ops.expand_dims(g, 1)
    bexp = ops.expand_dims(codes, 0)
    factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)
    return prod_last_axis(factors, int(depth))


def partition_weights(params: PartitionParams, X: object, beta: float) -> Any:
    r"""Soft partition weights ``(n, 2**depth)`` for a :class:`PartitionParams`."""
    Xv = ops.convert_to_tensor(np.asarray(X, dtype=np.float64), dtype="float64")
    W = ops.convert_to_tensor(params.W, dtype="float64")
    t = ops.convert_to_tensor(params.t, dtype="float64")
    return partition_weights_arrays(W, t, Xv, float(beta), params.depth)


def combine(weights: Any, region_outputs: Any) -> Any:
    r"""Blend per-region outputs: ``F = sum_l w_l out_l``."""
    if ops.ndim(region_outputs) == 2:
        return ops.einsum("nl,nl->n", weights, region_outputs)
    return ops.einsum("nl,nlk->nk", weights, region_outputs)


__all__ = ["combine", "partition_weights", "partition_weights_arrays", "prod_last_axis"]
