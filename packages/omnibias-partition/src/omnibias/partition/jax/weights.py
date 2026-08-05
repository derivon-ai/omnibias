# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX soft partition-of-unity weights (bit-identical twin of the numpy reference).

:func:`partition_weights_arrays` is the ``jit`` / ``grad`` / ``vmap``-traceable kernel (raw
arrays in, weights out); :func:`partition_weights` evaluates a
:class:`~omnibias.partition._core.params.PartitionParams`. Both reproduce
:func:`omnibias.partition._core.weights.partition_weights` bit-for-bit (float64, parity
``~1e-9``).

Terminology: the split gate ``sigmoid(beta (W.x - t))`` hardens as ``beta -> inf`` -- the
feasibility / temperature sense of "collapse", distinct from the **founding bias collapse**
(the multi-bias ``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see
``docs/theory.md``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from omnibias.partition._core.params import PartitionParams, region_code_matrix


def partition_weights_arrays(W: Any, t: Any, X: Any, beta: float, depth: int) -> Any:
    r"""Region weights ``(n, 2**depth)`` from raw arrays -- the ``jax``-traceable kernel."""
    codes = jnp.asarray(region_code_matrix(depth))  # (L, D)
    z = X @ W.T - t[None, :]  # (n, D)
    g = jax.nn.sigmoid(beta * z)  # (n, D)
    gexp = g[:, None, :]  # (n, 1, D)
    bexp = codes[None, :, :]  # (1, L, D)
    factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)  # (n, L, D)
    return jnp.prod(factors, axis=-1)  # (n, L)


def partition_weights(params: PartitionParams, X: Any, beta: float) -> Any:
    r"""Soft partition weights ``(n, 2**depth)`` for a :class:`PartitionParams` (float64)."""
    Xv = jnp.asarray(X, dtype=jnp.float64)
    return partition_weights_arrays(
        jnp.asarray(params.W), jnp.asarray(params.t), Xv, float(beta), params.depth
    )


def combine(weights: Any, region_outputs: Any) -> Any:
    r"""Blend per-region outputs by partition weights: ``F = sum_l w_l out_l`` (jax twin)."""
    if jnp.ndim(region_outputs) == 2:  # (n, L) scalar-per-region
        return jnp.einsum("nl,nl->n", weights, region_outputs)
    return jnp.einsum("nl,nlk->nk", weights, region_outputs)


__all__ = ["combine", "partition_weights", "partition_weights_arrays"]
