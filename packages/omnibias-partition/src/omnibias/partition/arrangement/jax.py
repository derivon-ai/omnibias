# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soft arrangement membership (jax; theory 01-03).

``beta -> inf`` is temperature collapse, not founding ``delta -> 0``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid
from omnibias.partition.arrangement._core import Arrangement


def soft_membership(
    arr: Arrangement,
    x: Array,
    signs: Sequence[int],
    *,
    beta: float,
) -> Array:
    if beta <= 0.0:
        raise ValueError("beta must be > 0 (temperature collapse axis)")
    w = jnp.asarray(arr.normals, dtype=x.dtype)
    t = jnp.asarray(arr.offsets, dtype=x.dtype)
    xv = x if x.ndim == 2 else x[None, :]
    z = xv @ w.T - t
    s = jnp.asarray(list(signs), dtype=x.dtype)
    return sigmoid(beta * s * z).prod(axis=-1)


def margin(arr: Arrangement, x: Array) -> Array:
    w = jnp.asarray(arr.normals, dtype=x.dtype)
    t = jnp.asarray(arr.offsets, dtype=x.dtype)
    xv = x if x.ndim == 2 else x[None, :]
    z = xv @ w.T - t
    return jnp.min(jnp.abs(z), axis=-1)


__all__ = ["margin", "soft_membership"]
