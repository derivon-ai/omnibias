# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hierarchical scan (JAX twin; theory 02-07)."""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.hierarchy import Cluster, hierarchical_value

import jax.numpy as jnp
from jax import Array


def hierarchical_scan(
    z: Array,
    tree: Cluster,
    offsets: Sequence[float],
    weights: Sequence[float],
    orders: Sequence[int],
    *,
    p: int = 6,
    eta: float = 0.5,
    base: str = "tanh",
) -> Array:
    zs = jnp.asarray(z).reshape(-1).tolist()
    vals = [
        hierarchical_value(
            float(zi), tree, offsets, weights, orders, p=p, eta=eta, base=base
        )
        for zi in zs
    ]
    zz = jnp.asarray(z)
    return jnp.asarray(vals, dtype=zz.dtype).reshape(zz.shape)


__all__ = ["hierarchical_scan"]
