# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX twin of soft component counts (theory 03-09).

``beta -> inf`` is temperature collapse (feasibility), not the
founding bias collapse (``delta -> 0``). Do not conflate the two.
No differentiable function equals a Betti number.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def soft_component_count(eigenvalues: Array, *, epsilon: float, beta: float) -> Array:
    ev = jnp.asarray(eigenvalues, dtype=jnp.float64).reshape(-1)
    return jnp.sum(1.0 / (1.0 + jnp.exp(-float(beta) * (float(epsilon) - ev))))


def persistence_loss(births: Array, deaths: Array, *, threshold: float, mode: str = "suppress") -> Array:
    pers = jnp.asarray(deaths, dtype=jnp.float64) - jnp.asarray(births, dtype=jnp.float64)
    if mode == "suppress":
        return jnp.sum(jnp.maximum(0.0, float(threshold) - pers))
    if mode == "encourage":
        return jnp.sum(jnp.maximum(0.0, pers - float(threshold)))
    raise ValueError("mode must be 'suppress' or 'encourage'")
