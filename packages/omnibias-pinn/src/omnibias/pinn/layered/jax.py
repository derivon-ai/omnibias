# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Transfer stack (JAX twin; theory 02-11)."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from omnibias.core.transfer import (
    Layer,
    bloch_dispersion,
    reflection_transmission,
    stack_matrix,
)


def transfer_apply(layers: Sequence[Layer], omega: Array) -> tuple[Array, Array]:
    ws = jnp.asarray(omega).reshape(-1).tolist()
    rs = []
    ts = []
    for w in ws:
        r, t = reflection_transmission(stack_matrix(layers, float(w)))
        rs.append(abs(r))
        ts.append(abs(t))
    dtype = jnp.asarray(omega).dtype
    return (
        jnp.asarray(rs, dtype=dtype).reshape(jnp.asarray(omega).shape),
        jnp.asarray(ts, dtype=dtype).reshape(jnp.asarray(omega).shape),
    )


def band_structure(layers: Sequence[Layer], omega: Array) -> Array:
    vals = [bloch_dispersion(layers, float(w)) for w in jnp.asarray(omega).reshape(-1).tolist()]
    return jnp.asarray(vals, dtype=jnp.asarray(omega).dtype).reshape(jnp.asarray(omega).shape)


__all__ = ["band_structure", "transfer_apply"]
