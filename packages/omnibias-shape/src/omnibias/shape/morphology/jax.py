# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX twin of 1-D soft dilation (theory 03-05).

``beta -> inf`` is temperature collapse (feasibility), not the
founding bias collapse (``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.shape.morphology._core import MorphResult, StructuringElement, named_worked_signal


def logsumexp_beta(a: Array, beta: float) -> Array:
    """Shifted ``lse_beta``. Reuses the struct formula when installed."""
    try:
        from omnibias.struct.jax import logsumexp_beta as _lse

        return _lse(a, float(beta), axis=-1)
    except ImportError:
        scaled = float(beta) * a
        peak = jnp.max(scaled, axis=-1, keepdims=True)
        return (jnp.squeeze(peak, axis=-1) + jnp.log(jnp.sum(jnp.exp(scaled - peak), axis=-1))) / float(beta)


def dilate(f: Array, se: StructuringElement, *, beta: float) -> Array:
    x = jnp.asarray(f, dtype=jnp.float64).reshape(-1)
    n = int(x.size)
    cols = []
    for off, val in zip(se.offsets, se.values, strict=True):
        src = jnp.arange(n) - int(off)
        ok = (src >= 0) & (src < n)
        gathered = jnp.where(ok, x[jnp.clip(src, 0, n - 1)], jnp.asarray(float("-inf"), dtype=jnp.float64))
        cols.append(gathered + float(val))
    win = jnp.stack(cols, axis=-1)
    return logsumexp_beta(win, beta)


def worked_center(beta: float = 2.0) -> Array:
    se = StructuringElement.flat(1)
    return dilate(jnp.asarray(named_worked_signal()), se, beta=beta)[2]


__all__ = ["MorphResult", "StructuringElement", "dilate", "logsumexp_beta", "worked_center"]
