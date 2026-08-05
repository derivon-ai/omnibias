# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""NTK-rebalance helpers (jax twin)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def ntk_balanced_loss(
    losses: dict[str, Array],
    *,
    ntk_traces: dict[str, Array] | None = None,
    epsilon: float = 1e-12,
) -> tuple[Array, dict[str, float]]:
    """Combine multiple loss terms with NTK-balanced weights.

    Mirror of the torch implementation. Returns the combined scalar
    loss and a dict of plain-Python weights (for logging).

    Notes
    -----
    The weight computation host-converts ``ntk_traces`` to Python floats
    (the geometric-mean balancing has to materialise the values). This
    function is therefore **not** ``jax.jit``-traceable through the
    ``ntk_traces`` argument; call it outside ``jit``, on host, with
    pre-evaluated traces. The returned scalar ``total`` is still a
    ``jax.Array`` and can be used inside an outer ``jit``.
    """
    if not losses:
        raise ValueError("ntk_balanced_loss: empty losses dict")
    if ntk_traces is None:
        weights = {k: 1.0 for k in losses}
    else:
        if set(ntk_traces) != set(losses):
            raise ValueError(
                f"ntk_traces keys {sorted(ntk_traces)!r} do not match "
                f"losses keys {sorted(losses)!r}"
            )
        log_t: dict[str, float] = {}
        for k, t in ntk_traces.items():
            t_val = float(jnp.maximum(t, epsilon))
            log_t[k] = math.log(t_val)
        mean_log = sum(log_t.values()) / len(log_t)
        weights = {k: math.exp(mean_log - log_t[k]) for k in losses}

    total: Array | None = None
    for k, L in losses.items():
        term = float(weights[k]) * L
        total = term if total is None else total + term
    assert total is not None
    return total, {k: float(weights[k]) for k in losses}


def estimate_ntk_trace(
    loss_fn: Callable[[Any], Array],
    params: Any,
) -> Array:
    """Cheap NTK trace estimator: ``sum_p (d loss / d p)^2``.

    Unlike torch -- where parameters are explicit ``nn.Parameter`` objects --
    in JAX we take a callable ``loss_fn(params) -> scalar`` and a pytree
    of ``params``. The trace is computed as the squared 2-norm of the
    flattened gradient.
    """
    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    out = jnp.zeros((), dtype=jnp.result_type(*[jnp.asarray(g) for g in leaves]) if leaves else jnp.float32)
    for g in leaves:
        out = out + jnp.sum(g * g)
    return out


__all__ = [
    "estimate_ntk_trace",
    "ntk_balanced_loss",
]
