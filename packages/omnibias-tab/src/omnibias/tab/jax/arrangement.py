# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Functional JAX arrangement / boosted-arrangement forwards.

Bit-identical twins of :class:`~omnibias.tab.torch.arrangement.ArrangementClassifier`
and :class:`~omnibias.tab.torch.arrangement.ArrangementBoosted` (float64, parity
``~1e-9``). Kernels are ``jit`` / ``grad`` / ``vmap``-traceable and compose with
any JAX encoder that emits a feature vector ``(..., d)``. Output rank is
``(..., k)`` with ``k = 1`` for binary cell logits.

Terminology: the split gate ``sigmoid(beta (W.x - t))`` hardens as ``beta -> inf``
(the feasibility / temperature sense of collapse), distinct from the founding
``delta -> 0`` bias collapse.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from omnibias.partition.jax.weights import combine, partition_weights_arrays


def _feature_rows(X: Any) -> tuple[Any, tuple[Any, ...]]:
    if X.ndim < 2:
        raise ValueError("X must have shape (..., n_features)")
    leading = X.shape[:-1]
    return X.reshape((-1, X.shape[-1])), leading


def _cell_2d(cell_logits: Any) -> Any:
    cell = jnp.asarray(cell_logits)
    return cell[:, None] if cell.ndim == 1 else cell


def arrangement_forward(
    W: Any,
    t: Any,
    cell_logits: Any,
    X: Any,
    beta: float,
) -> Any:
    r"""Soft arrangement logits ``(..., k)`` from raw arrays.

    ``W`` is ``(H, d)``, ``t`` is ``(H,)``, ``cell_logits`` is ``(2**H,)`` or
    ``(2**H, k)``, ``X`` is ``(..., d)``.
    """
    rows, leading = _feature_rows(X)
    depth = int(W.shape[0])
    weights = partition_weights_arrays(W, t, rows, beta, depth)
    cell = _cell_2d(cell_logits)
    logits = jnp.broadcast_to(cell, (weights.shape[0], cell.shape[0], cell.shape[1]))
    out = combine(weights, logits)
    return out.reshape(leading + (out.shape[-1],))


def boosted_forward(
    W_stack: Any,
    t_stack: Any,
    logits_stack: Any,
    X: Any,
    beta: float,
    learning_rate: float,
    base: float,
) -> Any:
    r"""Additive ensemble: ``base + lr * sum_m arrangement_m(X)``.

    ``W_stack`` is ``(M, H, d)``, ``t_stack`` is ``(M, H)``, ``logits_stack``
    is ``(M, 2**H)`` or ``(M, 2**H, k)``, ``X`` is ``(..., d)``.
    """
    n_members = int(W_stack.shape[0])
    stacked = jnp.asarray(logits_stack)
    k = int(stacked.shape[-1]) if stacked.ndim == 3 else 1
    if n_members == 0:
        return jnp.full(X.shape[:-1] + (k,), base, dtype=X.dtype)

    def _one(W: Any, t: Any, cell_logits: Any) -> Any:
        return arrangement_forward(W, t, cell_logits, X, beta)

    contrib = jax.vmap(_one)(W_stack, t_stack, logits_stack)
    return base + learning_rate * contrib.sum(axis=0)


__all__ = [
    "arrangement_forward",
    "boosted_forward",
]
