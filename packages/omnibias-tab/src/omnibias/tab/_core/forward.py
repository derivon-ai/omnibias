# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Numpy reference forward for the soft decision-tree ensemble (the parity ground truth).

The ensemble of ``T`` oblivious soft trees of ``depth D`` maps a batch ``X`` of shape
``(n, d)`` to raw scores ``F`` of shape ``(n, k)``:

1. gate pre-activations ``Z[n, m, j] = W[m, j] . X[n] - t[m, j]``;
2. gates ``G = sigmoid(beta * Z) in (0, 1)`` (the ``beta -> inf`` limit is a hard split);
3. leaf memberships ``P[n, m, l] = prod_j (G if bit_j(l) else 1 - G)`` -- a product of
   gates (native multiplicative interactions for ``depth >= 2``; for ``depth == 1`` this
   collapses to the additive sum-of-sigmoids ``F = b0 + sum_m g_m (leaves_1 - leaves_0)``);
4. scores ``F[n, k] = sum_{m, l} P[n, m, l] leaves[m, l, k] + b0[k]``.

The torch / jax twins reproduce this bit-for-bit (float64, parity ``~1e-9``).

Terminology: the gate's ``beta -> inf`` hardening is the feasibility / temperature sense
of "collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import numpy as np
from omnibias.tab._core.params import FloatArray, TabParams, leaf_code_matrix


def sigmoid_np(z: FloatArray) -> FloatArray:
    r"""Numerically stable logistic ``1 / (1 + exp(-z))`` (float64)."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def gate_activations(params: TabParams, X: FloatArray, beta: float) -> FloatArray:
    r"""Soft gate activations ``G`` of shape ``(n, n_trees, depth)`` in ``(0, 1)``."""
    Xv = np.asarray(X, dtype=np.float64)
    z = np.einsum("nd,mjd->nmj", Xv, params.W) - params.t[None, :, :]
    return sigmoid_np(beta * z)


def _memberships(G: FloatArray, depth: int) -> FloatArray:
    r"""Leaf memberships ``(n, n_trees, 2**depth)`` from gates ``G`` ``(n, n_trees, depth)``.

    Each leaf's membership is the **soft AND** of its root-to-leaf path conditions -- the
    product ``prod_j (g_j if the path turns right at level j else 1 - g_j)`` -- so for
    ``depth >= 2`` the leaves carry genuine multiplicative feature interactions. Per tree
    the memberships are non-negative and sum to ``1`` (a soft routing distribution).
    """
    codes = leaf_code_matrix(depth)  # (L, D)
    Gexp = G[:, :, None, :]  # (n, T, 1, D)
    B = codes[None, None, :, :]  # (1, 1, L, D)
    factors = B * Gexp + (1.0 - B) * (1.0 - Gexp)  # (n, T, L, D)
    return np.prod(factors, axis=-1)  # (n, T, L)


def leaf_memberships(params: TabParams, X: FloatArray, beta: float) -> FloatArray:
    r"""Per-tree soft-routing distribution ``P`` of shape ``(n, n_trees, 2**depth)``.

    ``P[i, m]`` is a probability vector over tree ``m``'s leaves (non-negative, sums to
    ``1``); ``P[i, m, l]`` is the soft-AND membership of sample ``i`` in leaf ``l``.
    """
    G = gate_activations(params, X, beta)
    return _memberships(G, params.depth)


def forward_np(params: TabParams, X: FloatArray, beta: float) -> FloatArray:
    r"""Raw ensemble scores ``F`` of shape ``(n, n_outputs)`` at gate sharpness ``beta``."""
    G = gate_activations(params, X, beta)
    P = _memberships(G, params.depth)
    F = np.einsum("nml,mlk->nk", P, params.leaves) + params.b0[None, :]
    return F


def hard_forward_np(params: TabParams, X: FloatArray) -> FloatArray:
    r"""The ``beta -> inf`` **hard-tree** scores: gates are crisp ``0/1`` indicators.

    This is the deploy-time model the certified soft->hard rounding gap bounds against.
    """
    Xv = np.asarray(X, dtype=np.float64)
    z = np.einsum("nd,mjd->nmj", Xv, params.W) - params.t[None, :, :]
    G = (z > 0.0).astype(np.float64)
    P = _memberships(G, params.depth)
    return np.einsum("nml,mlk->nk", P, params.leaves) + params.b0[None, :]


def softmax_np(F: FloatArray) -> FloatArray:
    z = F - np.max(F, axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=-1, keepdims=True)


def scores_to_prob(F: FloatArray, task: str) -> FloatArray:
    r"""Map raw scores to probabilities: sigmoid (binary) / softmax (multiclass)."""
    if task == "binary":
        return sigmoid_np(F)
    if task == "multiclass":
        return softmax_np(F)
    raise ValueError(f"scores_to_prob is only defined for classification, not task={task!r}")


def predict_np(params: TabParams, X: FloatArray, beta: float, *, hard: bool = False) -> FloatArray:
    r"""Task predictions: class labels (classification) or real scores (regression)."""
    F = hard_forward_np(params, X) if hard else forward_np(params, X, beta)
    task = params.config.task
    if task == "binary":
        return (F[:, 0] > 0.0).astype(np.float64)
    if task == "multiclass":
        return np.argmax(F, axis=-1).astype(np.float64)
    return F if F.shape[1] > 1 else F[:, 0]


__all__ = [
    "forward_np",
    "gate_activations",
    "hard_forward_np",
    "leaf_memberships",
    "predict_np",
    "scores_to_prob",
    "sigmoid_np",
    "softmax_np",
]
