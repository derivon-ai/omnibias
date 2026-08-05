# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Torch soft partition-of-unity weights (bit-identical twin of the numpy reference).

:func:`partition_weights_arrays` is the autograd-friendly kernel (raw tensors in, weights
out) that a trainable bridge module differentiates through; :func:`partition_weights`
evaluates a :class:`~omnibias.partition._core.params.PartitionParams`. Both reproduce
:func:`omnibias.partition._core.weights.partition_weights` bit-for-bit (float64, parity
``~1e-9``).

Terminology: the split gate ``sigmoid(beta (W.x - t))`` hardens as ``beta -> inf`` -- the
feasibility / temperature sense of "collapse", distinct from the **founding bias collapse**
(the multi-bias ``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see
``docs/theory.md``).
"""

from __future__ import annotations

import numpy as np
import torch
from omnibias.partition._core.params import PartitionParams, region_code_matrix
from torch import Tensor


def partition_weights_arrays(
    W: Tensor, t: Tensor, X: Tensor, beta: float, depth: int
) -> Tensor:
    r"""Region weights ``(n, 2**depth)`` from raw tensors -- the differentiable kernel."""
    codes = torch.as_tensor(region_code_matrix(depth), dtype=X.dtype, device=X.device)  # (L, D)
    z = X @ W.transpose(-1, -2) - t[None, :]  # (n, D)
    g = torch.sigmoid(beta * z)  # (n, D)
    gexp = g[:, None, :]  # (n, 1, D)
    bexp = codes[None, :, :]  # (1, L, D)
    factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)  # (n, L, D)
    return torch.prod(factors, dim=-1)  # (n, L)


def partition_weights(params: PartitionParams, X: object, beta: float) -> Tensor:
    r"""Soft partition weights ``(n, 2**depth)`` for a :class:`PartitionParams` (float64)."""
    Xv = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=torch.float64)
    W = torch.as_tensor(params.W, dtype=torch.float64)
    t = torch.as_tensor(params.t, dtype=torch.float64)
    return partition_weights_arrays(W, t, Xv, float(beta), params.depth)


def combine(weights: Tensor, region_outputs: Tensor) -> Tensor:
    r"""Blend per-region outputs by partition weights: ``F = sum_l w_l out_l`` (torch twin)."""
    if region_outputs.ndim == 2:  # (n, L) scalar-per-region
        return torch.einsum("nl,nl->n", weights, region_outputs)
    return torch.einsum("nl,nlk->nk", weights, region_outputs)


__all__ = ["combine", "partition_weights", "partition_weights_arrays"]
