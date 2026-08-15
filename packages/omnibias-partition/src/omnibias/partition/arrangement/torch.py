# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soft arrangement membership (torch; theory 01-03).

``beta -> inf`` is temperature collapse, not founding ``delta -> 0``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.partition.arrangement._core import Arrangement
from torch import Tensor


def soft_membership(
    arr: Arrangement,
    x: Tensor,
    signs: Sequence[int],
    *,
    beta: float,
) -> Tensor:
    if beta <= 0.0:
        raise ValueError("beta must be > 0 (temperature collapse axis)")
    w = torch.as_tensor(arr.normals, dtype=x.dtype, device=x.device)
    t = torch.as_tensor(arr.offsets, dtype=x.dtype, device=x.device)
    xv = x if x.ndim == 2 else x.unsqueeze(0)
    z = xv @ w.T - t
    s = torch.as_tensor(list(signs), dtype=x.dtype, device=x.device)
    return torch.sigmoid(beta * s * z).prod(dim=-1)


def margin(arr: Arrangement, x: Tensor) -> Tensor:
    w = torch.as_tensor(arr.normals, dtype=x.dtype, device=x.device)
    t = torch.as_tensor(arr.offsets, dtype=x.dtype, device=x.device)
    xv = x if x.ndim == 2 else x.unsqueeze(0)
    z = xv @ w.T - t
    return z.abs().amin(dim=-1)


__all__ = ["margin", "soft_membership"]
