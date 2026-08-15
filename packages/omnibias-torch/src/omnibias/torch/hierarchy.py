# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hierarchical scan (torch; theory 02-07). 1-D offset axis; eta=0 is dense."""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.hierarchy import Cluster, hierarchical_value

import torch
from torch import Tensor


def hierarchical_scan(
    z: Tensor,
    tree: Cluster,
    offsets: Sequence[float],
    weights: Sequence[float],
    orders: Sequence[int],
    *,
    p: int = 6,
    eta: float = 0.5,
    base: str = "tanh",
) -> Tensor:
    zs = z.detach().reshape(-1).tolist()
    vals = [
        hierarchical_value(
            float(zi), tree, offsets, weights, orders, p=p, eta=eta, base=base
        )
        for zi in zs
    ]
    return torch.tensor(vals, dtype=z.dtype, device=z.device).reshape(z.shape)


__all__ = ["hierarchical_scan"]
