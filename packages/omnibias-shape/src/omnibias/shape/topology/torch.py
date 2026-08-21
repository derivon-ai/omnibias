# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""PyTorch twin of soft component counts (theory 03-09).

``beta -> inf`` is temperature collapse (feasibility), not the
founding bias collapse (``delta -> 0``). Do not conflate the two.
No differentiable function equals a Betti number.
"""

from __future__ import annotations

import torch
from torch import Tensor


def soft_component_count(eigenvalues: Tensor, *, epsilon: float, beta: float) -> Tensor:
    ev = torch.as_tensor(eigenvalues, dtype=torch.float64).reshape(-1)
    z = float(beta) * (float(epsilon) - ev)
    return (1.0 / (1.0 + torch.exp(-z))).sum()


def persistence_loss(births: Tensor, deaths: Tensor, *, threshold: float, mode: str = "suppress") -> Tensor:
    pers = torch.as_tensor(deaths, dtype=torch.float64) - torch.as_tensor(births, dtype=torch.float64)
    if mode == "suppress":
        return torch.relu(float(threshold) - pers).sum()
    if mode == "encourage":
        return torch.relu(pers - float(threshold)).sum()
    raise ValueError("mode must be 'suppress' or 'encourage'")
