# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""PyTorch twin of arrangement-LP soft membership (theory 03-02).

``soft_membership`` matches the numpy / jax twins at float64. KKT mode
reuses ``lp_layer``. Soft mode is temperature collapse (``beta -> inf``,
feasibility), not the founding bias collapse (``delta -> 0``). Do not conflate
the two.
"""

from __future__ import annotations

import torch
from omnibias.convex.arrangement._core import (
    DiffMode,
    LearnedPolytope,
    LPOutput,
    named_pentagon,
    soft_cell_gap_bound,
    solve_arrangement_lp,
    sound_lower_bound,
)
from torch import Tensor


def soft_membership(normals: Tensor, offsets: Tensor, x: Tensor, *, beta: float) -> Tensor:
    """``prod_i sigma(beta (b_i - a_i · x))``."""
    a = torch.as_tensor(normals, dtype=torch.float64)
    b = torch.as_tensor(offsets, dtype=torch.float64).reshape(-1)
    xv = torch.as_tensor(x, dtype=torch.float64)
    if xv.ndim == 1:
        slack = b - a @ xv
        return torch.prod(torch.sigmoid(float(beta) * slack))
    slack = b[None, :] - xv @ a.T
    return torch.prod(torch.sigmoid(float(beta) * slack), dim=-1)


class LPLayer:
    """Learned-polytope LP. ``mode`` is returned so the gradient is interpretable."""

    def __init__(self, mode: DiffMode, *, beta: float | None = None) -> None:
        self.mode = DiffMode(mode)
        self.beta = 8.0 if beta is None else float(beta)
        if self.mode is DiffMode.SOFT and self.beta <= 0.0:
            raise ValueError("SOFT mode needs beta > 0")

    def forward(self, polytope: LearnedPolytope, c: Tensor) -> LPOutput:
        cv = torch.as_tensor(c, dtype=torch.float64).detach().cpu().numpy()
        return solve_arrangement_lp(polytope, cv, mode=self.mode, beta=self.beta)


__all__ = [
    "DiffMode",
    "LPLayer",
    "LPOutput",
    "LearnedPolytope",
    "named_pentagon",
    "soft_cell_gap_bound",
    "soft_membership",
    "solve_arrangement_lp",
    "sound_lower_bound",
]
