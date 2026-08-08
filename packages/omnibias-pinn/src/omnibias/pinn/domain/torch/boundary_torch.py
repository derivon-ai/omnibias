# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch boundary-factor jets mirroring :mod:`omnibias.pinn.domain._core.boundary`."""

from __future__ import annotations

import numpy as np
import torch
from omnibias.pinn.domain._core.boundary import BCMode, boundary_factor_jet
from omnibias.pinn.domain._core.sdf import SDF
from torch import Tensor


def boundary_factor_jet_at(
    sdf: SDF,
    x0: Tensor,
    *,
    order: int,
    mode: BCMode = "dirichlet",
    normalize: bool = True,
    robin_alpha: float = 1.0,
    robin_beta: float = 0.0,
    h: float = 1e-6,
) -> Tensor:
    """Exact multivariate jet of the BC distance factor at ``x0``."""
    if x0.ndim != 1:
        raise ValueError(f"x0 must be 1-D, got shape {tuple(x0.shape)}")
    jet = boundary_factor_jet(
        sdf,
        np.asarray(x0.detach().cpu(), dtype=float),
        order=order,
        mode=mode,
        normalize=normalize,
        robin_alpha=robin_alpha,
        robin_beta=robin_beta,
        h=h,
    )
    return torch.tensor(jet, dtype=x0.dtype, device=x0.device)


__all__ = ["boundary_factor_jet_at"]
