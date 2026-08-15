# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""BEM-Net (torch; theory 02-06). PDE exact off-surface; BC approximated."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from omnibias.core.conjugate import HardyDictionary
from omnibias.pinn.bem._core import (
    KernelSpec,
    Surface,
    pde_residual_off_surface,
    single_layer,
)
from omnibias.pinn.bem._core import (
    half_plane_dtn as half_plane_dtn_core,
)
from torch import Tensor


class BEMNet(nn.Module):
    """Single-layer density on a circle. Combined-layer is the default name only."""

    def __init__(
        self,
        surface: Surface,
        kernel: KernelSpec,
        *,
        layers: str = "combined",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _ = layers
        self.surface = surface
        self.kernel = kernel
        dt = torch.get_default_dtype() if dtype is None else dtype
        self.log_density = nn.Parameter(torch.zeros(surface.n_quad, dtype=dt))

    def density(self) -> Tensor:
        return cast(Tensor, self.log_density)

    def evaluate(self, x: Tensor) -> Tensor:
        dens = [float(v) for v in self.density().detach().reshape(-1).tolist()]
        rows = x.detach().reshape(-1, x.shape[-1])
        vals = [
            single_layer((float(pt[0]), float(pt[1])), self.surface, dens, self.kernel)
            for pt in rows.tolist()
        ]
        return torch.tensor(vals, dtype=x.dtype, device=x.device).reshape(x.shape[:-1])

    def pde_residual(self, x: Tensor) -> Tensor:
        dens = [float(v) for v in self.density().detach().reshape(-1).tolist()]
        rows = x.detach().reshape(-1, x.shape[-1])
        vals = [
            pde_residual_off_surface(
                (float(pt[0]), float(pt[1])), self.surface, dens, self.kernel
            )
            for pt in rows.tolist()
        ]
        return torch.tensor(vals, dtype=x.dtype, device=x.device).reshape(x.shape[:-1])


def half_plane_dtn(
    dictionary: HardyDictionary, coeffs: Tensor, y: Tensor
) -> Tensor:
    c = [float(v) for v in coeffs.detach().reshape(-1).tolist()]
    vals = [
        half_plane_dtn_core(dictionary, c, float(yi))
        for yi in y.detach().reshape(-1).tolist()
    ]
    return torch.tensor(vals, dtype=y.dtype, device=y.device).reshape(y.shape)


__all__ = ["BEMNet", "half_plane_dtn"]
