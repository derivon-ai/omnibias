# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hermite ladder network (torch; theory 02-10).

Raw tower is not the QHO eigenbasis. ``normalization`` is required.
"""

from __future__ import annotations

from typing import Literal, cast

from omnibias.core.ladder import Normalization, hermite_function

import torch
import torch.nn as nn
from torch import Tensor


class HermiteBasis(nn.Module):
    def __init__(
        self,
        n_levels: int,
        *,
        normalization: Normalization,
        centres: int = 1,
        learnable_scale: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        self.n_levels = int(n_levels)
        self.normalization = Normalization(normalization)
        dt = torch.get_default_dtype() if dtype is None else dtype
        c0 = torch.zeros(int(centres), dtype=dt)
        if learnable_scale:
            self.log_scale = nn.Parameter(torch.zeros(int(centres), dtype=dt))
            self.centres = nn.Parameter(c0)
        else:
            self.register_buffer("log_scale", torch.zeros(int(centres), dtype=dt))
            self.register_buffer("centres", c0)

    def forward(self, x: Tensor) -> Tensor:
        scale = float(cast(Tensor, self.log_scale)[0].detach().exp())
        centre = float(cast(Tensor, self.centres)[0].detach())
        xs = x.detach().reshape(-1).tolist()
        rows = [
            [
                hermite_function(
                    n, float(xi), normalization=self.normalization, scale=scale, centre=centre
                )
                for n in range(self.n_levels)
            ]
            for xi in xs
        ]
        return torch.tensor(rows, dtype=x.dtype, device=x.device).reshape(
            *x.shape, self.n_levels
        )

    def apply_operator(self, coeffs: Tensor, which: Literal["N", "H"]) -> Tensor:
        n = coeffs.shape[-1]
        idx = torch.arange(n, dtype=coeffs.dtype, device=coeffs.device)
        if which == "N":
            return coeffs * idx
        return coeffs * (idx + 0.5)

    def raise_(self, coeffs: Tensor) -> Tensor:
        out = torch.zeros_like(coeffs)
        out[..., 1:] = coeffs[..., :-1]
        return out

    def lower(self, coeffs: Tensor) -> Tensor:
        out = torch.zeros_like(coeffs)
        n = coeffs.shape[-1]
        idx = torch.arange(1, n, dtype=coeffs.dtype, device=coeffs.device)
        out[..., :-1] = coeffs[..., 1:] * idx
        return out


class LadderNet(nn.Module):
    """Depth indexed by excitation number rather than arbitrary width."""

    def __init__(
        self,
        n_levels: int,
        *,
        normalization: Normalization,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.basis = HermiteBasis(n_levels, normalization=normalization, dtype=dtype)
        dt = torch.get_default_dtype() if dtype is None else dtype
        self.readout = nn.Linear(n_levels, 1, dtype=dt)

    def forward(self, x: Tensor) -> Tensor:
        feats = self.basis(x)
        out: Tensor = self.readout(feats).squeeze(-1)
        return out


__all__ = ["HermiteBasis", "LadderNet"]
