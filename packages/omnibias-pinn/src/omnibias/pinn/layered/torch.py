# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Transfer stack (torch; theory 02-11). 1-D layered only."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from omnibias.core.transfer import (
    Layer,
    bloch_dispersion,
    reflection_transmission,
    stack_matrix,
)
from torch import Tensor


class TransferStack(nn.Module):
    """Learnable thicknesses; structural identities stay in the algebra."""

    def __init__(self, n_layers: int, *, lossless: bool = True, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.lossless = bool(lossless)
        dt = torch.get_default_dtype() if dtype is None else dtype
        self.log_n = nn.Parameter(torch.zeros(n_layers, dtype=dt))
        self.log_d = nn.Parameter(torch.zeros(n_layers, dtype=dt))

    def layers(self) -> tuple[Layer, ...]:
        ns = cast(Tensor, self.log_n).detach().exp()
        ds = cast(Tensor, self.log_d).detach().exp()
        out: list[Layer] = []
        for n, d in zip(ns.tolist(), ds.tolist(), strict=True):
            out.append(Layer(complex(float(n), 0.0), float(d)))
        return tuple(out)

    def forward(self, omega: Tensor) -> tuple[Tensor, Tensor]:
        rs = []
        ts = []
        for w in omega.detach().reshape(-1).tolist():
            m = stack_matrix(self.layers(), float(w))
            r, t = reflection_transmission(m)
            rs.append(abs(r))
            ts.append(abs(t))
        return (
            torch.tensor(rs, dtype=omega.dtype, device=omega.device).reshape(omega.shape),
            torch.tensor(ts, dtype=omega.dtype, device=omega.device).reshape(omega.shape),
        )

    def band_structure(self, omega: Tensor) -> Tensor:
        vals = [
            bloch_dispersion(self.layers(), float(w))
            for w in omega.detach().reshape(-1).tolist()
        ]
        return torch.tensor(vals, dtype=omega.dtype, device=omega.device).reshape(omega.shape)


__all__ = ["TransferStack"]
