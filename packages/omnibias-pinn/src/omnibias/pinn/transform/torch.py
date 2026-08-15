# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named linearizing transforms (torch; theory 02-13).

Exactness to jet truncation order N. Named transforms only; 03-11 search
is not claimed. A multi-kink sum is not the n-soliton formula unless
built by Bäcklund permutability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omnibias.core.transforms_pde import cole_hopf_u, darboux_dress, miura_v
from torch import Tensor


class ColeHopfField(nn.Module):
    """Parameterizes ``phi`` of the heat equation; ``u = -2 nu phi_x / phi``."""

    def __init__(self, *, nu: float = 1.0, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.nu = float(nu)
        dt = torch.get_default_dtype() if dtype is None else dtype
        self.log_k = nn.Parameter(torch.zeros((), dtype=dt))

    def phi(self, x: Tensor, t: Tensor) -> Tensor:
        k = float(self.log_k.detach().exp())
        return torch.exp(k * (x + t))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        k = float(self.log_k.detach().exp())
        return torch.full_like(x, cole_hopf_u(1.0, k, nu=self.nu))

    def linear_residual(self, x: Tensor, t: Tensor) -> Tensor:
        """Heat residual on ``phi = exp(k(x+t))``: ``phi_t - phi_xx``."""
        k = float(self.log_k.detach().exp())
        phi = self.phi(x, t)
        return k * phi - (k * k) * phi


class MiuraLift(nn.Module):
    """``v -> u = v^2 + v_x``. Exact via the jet product, truncation order N."""

    def __init__(self, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        dt = torch.get_default_dtype() if dtype is None else dtype
        self.amp = nn.Parameter(torch.ones((), dtype=dt))

    def forward(self, u: Tensor, u_x: Tensor) -> Tensor:
        a = float(self.amp.detach())
        xs = u.detach().reshape(-1).tolist()
        dxs = u_x.detach().reshape(-1).tolist()
        vals = [miura_v(a * float(v), a * float(dv)) for v, dv in zip(xs, dxs, strict=True)]
        return torch.tensor(vals, dtype=u.dtype, device=u.device).reshape(u.shape)


def darboux_step(psi: Tensor, psi_x: Tensor, u: Tensor) -> Tensor:
    xs = psi.detach().reshape(-1).tolist()
    dxs = psi_x.detach().reshape(-1).tolist()
    us = u.detach().reshape(-1).tolist()
    vals = [
        darboux_dress(float(p), float(px), float(uu))
        for p, px, uu in zip(xs, dxs, us, strict=True)
    ]
    return torch.tensor(vals, dtype=u.dtype, device=u.device).reshape(u.shape)


__all__ = ["ColeHopfField", "MiuraLift", "darboux_step"]
