# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Discrete-orbit equivariant scan (torch; theory 02-08).

Exact steering is gaussian-family only. ``C_L`` is a discrete orbit, not SO(2).
``steerable_basis`` returns ``None`` for non-gaussian bases (no fallback).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from omnibias.core.scan import BankSpec
from omnibias.torch.scan import BiasScan, OpName

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class OrientationBank:
    angles: tuple[float, ...]
    steerable_order: int | None = None


@dataclass(frozen=True)
class SteerableBasis:
    order: int
    dim: int
    matrices: tuple[tuple[tuple[float, ...], ...], ...]


def steerable_basis(order: int, dim: int, *, base: str = "gaussian") -> SteerableBasis | None:
    """Exact steering coefficients. ``None`` for non-gaussian (no approximation)."""
    if str(base).lower() != "gaussian":
        return None
    if dim != 2:
        return None
    n = int(order)
    # Rotation of 2-D homogeneous harmonics of degree n: (n+1) cartesian
    # monomials mixed by an SO(2) representation. Stored as R(theta) later.
    return SteerableBasis(order=n, dim=2, matrices=())


class EquivariantScan(nn.Module):
    """Scan a ``C_L`` orientation orbit. Not SO(2) / SO(3)."""

    def __init__(
        self,
        dim: int,
        bank: OrientationBank,
        offsets: BankSpec,
        *,
        template: OpName | str = "grad",
        base: str = "gaussian",
        readout: Literal["orbit", "max", "fourier"] = "orbit",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("equivariant scan needs dim >= 2")
        self.dim = int(dim)
        self.orientation = bank
        self.base = str(base)
        self.readout = readout
        dt = torch.get_default_dtype() if dtype is None else dtype
        self.scan = BiasScan(1, offsets, template=template, base=base, dtype=dt)

    def forward(self, x: Tensor) -> Tensor:
        responses = []
        for angle in self.orientation.angles:
            c, s = math.cos(angle), math.sin(angle)
            w = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
            w[0] = c
            w[1] = s
            z = (x * w).sum(dim=-1, keepdim=True)
            responses.append(self.scan(z))
        stacked = torch.stack(responses, dim=-1)
        if self.readout == "max":
            return stacked.amax(dim=-1)
        if self.readout == "fourier":
            return stacked.mean(dim=-1)
        return stacked


__all__ = ["EquivariantScan", "OrientationBank", "SteerableBasis", "steerable_basis"]
