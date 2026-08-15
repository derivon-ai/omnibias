# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soliton field (torch; theory 02-09)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn
from omnibias.core.tanh_method import (
    PDESpec,
    TravellingWaveAnsatz,
    evaluate_ansatz,
    substitute,
)
from torch import Tensor


class SolitonField(nn.Module):
    """Sum of tanh-polynomial travelling waves. Not the n-soliton formula."""

    def __init__(
        self,
        ansatz: Sequence[TravellingWaveAnsatz],
        *,
        learn_interaction: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not ansatz:
            raise ValueError("at least one travelling-wave ansatz is required")
        self.ansatz = tuple(ansatz)
        self.learn_interaction = bool(learn_interaction)
        dt = torch.get_default_dtype() if dtype is None else dtype
        amps = torch.ones(len(self.ansatz), dtype=dt)
        if learn_interaction:
            self.amps = nn.Parameter(amps)
        else:
            self.register_buffer("amps", amps)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        amp = cast(Tensor, self.amps)
        xs = x.detach().reshape(-1)
        ts = t.detach().reshape(-1)
        out = torch.zeros(xs.shape[0], dtype=x.dtype, device=x.device)
        for a, wave in zip(amp.detach().tolist(), self.ansatz, strict=True):
            vals = [
                float(a) * evaluate_ansatz(wave, float(xi), float(ti))
                for xi, ti in zip(xs.tolist(), ts.tolist(), strict=True)
            ]
            out = out + torch.tensor(vals, dtype=x.dtype, device=x.device)
        return out.reshape(x.shape)

    def exact_residual(self, x: Tensor, t: Tensor, pde: PDESpec) -> Tensor:
        """Polynomial residual in ``T``; zero iff the ansatz is exact."""
        xs = x.detach().reshape(-1)
        ts = t.detach().reshape(-1)
        acc = torch.zeros(xs.shape[0], dtype=x.dtype, device=x.device)
        for wave in self.ansatz:
            coeffs = substitute(pde, wave)
            rows = []
            for xi, ti in zip(xs.tolist(), ts.tolist(), strict=True):
                z = float(wave.wavenumber) * float(xi) - float(wave.frequency) * float(ti)
                import math

                tnh = math.tanh(z)
                res = 0.0
                p = 1.0
                for c in coeffs:
                    res += float(c) * p
                    p *= tnh
                rows.append(res)
            acc = acc + torch.tensor(rows, dtype=x.dtype, device=x.device)
        return acc.reshape(x.shape)


__all__ = ["SolitonField"]
