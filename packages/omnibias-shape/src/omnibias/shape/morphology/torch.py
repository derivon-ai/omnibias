# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""PyTorch twin of 1-D soft dilation (theory 03-05).

``beta -> inf`` is temperature collapse (feasibility), not the
founding bias collapse (``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

import torch
from omnibias.shape.morphology._core import MorphResult, StructuringElement, named_worked_signal
from torch import Tensor


def logsumexp_beta(a: Tensor, beta: float) -> Tensor:
    """Shifted ``lse_beta``. Reuses the struct formula when installed."""
    try:
        from omnibias.struct.torch import logsumexp_beta as _lse

        return _lse(a, float(beta), axis=-1)
    except ImportError:
        scaled = float(beta) * a
        peak = scaled.amax(dim=-1, keepdim=True)
        return (peak.squeeze(-1) + torch.log(torch.exp(scaled - peak).sum(dim=-1))) / float(beta)


def dilate(f: Tensor, se: StructuringElement, *, beta: float) -> Tensor:
    x = torch.as_tensor(f, dtype=torch.float64).reshape(-1)
    n = int(x.numel())
    cols = []
    for off, val in zip(se.offsets, se.values, strict=True):
        src = torch.arange(n, dtype=torch.int64) - int(off)
        ok = (src >= 0) & (src < n)
        gathered = torch.where(ok, x[src.clamp(0, n - 1)], torch.as_tensor(float("-inf"), dtype=torch.float64))
        cols.append(gathered + float(val))
    win = torch.stack(cols, dim=-1)
    return logsumexp_beta(win, beta)


def worked_center(beta: float = 2.0) -> Tensor:
    se = StructuringElement.flat(1)
    return dilate(torch.as_tensor(named_worked_signal()), se, beta=beta)[2]


__all__ = ["MorphResult", "StructuringElement", "dilate", "logsumexp_beta", "worked_center"]
