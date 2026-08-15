# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Chart-coordinate bias scan (torch; theory 02-08). Discrete ``C_L``, not SO(2)."""

from __future__ import annotations

import torch
from omnibias.core.scan import BankSpec
from omnibias.geometry._core.charts import ChartSpec
from omnibias.geometry.torch.ops.pullback import pullback_metric
from torch import Tensor


def chart_scan(
    chart: ChartSpec,
    x: Tensor,
    direction: Tensor,
    offsets: BankSpec,
    *,
    metric_correction: bool = True,
) -> Tensor:
    """Scan in chart coordinates with optional ``sqrt(g_vv)`` spacing."""
    d = direction / direction.norm().clamp_min(1e-12)
    z = (x * d).sum(dim=-1)
    if metric_correction:
        g = pullback_metric(x, chart)
        gd = torch.matmul(g, d)
        gvv = (gd * d).sum(dim=-1).clamp_min(1e-18)
        z = z * gvv.sqrt()
    off = torch.tensor(list(offsets.offsets), dtype=x.dtype, device=x.device)
    return z.unsqueeze(-1) + off


__all__ = ["chart_scan"]
