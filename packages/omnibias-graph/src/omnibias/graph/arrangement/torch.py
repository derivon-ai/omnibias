# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Face-Net (torch; theory 02-02). Temperature collapse; subgraph sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
from omnibias.graph.arrangement._core import ArrangementGraph, _message_topology, node_features
from omnibias.partition.arrangement import Arrangement
from torch import Tensor


class FaceNet(nn.Module):
    """Message passing on a discovered arrangement graph. Not a complete lattice."""

    def __init__(
        self,
        arrangement_dim: int,
        hidden: int,
        rounds: int,
        *,
        beta: float = 5.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        self.rounds = int(rounds)
        self.beta = float(beta)
        dt = torch.get_default_dtype() if dtype is None else dtype
        in_dim = 2 + int(arrangement_dim)
        self.lift = nn.Linear(in_dim, hidden, dtype=dt)
        self.msg = nn.Linear(hidden + 1, hidden, dtype=dt)
        self.readout = nn.Linear(hidden, 1, dtype=dt)

    def forward(self, graph: ArrangementGraph, arr: Arrangement) -> Tensor:
        feats = torch.tensor(node_features(arr, graph, beta=self.beta),
                             dtype=self.lift.weight.dtype, device=self.lift.weight.device)
        h = torch.tanh(self.lift(feats))
        n = h.shape[0]
        receivers, senders, labels, degrees = _message_topology(graph, n)
        if receivers:
            dst = torch.tensor(receivers, dtype=torch.long, device=h.device)
            src = torch.tensor(senders, dtype=torch.long, device=h.device)
            crossed = torch.tensor(labels, dtype=h.dtype, device=h.device).reshape(-1, 1)
            degree = torch.tensor(degrees, dtype=h.dtype, device=h.device).reshape(-1, 1)
            for _ in range(self.rounds):
                cat = torch.cat([h[src], crossed], dim=-1)
                messages = torch.tanh(self.msg(cat))
                aggregated = torch.zeros_like(h).index_add(0, dst, messages)
                h = h + aggregated / degree
        return self.readout(h).squeeze(-1)


__all__ = ["FaceNet"]
