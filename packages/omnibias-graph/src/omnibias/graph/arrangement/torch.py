# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Face-Net (torch; theory 02-02). Temperature collapse; subgraph sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
from omnibias.graph.arrangement._core import ArrangementGraph, node_features
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
        feats = torch.tensor(node_features(arr, graph, beta=self.beta), dtype=self.lift.weight.dtype)
        h = torch.tanh(self.lift(feats))
        n = h.shape[0]
        adj = torch.zeros(n, n, dtype=h.dtype)
        crossed = torch.zeros(n, n, dtype=h.dtype)
        for u, v, k in graph.edges:
            adj[u, v] = adj[v, u] = 1.0
            crossed[u, v] = crossed[v, u] = float(k)
        for _ in range(self.rounds):
            msgs = []
            for i in range(n):
                acc = torch.zeros_like(h[i])
                deg = 0.0
                for j in range(n):
                    if float(adj[i, j].detach()) <= 0.0:
                        continue
                    cat = torch.cat([h[j], crossed[i, j].reshape(1)])
                    acc = acc + torch.tanh(self.msg(cat))
                    deg += 1.0
                msgs.append(acc / max(deg, 1.0))
            h = h + torch.stack(msgs)
        return self.readout(h).squeeze(-1)


__all__ = ["FaceNet"]
