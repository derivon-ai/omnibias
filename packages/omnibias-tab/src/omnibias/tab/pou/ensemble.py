# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Parameter-efficient k-head residual (TabM / BatchEnsemble, theory 05-04).

Shared linear ``W`` with per-member rank-1 scales ``(r_k, s_k)``. Prediction is
the mean of ``k`` heads. Trained on the boost residual ``y - F_boost``.
"""

from __future__ import annotations

import torch
from omnibias.partition.torch.weights import combine as combine_region_outputs
from torch import Tensor, nn

_DTYPE = torch.float64


class LinearBatchEnsemble(nn.Module):
    r"""BatchEnsemble linear map: ``y_k = (s_k * W * r_k) x + b_k``.

    ``forward`` maps ``(n, in)`` or ``(n, k, in)`` to ``(n, k, out)``.
    """

    def __init__(self, in_features: int, out_features: int, ensemble_k: int) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or ensemble_k < 1:
            raise ValueError("in_features, out_features, ensemble_k must be >= 1")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.ensemble_k = int(ensemble_k)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=_DTYPE))
        self.r = nn.Parameter(torch.ones(ensemble_k, in_features, dtype=_DTYPE))
        self.s = nn.Parameter(torch.ones(ensemble_k, out_features, dtype=_DTYPE))
        self.bias = nn.Parameter(torch.zeros(ensemble_k, out_features, dtype=_DTYPE))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            xr = x.unsqueeze(1) * self.r.unsqueeze(0)  # (n, k, in)
        elif x.ndim == 3:
            if x.shape[1] != self.ensemble_k:
                raise ValueError(
                    f"x.shape[1] must equal ensemble_k={self.ensemble_k}, got {x.shape[1]}"
                )
            xr = x * self.r.unsqueeze(0)
        else:
            raise ValueError(f"x must be 2-D or 3-D, got shape {tuple(x.shape)}")
        y = torch.matmul(xr, self.weight.t())  # (n, k, out)
        return y * self.s.unsqueeze(0) + self.bias.unsqueeze(0)


class TabMResidual(nn.Module):
    r"""Two-layer BatchEnsemble MLP; output is the mean over ``k`` members."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        *,
        ensemble_k: int = 8,
        hidden: int = 64,
        n_regions: int | None = None,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.ensemble_k = int(ensemble_k)
        self.n_regions = None if n_regions is None else int(n_regions)
        self.scale = nn.Parameter(torch.ones(n_features, dtype=_DTYPE))
        self.fc1 = LinearBatchEnsemble(n_features, hidden, ensemble_k)
        self.fc2 = LinearBatchEnsemble(hidden, n_outputs, ensemble_k)
        if self.n_regions is not None:
            if self.n_regions < 1:
                raise ValueError("n_regions must be >= 1")
            self.region_head: LinearBatchEnsemble | None = LinearBatchEnsemble(
                n_features, self.n_regions * n_outputs, ensemble_k
            )
        else:
            self.region_head = None

    def forward(self, x: Tensor, pou_weights: Tensor | None = None) -> Tensor:
        h = torch.relu(self.fc1(x * self.scale.unsqueeze(0)))
        o = self.fc2(h).mean(dim=1)  # (n, n_outputs)
        if self.region_head is not None and pou_weights is not None:
            raw = self.region_head(x).mean(dim=1)
            n = int(x.shape[0])
            raw = raw.reshape(n, int(self.n_regions), self.n_outputs)
            o = o + combine_region_outputs(pou_weights, raw)
        return o


__all__ = ["LinearBatchEnsemble", "TabMResidual"]
