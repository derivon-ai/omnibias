# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""PyTorch twin of activation-mixture CDF / PDF (theory 03-04).

Mixtures come from the founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not appear.
Do not conflate the two.
"""

from __future__ import annotations

import torch
from omnibias.measure.transport._core import ActivationMixture
from torch import Tensor


def _sigmoid(z: Tensor) -> Tensor:
    x = torch.as_tensor(z, dtype=torch.float64)
    return torch.where(x >= 0.0, 1.0 / (1.0 + torch.exp(-x)), torch.exp(x) / (1.0 + torch.exp(x)))


def cdf(mix: ActivationMixture, x: Tensor) -> Tensor:
    xv = torch.as_tensor(x, dtype=torch.float64)
    loc = torch.as_tensor(mix.loc(), dtype=torch.float64)
    alpha = torch.as_tensor(mix.alpha, dtype=torch.float64)
    w = torch.as_tensor(mix.weights, dtype=torch.float64)
    z = alpha * (xv[..., None] - loc)
    return torch.sum(w * _sigmoid(z), dim=-1)


def pdf(mix: ActivationMixture, x: Tensor) -> Tensor:
    xv = torch.as_tensor(x, dtype=torch.float64)
    loc = torch.as_tensor(mix.loc(), dtype=torch.float64)
    alpha = torch.as_tensor(mix.alpha, dtype=torch.float64)
    w = torch.as_tensor(mix.weights, dtype=torch.float64)
    z = alpha * (xv[..., None] - loc)
    s = _sigmoid(z)
    return torch.sum(w * alpha * s * (1.0 - s), dim=-1)


def quantile(mix: ActivationMixture, q: Tensor, *, steps: int = 40) -> Tensor:
    qq = torch.as_tensor(q, dtype=torch.float64)
    mean = torch.as_tensor(mix.mean(), dtype=torch.float64)
    spread = 1.0 / float(mix.alpha.mean())
    x = mean + spread * (2.0 * qq - 1.0)
    for _ in range(int(steps)):
        dens = torch.clamp(pdf(mix, x), min=1e-30)
        x = x - (cdf(mix, x) - qq) / dens
    return x


__all__ = ["ActivationMixture", "cdf", "pdf", "quantile"]
