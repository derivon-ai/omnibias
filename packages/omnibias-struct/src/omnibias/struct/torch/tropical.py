# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tropical homotopy twins (torch; theory 01-08).

``beta -> inf`` is temperature collapse, not founding ``delta -> 0``.
Do not conflate the two.
"""

from __future__ import annotations

import torch
from omnibias.struct._core.tropical import TropicalLinear
from omnibias.struct.torch._logsumexp import logsumexp_beta, softmax_beta
from torch import Tensor


def scores(poly: TropicalLinear, x: Tensor) -> Tensor:
    m = torch.as_tensor(poly.exponents, dtype=x.dtype, device=x.device)
    a = torch.as_tensor(poly.coeffs, dtype=x.dtype, device=x.device)
    xv = x if x.ndim == 2 else x.unsqueeze(0)
    return a + xv @ m.T


def relaxed_value(poly: TropicalLinear, x: Tensor, *, beta: float) -> Tensor:
    return logsumexp_beta(scores(poly, x), beta, axis=-1)


def relaxed_weights(poly: TropicalLinear, x: Tensor, *, beta: float) -> Tensor:
    return softmax_beta(scores(poly, x), beta, axis=-1)


__all__ = ["relaxed_value", "relaxed_weights", "scores"]
