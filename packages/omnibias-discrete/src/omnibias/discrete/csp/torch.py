# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""PyTorch twin of CSP softmax / energy (theory 03-03).

``softmax_rows`` matches numpy / jax at float64. Simplex and clause
``beta -> inf`` are temperature collapse (feasibility), not the
founding bias collapse (``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

import torch
from omnibias.discrete.csp._core import (
    CSP,
    CSPResult,
    csp_solve,
    triangle_colouring,
)
from torch import Tensor


def softmax_rows(logits: Tensor, *, beta: float) -> Tensor:
    z = float(beta) * torch.as_tensor(logits, dtype=torch.float64)
    z = z - torch.amax(z, dim=-1, keepdim=True)
    w = torch.exp(z)
    return w / torch.sum(w, dim=-1, keepdim=True)


def energy(csp: CSP, x: Tensor) -> Tensor:
    xv = torch.as_tensor(x, dtype=torch.float64).detach().cpu().numpy()
    return torch.as_tensor(csp.energy(xv), dtype=torch.float64)


__all__ = ["CSP", "CSPResult", "csp_solve", "energy", "softmax_rows", "triangle_colouring"]
