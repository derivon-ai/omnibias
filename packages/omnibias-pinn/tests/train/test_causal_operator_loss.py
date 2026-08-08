# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for causal_operator_loss on a space-time product grid."""

from __future__ import annotations

import torch
from omnibias.pinn.operator.torch.losses import causal_operator_loss


def test_causal_operator_loss_finite_on_product_grid():
    n_x, n_t, F = 8, 5, 2
    x = torch.linspace(0.0, 1.0, n_x, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64)
    xx = x.repeat(n_t)
    tt = t.repeat_interleave(n_x)
    coords = torch.stack([xx, tt], dim=-1)  # (Q, 2), Q = n_x * n_t
    # Residual grows in time -- causal weighting should down-weight late bins.
    resid = (tt * 0.5).repeat(F)
    loss = causal_operator_loss(resid, coords, epsilon=1.0)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
    # Plain MSE of the same residual.
    plain = torch.mean(resid**2)
    # Causal weights < 1 for later bins, so causal loss <= plain MSE.
    assert float(loss) <= float(plain) + 1e-12
