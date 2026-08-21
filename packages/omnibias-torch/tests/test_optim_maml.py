# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch exact-MAML twin (09-16)."""

from __future__ import annotations

import torch
from omnibias.torch.optim_maml import (
    ift_dtheta_dalpha,
    inner_newton_quadratic,
    quadratic_worked_example,
)


def test_g1_and_g4_quadratic() -> None:
    torch.set_default_dtype(torch.float64)
    theta = torch.tensor(2.5)
    alpha = torch.tensor(-0.3)
    theta_p = inner_newton_quadratic(theta, alpha)
    assert abs(float(theta_p - alpha)) < 1e-12
    ift = ift_dtheta_dalpha(theta, alpha)
    assert abs(float(ift) - 1.0) < 1e-12
    ex = quadratic_worked_example()
    assert ex["ift_err"] < 1e-10
