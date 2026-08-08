# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke tests for the torch causal marching driver."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import TimeWindowSchedule
from omnibias.pinn.train.torch import march_solve


class _TinyField(nn.Module):
    """Linear field u(x,t) = a*x + b*t + c -- enough to exercise the loop."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.c = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.a * coords[:, 0] + self.b * coords[:, 1] + self.c


def test_march_solve_runs_all_windows():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=2, n_time_bins=4, epsilon=1.0, tolerance=1e-6
    )
    field = _TinyField()

    def residual_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        # Soft residual: push u toward sin(pi x) * exp(-t).
        u = fld(coords)
        target = torch.sin(np.pi * coords[:, 0]) * torch.exp(-coords[:, 1])
        return u - target

    ic = np.sin(np.pi * np.linspace(0.0, 1.0, 16, endpoint=False))
    # n_slice must match; march_solve uses n_slice=16 below.
    result = march_solve(
        field,
        residual_fn,
        cs,
        schedule,
        steps_per_window=5,
        lr=1e-2,
        per_bin=4,
        n_slice=16,
        ic_values=ic,
        seed=0,
        check_trivial=True,
    )
    assert len(result.windows) == 2
    assert result.windows[0].window_index == 0
    assert result.windows[1].bounds[1] == 1.0
    assert result.trivial is not None
    # Healthy target -- should not be flagged trivial after a few steps.
    assert not result.trivial.is_trivial


def test_march_solve_hard_ic_skips_penalty():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 0.5)), time_axis="t"
    )
    schedule = TimeWindowSchedule(0.0, 0.5, n_windows=1, n_time_bins=2)
    field = _TinyField()

    def residual_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        return fld(coords)

    result = march_solve(
        field,
        residual_fn,
        cs,
        schedule,
        steps_per_window=2,
        per_bin=2,
        n_slice=4,
        ic_mode="hard",
        check_trivial=False,
    )
    assert result.diagnostics["ic_mode"] == "hard"
    assert len(result.windows) == 1
