# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the torch causal marching driver."""

from __future__ import annotations

import numpy as np
import pytest
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


def _target(coords: torch.Tensor) -> torch.Tensor:
    return torch.sin(np.pi * coords[:, 0]) * torch.exp(-coords[:, 1])


def test_march_solve_runs_all_windows():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=2, n_time_bins=4, epsilon=1.0, tolerance=1e-6
    )
    field = _TinyField()

    def residual_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        return fld(coords) - _target(coords)

    ic = np.sin(np.pi * np.linspace(0.0, 1.0, 16, endpoint=False))
    result = march_solve(
        field,
        residual_fn,
        cs,
        schedule,
        steps_per_window=5,
        max_steps_per_window=20,
        lr=1e-2,
        per_bin=4,
        n_slice=16,
        ic_values=ic,
        seed=0,
        check_trivial=True,
        advance_policy="force",
    )
    assert len(result.windows) == 2
    assert result.windows[0].window_index == 0
    assert result.windows[1].bounds[1] == 1.0
    assert result.windows[0].seam_mse is not None
    assert result.trivial is not None
    assert not result.trivial.is_trivial


def test_march_solve_requires_ic():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 0.5)), time_axis="t"
    )
    schedule = TimeWindowSchedule(0.0, 0.5, n_windows=1, n_time_bins=2)
    with pytest.raises(ValueError, match="ic_values or ic_fn"):
        march_solve(
            _TinyField(),
            lambda fld, coords: fld(coords),
            cs,
            schedule,
            steps_per_window=1,
            per_bin=2,
            n_slice=4,
            check_trivial=False,
        )


def test_march_solve_hard_ic_skips_penalty():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 0.5)), time_axis="t"
    )
    schedule = TimeWindowSchedule(0.0, 0.5, n_windows=1, n_time_bins=2)
    field = _TinyField()
    called: list[bool] = []

    def factory(fld: nn.Module, coords: torch.Tensor, values: torch.Tensor) -> nn.Module:
        called.append(True)
        assert coords.shape[0] == 4
        assert values.shape[0] == 4
        return fld

    result = march_solve(
        field,
        lambda fld, coords: fld(coords),
        cs,
        schedule,
        steps_per_window=2,
        per_bin=2,
        n_slice=4,
        ic_values=np.zeros(4),
        ic_mode="hard",
        hard_ic_factory=factory,
        check_trivial=False,
        advance_policy="force",
    )
    assert result.diagnostics["ic_mode"] == "hard"
    assert called == [True]
    assert len(result.windows) == 1


def test_march_solve_gate_refuses_unconverged_advance():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    # Impossible tolerance with a messy residual -> never unlocks.
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=3, n_time_bins=4, epsilon=100.0, tolerance=0.999
    )
    field = _TinyField()
    ic = np.sin(np.pi * np.linspace(0.0, 1.0, 8, endpoint=False))
    result = march_solve(
        field,
        lambda fld, coords: fld(coords) + 1.0,
        cs,
        schedule,
        steps_per_window=2,
        max_steps_per_window=4,
        per_bin=2,
        n_slice=8,
        ic_values=ic,
        seed=1,
        check_trivial=False,
        advance_policy="gate",
    )
    assert len(result.windows) == 1
    assert result.windows[0].exhausted
    assert not result.windows[0].converged
    assert not result.all_converged


def test_march_solve_vector_residual_no_sign_cancel():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 0.5)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 0.5, n_windows=1, n_time_bins=2, epsilon=1.0, tolerance=1e-12
    )
    field = _TinyField()

    def residual_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        u = fld(coords)
        # Opposing components that would cancel under a naive mean.
        return torch.stack([u, -u], dim=-1)

    result = march_solve(
        field,
        residual_fn,
        cs,
        schedule,
        steps_per_window=3,
        per_bin=2,
        n_slice=4,
        ic_values=np.zeros(4),
        check_trivial=False,
        advance_policy="force",
    )
    assert len(result.windows) == 1
    assert np.isfinite(result.windows[0].final_loss)


def test_decaying_solution_not_false_trivial():
    """Same-time variance guard must not flag a physically decaying field."""
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=1, n_time_bins=4, epsilon=0.1, tolerance=1e-12
    )

    class _Exact(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Keep a leaf so Adam / backward stay well-defined; scale is zero.
            self.dummy = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            return _target(coords) + self.dummy * 0.0

    field = _Exact()
    ic = np.sin(np.pi * np.linspace(0.0, 1.0, 16, endpoint=False))
    result = march_solve(
        field,
        lambda fld, coords: fld(coords) - _target(coords),
        cs,
        schedule,
        steps_per_window=1,
        per_bin=4,
        n_slice=16,
        ic_values=ic,
        check_trivial=True,
        trivial_mode="variance",
        advance_policy="force",
    )
    assert result.trivial is not None
    assert not result.trivial.is_trivial
