# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX causal-marching window geometry and diagnostic parity."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import TimeWindowSchedule
from omnibias.pinn.train import jax as train_jax
from omnibias.pinn.train import torch as train_torch


class _TorchField(nn.Module):
    def __init__(self, a: float, b: float, c: float) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float64))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float64))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.a * coords[:, 0] + self.b * coords[:, 1] + self.c


def test_march_window_diagnostics_parity():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=2, n_time_bins=4, epsilon=1.0, tolerance=1e-12
    )
    a, b, c = 0.25, -0.1, 0.05
    ic = np.sin(np.pi * np.linspace(0.0, 1.0, 16, endpoint=False))

    torch_field = _TorchField(a, b, c)

    def torch_resid(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        target = torch.sin(np.pi * coords[:, 0]) * torch.exp(-coords[:, 1])
        return fld(coords) - target

    torch_result = train_torch.march_solve(
        torch_field,
        torch_resid,
        cs,
        schedule,
        steps_per_window=8,
        max_steps_per_window=8,
        lr=1e-2,
        per_bin=4,
        n_slice=16,
        ic_values=ic,
        seed=7,
        check_trivial=True,
        advance_policy="force",
        dtype=torch.float64,
    )

    params = {
        "a": jnp.asarray(a, dtype=jnp.float64),
        "b": jnp.asarray(b, dtype=jnp.float64),
        "c": jnp.asarray(c, dtype=jnp.float64),
    }

    def apply(p, coords):
        return p["a"] * coords[:, 0] + p["b"] * coords[:, 1] + p["c"]

    def jax_resid(p, coords):
        target = jnp.sin(np.pi * coords[:, 0]) * jnp.exp(-coords[:, 1])
        return apply(p, coords) - target

    jax_result = train_jax.march_solve(
        params,
        apply,
        jax_resid,
        cs,
        schedule,
        steps_per_window=8,
        max_steps_per_window=8,
        lr=1e-2,
        per_bin=4,
        n_slice=16,
        ic_values=ic,
        seed=7,
        check_trivial=True,
        advance_policy="force",
    )

    assert len(torch_result.windows) == len(jax_result.windows) == 2
    for tw, jw in zip(torch_result.windows, jax_result.windows, strict=True):
        assert tw.window_index == jw.window_index
        assert tw.bounds == jw.bounds
        assert tw.epsilon == jw.epsilon
        assert tw.steps_run == jw.steps_run
        # Shared pure-Python causality report from identical L/w arrays early
        # in training may drift after Adam steps; compare geometry + schema.
        assert tw.causality.n_bins == jw.causality.n_bins
        assert tw.exhausted == jw.exhausted
    assert torch_result.trivial is not None and jax_result.trivial is not None
    assert torch_result.trivial.mode == jax_result.trivial.mode
