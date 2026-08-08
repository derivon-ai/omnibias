# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the JAX causal marching driver."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import TimeWindowSchedule
from omnibias.pinn.train.jax import march_solve


def _apply(params, coords):
    return (
        params["a"] * coords[:, 0]
        + params["b"] * coords[:, 1]
        + params["c"]
    )


def _target(coords):
    return jnp.sin(np.pi * coords[:, 0]) * jnp.exp(-coords[:, 1])


def test_march_solve_jax_runs():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=2, n_time_bins=4, epsilon=1.0, tolerance=1e-6
    )
    params = {
        "a": jnp.asarray(0.1),
        "b": jnp.asarray(0.0),
        "c": jnp.asarray(0.0),
    }
    ic = np.sin(np.pi * np.linspace(0.0, 1.0, 16, endpoint=False))
    result = march_solve(
        params,
        _apply,
        lambda p, coords: _apply(p, coords) - _target(coords),
        cs,
        schedule,
        steps_per_window=5,
        max_steps_per_window=20,
        lr=1e-2,
        per_bin=4,
        n_slice=16,
        ic_values=ic,
        seed=0,
        advance_policy="force",
    )
    assert len(result.windows) == 2
    assert result.windows[0].seam_mse is not None
    assert result.trivial is not None
    assert not result.trivial.is_trivial


def test_march_solve_jax_requires_ic():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 0.5)), time_axis="t"
    )
    schedule = TimeWindowSchedule(0.0, 0.5, n_windows=1, n_time_bins=2)
    with pytest.raises(ValueError, match="ic_values or ic_fn"):
        march_solve(
            {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0), "c": jnp.asarray(0.0)},
            _apply,
            lambda p, coords: _apply(p, coords),
            cs,
            schedule,
            steps_per_window=1,
            per_bin=2,
            n_slice=4,
            check_trivial=False,
        )


def test_march_solve_jax_gate_exhaustion():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=3, n_time_bins=4, epsilon=100.0, tolerance=0.999
    )
    params = {
        "a": jnp.asarray(0.1),
        "b": jnp.asarray(0.0),
        "c": jnp.asarray(0.0),
    }
    ic = np.sin(np.pi * np.linspace(0.0, 1.0, 8, endpoint=False))
    result = march_solve(
        params,
        _apply,
        lambda p, coords: _apply(p, coords) + 1.0,
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


def test_march_solve_jax_vector_residual():
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 0.5)), time_axis="t"
    )
    schedule = TimeWindowSchedule(0.0, 0.5, n_windows=1, n_time_bins=2)
    params = {
        "a": jnp.asarray(0.1),
        "b": jnp.asarray(0.0),
        "c": jnp.asarray(0.0),
    }

    def residual_fn(p, coords):
        u = _apply(p, coords)
        return jnp.stack([u, -u], axis=-1)

    result = march_solve(
        params,
        _apply,
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
    assert np.isfinite(result.windows[0].final_loss)

