# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sharpness-scheduled cubic step, JAX (theory 08-06)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from omnibias.jax.optim_sharpness import (
    SharpnessSchedule,
    sharpness_lambda_max,
    sharpness_scheduled_minimize,
    sharpness_scheduled_step,
)

jax.config.update("jax_enable_x64", True)

STIFF = 1.0e4
G1_SCHEDULE = SharpnessSchedule(n_lanczos=4, c=1e-3, ell_min=1e-6)
SEEDS = (0, 1, 2, 3, 4)


def _stiff_loss(params: Array) -> Array:
    p = jnp.reshape(params, (-1,))
    return 0.5 * (p[0] * p[0] + STIFF * p[1] * p[1])


def _start(seed: int) -> Array:
    return jnp.array([1.0 + 0.02 * seed, 1.0 - 0.01 * seed], dtype=jnp.float64)


def test_lambda_max_stiff_quadratic() -> None:
    ell = sharpness_lambda_max(
        _stiff_loss, jnp.array([1.0, 1.0], dtype=jnp.float64), n_lanczos=4
    )
    assert abs(ell - STIFF) / STIFF <= 1e-12


def test_twenty_steps_reach_loss_floor() -> None:
    params, losses, ells = sharpness_scheduled_minimize(
        _stiff_loss,
        jnp.array([1.0, 1.0], dtype=jnp.float64),
        schedule=G1_SCHEDULE,
        steps=20,
    )
    assert math.isfinite(losses[-1])
    assert bool(jnp.all(jnp.isfinite(params)))
    assert losses[-1] < 1e-8
    assert ells
    assert max(ells) == pytest.approx(STIFF, rel=1e-12)


def test_g1_five_seeds_finite() -> None:
    finite = 0
    finals: list[float] = []
    for seed in SEEDS:
        params, losses, _ells = sharpness_scheduled_minimize(
            _stiff_loss, _start(seed), schedule=G1_SCHEDULE, steps=20
        )
        ok = math.isfinite(losses[-1]) and bool(jnp.all(jnp.isfinite(params)))
        finite += int(ok)
        finals.append(losses[-1])
    assert finite == len(SEEDS)
    assert max(finals) < 1e-8


def test_zero_sharpness_refuses_step() -> None:
    def constant(params: Array) -> Array:
        return 0.0 * jnp.reshape(params, (-1,))[0]

    schedule = SharpnessSchedule(c=1.0, ell_min=0.0, target="cubic_sigma")
    with pytest.raises(ValueError, match="refusing the step"):
        sharpness_scheduled_step(
            constant, jnp.array([1.0, 1.0], dtype=jnp.float64), schedule=schedule
        )
