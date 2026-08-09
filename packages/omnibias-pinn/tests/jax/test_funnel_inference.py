# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Funnel inference unit tests."""

from __future__ import annotations

import numpy as np
from omnibias.pinn.jax.discovery.funnel import (
    FunnelState,
    funnel_next_lambda,
    run_funnel_loop,
    signed_max_residual_near_origin,
)


def test_funnel_secant_finds_linear_zero() -> None:
    # r(lam) = 2*(lam - 0.5) exactly -> secant should land on 0.5
    state = FunnelState()
    state.record(0.4, 2 * (0.4 - 0.5))
    state.record(0.6, 2 * (0.6 - 0.5))
    lam = funnel_next_lambda(state)
    assert abs(lam - 0.5) < 1e-12


def test_signed_max_near_origin() -> None:
    y = np.array([-1.0, -0.5, 0.0, 0.4, 0.5, 1.0])
    r = np.array([9.0, -0.5, 0.1, 0.2, 0.5, 8.0])
    s = signed_max_residual_near_origin(y, r, radius=0.5)
    # among |y|<=0.5 values [-0.5,0.1,0.2,0.5], max abs is ±0.5
    assert abs(abs(s) - 0.5) < 1e-12


def test_run_funnel_loop() -> None:
    def train_and_residual(lam: float):
        y = np.linspace(-1.0, 1.0, 11)
        # residual proxy: (lam - 0.7) everywhere
        r = np.full_like(y, lam - 0.7)
        return lam, y, r

    state = run_funnel_loop(
        lam0=0.5, train_and_residual=train_and_residual, n_updates=4, delta_lambda=0.05
    )
    assert len(state.lambdas) == 4
    # last recorded residual should approach 0
    assert abs(state.residuals[-1]) < abs(state.residuals[0]) + 1e-9
