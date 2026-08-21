# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-07 field consumers: curriculum schedule and grid-free V-cycle."""

from __future__ import annotations

import numpy as np
from omnibias.core.scale import ScaleBand, ScaledPack, stiffness_matrix
from omnibias.core.spectral_design import peak_frequency
from omnibias.fields.scale import (
    _jacobi,
    grid_free_vcycle,
    lstsq_readout_mse,
    residual_norm,
    scale_schedule,
)


def test_schedule_tracks_target_band() -> None:
    def band(t: float) -> float:
        return 2.0 + 14.0 * t

    alphas = scale_schedule(target_band=band, steps=5, base="gaussian", order=1)
    assert len(alphas) == 5
    for t, a in zip(np.linspace(0.0, 1.0, 5), alphas, strict=True):
        assert abs(peak_frequency("gaussian", 1, a) - band(float(t))) <= 1e-12


def test_g5_vcycle_reduces_residual() -> None:
    packs = (
        ScaledPack(order=0, mean=-0.8, alpha=0.8),
        ScaledPack(order=0, mean=0.8, alpha=0.8),
        ScaledPack(order=0, mean=-0.4, alpha=2.0),
        ScaledPack(order=0, mean=0.0, alpha=2.0),
        ScaledPack(order=0, mean=0.4, alpha=2.0),
    )
    a = -stiffness_matrix(packs, derivative_order=2)
    rng = np.random.default_rng(0)
    true = rng.normal(size=len(packs))
    rhs = a @ true
    bands = (ScaleBand(0.5, 1.0), ScaleBand(1.5, 2.5))
    u0 = np.zeros(len(packs))
    single = _jacobi(a, rhs, u0.copy(), omega=0.6, sweeps=4)
    cycled = np.asarray(
        grid_free_vcycle(packs, rhs, bands=bands, u0=u0, pre_sweeps=2, post_sweeps=2),
        dtype=np.float64,
    )
    r0 = float(np.linalg.norm(rhs))
    r_single = residual_norm(packs, rhs, single)
    r_cycle = residual_norm(packs, rhs, cycled)
    assert r_cycle <= r0 / 5.0
    assert r_cycle < r_single


def test_g4_derived_schedule_matches_or_beats_hand_tuned() -> None:
    freq = 8
    x = np.linspace(0.0, 1.0, 256)
    y = np.sin(2.0 * np.pi * freq * x)
    hand = (1.0, 2.0, 4.0)
    derived = scale_schedule(
        target_band=lambda t: 2.0 * np.pi * (1.0 + (freq - 1) * t),
        steps=3,
        base="gaussian",
        order=1,
    )
    mse_hand = lstsq_readout_mse(x, y, hand, n_means=16)
    mse_derived = lstsq_readout_mse(x, y, derived, n_means=16)
    assert mse_derived <= mse_hand * 1.0000001
