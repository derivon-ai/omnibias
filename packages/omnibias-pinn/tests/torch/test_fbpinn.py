# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for FBPINNField, NTK eigenspectrum, and SpectralBandScheduler."""

from __future__ import annotations

import numpy as np
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.train import SpectralBandScheduler
from omnibias.pinn.torch.fields import FBPINNField, build_fbpinn_field
from omnibias.pinn.torch.losses import ntk_eigenspectrum, spectral_bias_index


def test_fbpinn_window_weights_sum_to_one():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = build_fbpinn_field(
        coordinate_spec=cs, components=comps, n_windows=4, overlap=0.5, hidden=4
    )
    x = torch.linspace(0.05, 0.95, 64, dtype=torch.float64).unsqueeze(-1)
    w = field.window_weights(x)
    assert torch.allclose(w.sum(dim=-1), torch.ones(64, dtype=torch.float64), atol=1e-10)


def test_fbpinn_forward_finite():
    cs = CoordinateSpec(("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t")
    comps = ComponentSpec(("u",))
    field = FBPINNField(
        coordinate_spec=cs, components=comps, n_windows=3, hidden=8
    )
    coords = torch.rand(32, 2, dtype=torch.float64)
    u = field.forward_values(coords)
    assert u.shape == (32, 1)
    assert torch.isfinite(u).all()


def test_spectral_band_scheduler_grows():
    sched = SpectralBandScheduler(n_bands_max=4, n_bands_init=2, L=1.0)
    assert len(sched.bands) == 2
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    # Residual with energy at a high mode.
    resid = np.sin(2 * np.pi * 20 * x)[None, :]
    bands = sched.observe(resid)
    assert len(bands) >= 2
    assert len(bands) <= 4


def test_ntk_eigenspectrum_and_bias_index():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = build_fbpinn_field(
        coordinate_spec=cs, components=comps, n_windows=2, hidden=4
    )
    coords = torch.linspace(0.0, 1.0, 16, dtype=torch.float64).unsqueeze(-1)

    def residual_fn():
        return field.forward_values(coords)[:, 0] - torch.sin(2 * np.pi * coords[:, 0])

    evals = ntk_eigenspectrum(residual_fn, list(field.parameters()), n_eigen=8)
    assert evals.ndim == 1
    assert evals.numel() >= 2
    # Descending.
    assert torch.all(evals[:-1] >= evals[1:] - 1e-10)
    idx = spectral_bias_index(evals)
    assert 0.0 <= idx <= 1.0
