# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for FBPINNField, NTK eigenspectrum, and SpectralBandScheduler."""

from __future__ import annotations

import numpy as np
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.fbpinn import default_multilevel_specs
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.fields import (
    FBPINNField,
    MscaleVectorField,
    OneLayerVectorField,
    build_fbpinn_field,
)
from omnibias.pinn.torch.losses import (
    empirical_jacobian,
    fourier_mode_learning_rates,
    kernel_task_alignment,
    ntk_eigenspectrum,
    ntk_tail_head_index,
    spectral_bias_index,
)
from omnibias.pinn.train import SpectralBandScheduler


def test_fbpinn_window_weights_sum_to_one():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = build_fbpinn_field(
        coordinate_spec=cs, components=comps, n_windows=4, overlap=0.5, hidden=4
    )
    x = torch.linspace(0.05, 0.95, 64, dtype=torch.float64).unsqueeze(-1)
    w = field.window_weights(x)
    assert torch.allclose(w.sum(dim=-1), torch.ones(64, dtype=torch.float64), atol=1e-10)


def test_fbpinn_multilevel_has_multiple_levels():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = build_fbpinn_field(
        coordinate_spec=cs,
        components=comps,
        n_levels=3,
        hidden=4,
    )
    assert field.n_levels == 3
    assert field.n_windows == 1 + 2 + 4


def test_fbpinn_forward_finite():
    cs = CoordinateSpec(("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t")
    comps = ComponentSpec(("u",))
    field = FBPINNField(
        coordinate_spec=cs, components=comps, n_windows=3, hidden=8
    )
    coords = torch.rand(32, 2)
    u = field.forward_values(coords)
    assert u.shape == (32, 1)
    assert torch.isfinite(u).all()


def test_fbpinn_derivative_matches_finite_difference():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = build_fbpinn_field(
        coordinate_spec=cs,
        components=comps,
        level_specs=default_multilevel_specs(2, base_windows=1),
        hidden=6,
    )
    x = torch.linspace(0.1, 0.9, 9, dtype=torch.float64).unsqueeze(-1)
    ux = ops.derivative(field(x), "u", axis=0, order=1)
    h = 1e-5
    with torch.no_grad():
        fp = field.forward_values(x + h)[:, 0]
        fm = field.forward_values(x - h)[:, 0]
        fd = (fp - fm) / (2 * h)
    assert torch.allclose(ux, fd, rtol=2e-2, atol=1e-2)


def test_spectral_band_scheduler_grows_and_resumes():
    sched = SpectralBandScheduler(n_bands_max=4, n_bands_init=2, L=1.0)
    assert len(sched.bands) == 2
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    resid = np.sin(2 * np.pi * 20 * x)[None, :]
    bands = sched.observe(resid)
    assert len(bands) >= 2
    assert len(bands) <= 4
    state = sched.state_dict()
    sched2 = SpectralBandScheduler(n_bands_max=4, n_bands_init=2, L=1.0)
    sched2.load_state_dict(state)
    assert sched2.bands == sched.bands
    assert sched2._calls == sched._calls


def test_spectral_band_scheduler_applies_to_mscale():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    sched = SpectralBandScheduler(
        n_bands_max=4,
        n_bands_init=4,
        L=1.0,
        update_steps=(1,),
    )
    field = MscaleVectorField(
        coordinate_spec=cs,
        components=comps,
        hidden=16,
        depth=1,
        scales=sched.bands,
    )
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    resid = np.sin(2 * np.pi * 20 * x)[None, :]
    sched.step(resid, field, step=1)
    assert len(field.scales) == len(sched.bands)


def test_ntk_eigenspectrum_nonzero_linear_field():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = OneLayerVectorField(
        coordinate_spec=cs, components=comps, hidden=2, base="tanh", dtype=torch.float64
    )
    coords = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
    target = torch.zeros(2, dtype=torch.float64)

    def residual_fn():
        return field.forward_values(coords)[:, 0] - target

    evals = ntk_eigenspectrum(residual_fn, list(field.parameters()), n_eigen=2)
    assert evals.numel() >= 1
    assert torch.all(evals > 0.0)
    j = empirical_jacobian(residual_fn, list(field.parameters()))
    evals_jjt = torch.linalg.eigvalsh(j @ j.T)
    evals_jjt = torch.sort(evals_jjt, descending=True).values
    assert torch.allclose(evals_jjt[: evals.numel()], evals, atol=1e-6)


def test_fourier_mode_rates_and_spectral_bias_index():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = build_fbpinn_field(
        coordinate_spec=cs, components=comps, n_windows=2, hidden=4
    )
    coords = torch.linspace(0.0, 1.0, 16, dtype=torch.float64).unsqueeze(-1)
    modes = (2, 8)

    def residual_fn():
        return field.forward_values(coords)[:, 0] - torch.sin(
            2 * torch.pi * 8 * coords[:, 0]
        )

    rates = fourier_mode_learning_rates(
        residual_fn, list(field.parameters()), coords=coords, modes=modes, L=1.0
    )
    assert rates.numel() == 2
    assert torch.all(rates >= 0.0)
    align = kernel_task_alignment(rates, (0.1, 1.0))
    assert 0.0 <= align <= 1.0
    idx = spectral_bias_index(rates)
    assert 0.0 <= idx <= 1.0
    evals = ntk_eigenspectrum(residual_fn, list(field.parameters()), n_eigen=4)
    legacy = ntk_tail_head_index(evals)
    assert 0.0 <= legacy <= 1.0
