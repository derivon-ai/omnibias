# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity test: new ``omnibias.pinn`` API reproduces the existing KS
solver's residual numerics on a pinned smoke config.

NON-MODIFYING: only imports from
``research.experiments.kuramoto_sivashinsky.{physics,solvers._causal}``.

This test depends on the project's private ``research/`` tree and is
deliberately skipped in clean public installs (the tree is not shipped
on PyPI / GitHub). Run inside the internal repo with ``-m
needs_research`` to enable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch import equations as eq
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields.spectral import SpectralVectorField
from omnibias.pinn.torch.losses import causal_residual_loss

pytestmark = pytest.mark.needs_research
pytest.importorskip(
    "research.experiments.kuramoto_sivashinsky.physics",
    reason="research.experiments tree is private and not shipped publicly",
)
pytest.importorskip(
    "research.experiments.kuramoto_sivashinsky.solvers._causal",
    reason="research.experiments tree is private and not shipped publicly",
)

from research.experiments.kuramoto_sivashinsky.physics import (  # noqa: E402
    KSParams,
    ks_residual,
)
from research.experiments.kuramoto_sivashinsky.solvers._causal import (  # noqa: E402
    causal_residual_loss_fourier_1d,
)


def _build_u_field_1d(K: int = 6, time_hidden: int = 8, *, seed: int = 11):
    coord = CoordinateSpec(
        axes=("x", "t"),
        periodicity=(True, False),
        domain=((0.0, 22.0), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(seed)
    return SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=time_hidden, time_depth=1,
        L=22.0, activation="tanh", dtype=torch.float64,
    )


def _grid_collocation_1d(n_x: int, n_t: int, L: float, T: float):
    x = torch.linspace(0.0, L, n_x + 1, dtype=torch.float64)[:-1]
    t_samples = torch.linspace(0.0, T, n_t + 1, dtype=torch.float64)[:-1]
    x_b = x.unsqueeze(0).expand(n_t, n_x).reshape(-1)
    t_b = t_samples.view(-1, 1).expand(n_t, n_x).reshape(-1)
    return torch.stack([x_b, t_b], dim=-1)


def _research_ks_residual(state, *, params: KSParams) -> torch.Tensor:
    u = tops.value(state, "u")
    u_x = tops.derivative(state, "u", axis="x", order=1)
    u_xx = tops.derivative(state, "u", axis="x", order=2)
    u_xxxx = tops.derivative(state, "u", axis="x", order=4)
    u_t = tops.derivative(state, "u", axis="t", order=1)
    return ks_residual(u, u_x.unsqueeze(-1), u_xx, u_xxxx, u_t, params)


def test_ks_residual_parity_with_research_physics():
    field = _build_u_field_1d(K=6, time_hidden=8)
    coords = _grid_collocation_1d(n_x=24, n_t=4, L=22.0, T=0.4)
    state = field(coords)

    new = eq.KuramotoSivashinsky(form="1d")(state)
    ref = _research_ks_residual(state, params=KSParams())
    assert torch.allclose(new.residual, ref, rtol=1e-12, atol=1e-12)


def test_ks_loss_parity_plain_mse():
    field = _build_u_field_1d(K=6, time_hidden=8)
    coords = _grid_collocation_1d(n_x=24, n_t=4, L=22.0, T=0.4)
    state = field(coords)
    new = eq.KuramotoSivashinsky(form="1d")(state)
    ref = _research_ks_residual(state, params=KSParams())
    new_mse = (new.residual * new.residual).mean()
    ref_mse = (ref * ref).mean()
    assert torch.allclose(new_mse, ref_mse, rtol=1e-12, atol=1e-12)


def test_ks_sobolev_causal_loss_matches_research_helper():
    field = _build_u_field_1d(K=6, time_hidden=8)
    n_x, n_t = 32, 4
    coords = _grid_collocation_1d(n_x=n_x, n_t=n_t, L=22.0, T=0.4)
    state = field(coords)
    new = eq.KuramotoSivashinsky(form="1d")(state)
    res_2d = new.residual.reshape(n_t, n_x)

    eps, sob_p, L = 1.5, 1.0, 22.0
    research_loss = causal_residual_loss_fourier_1d(
        res_2d, L=L, sobolev_p=sob_p, epsilon=eps,
    )
    new_loss = causal_residual_loss(
        res_2d, epsilon=eps, L=L, sobolev_p=sob_p,
    )
    # Same fftfreq-precision drift as the NS-2D parity test.
    assert torch.allclose(new_loss, research_loss, rtol=1e-6, atol=1e-9)


def test_ks_smoke_training_step_with_pinn_api():
    torch.manual_seed(2026)
    field = _build_u_field_1d(K=6, time_hidden=8, seed=12)
    coords = _grid_collocation_1d(n_x=24, n_t=4, L=22.0, T=0.4)

    optim = torch.optim.Adam(field.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(3):
        optim.zero_grad()
        state = field(coords)
        out = eq.KuramotoSivashinsky(form="1d")(state)
        loss = (out.residual * out.residual).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))

    assert all(np.isfinite(losses)), losses
    assert losses[-1] <= losses[0] * 1.5
