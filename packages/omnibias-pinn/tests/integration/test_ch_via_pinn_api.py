# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity test: new ``omnibias.pinn`` API reproduces the existing
Cahn-Hilliard solver's residual numerics in 1D / 2D / 3D.

NON-MODIFYING: only imports from
``research.experiments.cahn_hilliard.physics``.

This test depends on the project's private ``research/`` tree and is
deliberately skipped in clean public installs.
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

pytestmark = pytest.mark.needs_research
pytest.importorskip(
    "research.experiments.cahn_hilliard.physics",
    reason="research.experiments tree is private and not shipped publicly",
)

from research.experiments.cahn_hilliard.physics import (  # noqa: E402
    CahnHilliardParams,
    cahn_hilliard_residual,
)
from research.experiments.cahn_hilliard.physics import (
    GinzburgLandauPotential as RGinzburgLandauPotential,
)


def _build_c_field(D_spatial: int, *, K: int = 4, time_hidden: int = 8, seed: int = 13):
    spatial_axes = ("x", "y", "z")[:D_spatial]
    domain = ((0.0, 2.0 * math.pi),) * D_spatial + ((0.0, 1.0),)
    coord = CoordinateSpec(
        axes=spatial_axes + ("t",),
        periodicity=(True,) * D_spatial + (False,),
        domain=domain,
        time_axis="t",
    )
    components = ComponentSpec(names=("c",), groups={})
    torch.manual_seed(seed)
    return SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=time_hidden, time_depth=1,
        activation="tanh", dtype=torch.float64,
    )


def _grid_collocation(D_spatial: int, *, n_xy: int, n_t: int, L: float, T: float):
    spatial = []
    for _ in range(D_spatial):
        spatial.append(torch.linspace(0.0, L, n_xy + 1, dtype=torch.float64)[:-1])
    grids = torch.meshgrid(*spatial, indexing="ij")
    flat_spatial = [g.flatten() for g in grids]
    n_pts = flat_spatial[0].shape[0]
    t_samples = torch.linspace(0.0, T, n_t + 1, dtype=torch.float64)[:-1]
    rep_spatial = [s.unsqueeze(0).expand(n_t, -1).reshape(-1) for s in flat_spatial]
    rep_t = t_samples.view(-1, 1).expand(n_t, n_pts).reshape(-1)
    return torch.stack(rep_spatial + [rep_t], dim=-1)


def _research_ch_residual(state, *, M: float, kappa: float) -> torch.Tensor:
    c = tops.value(state, "c")
    grad_c = tops.gradient(state, "c")
    lap_c = tops.laplacian(state, "c")
    bih_c = tops.biharmonic(state, "c")
    c_t = tops.derivative(state, "c", axis="t", order=1)
    return cahn_hilliard_residual(
        c, grad_c, lap_c, bih_c, c_t,
        RGinzburgLandauPotential(W=1.0),
        CahnHilliardParams(M=M, kappa=kappa),
    )


@pytest.mark.parametrize("D_spatial", [1, 2, 3])
def test_ch_residual_parity_with_research_physics(D_spatial):
    field = _build_c_field(D_spatial, K=3, time_hidden=8)
    n_xy, n_t = 6, 3
    coords = _grid_collocation(
        D_spatial, n_xy=n_xy, n_t=n_t, L=2.0 * math.pi, T=0.4,
    )
    state = field(coords)

    M, kappa = 1.5, 2e-3
    new = eq.CahnHilliard(
        M=M, kappa=kappa,
        potential=eq.GinzburgLandauPotential(W=1.0),
    )(state)
    ref = _research_ch_residual(state, M=M, kappa=kappa)
    assert torch.allclose(new.residual, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("D_spatial", [1, 2, 3])
def test_ch_loss_parity_plain_mse(D_spatial):
    field = _build_c_field(D_spatial, K=3, time_hidden=8)
    n_xy, n_t = 6, 3
    coords = _grid_collocation(
        D_spatial, n_xy=n_xy, n_t=n_t, L=2.0 * math.pi, T=0.4,
    )
    state = field(coords)
    M, kappa = 1.0, 1e-3
    new = eq.CahnHilliard(M=M, kappa=kappa)(state)
    ref = _research_ch_residual(state, M=M, kappa=kappa)
    new_mse = (new.residual * new.residual).mean()
    ref_mse = (ref * ref).mean()
    assert torch.allclose(new_mse, ref_mse, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("D_spatial", [1, 2])
def test_ch_smoke_training_step_with_pinn_api(D_spatial):
    torch.manual_seed(2026)
    field = _build_c_field(D_spatial, K=3, time_hidden=8, seed=14 + D_spatial)
    coords = _grid_collocation(
        D_spatial, n_xy=6, n_t=3, L=2.0 * math.pi, T=0.4,
    )

    optim = torch.optim.Adam(field.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(3):
        optim.zero_grad()
        state = field(coords)
        out = eq.CahnHilliard(M=1.0, kappa=1e-3)(state)
        loss = (out.residual * out.residual).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))

    assert all(np.isfinite(losses)), losses
    assert losses[-1] <= losses[0] * 1.5
