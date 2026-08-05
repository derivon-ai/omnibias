# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke integration test: Gaussian wavepacket free-particle TDSE.

Free-particle TDSE: ``i hbar psi_t = -hbar^2/(2m) psi_xx``, with the
canonical analytic solution
``psi(x, t) = (1 / (pi^(1/4) sqrt(alpha))) exp(-x^2 / (2 alpha^2))``
on a 1D + time domain. We train the qpinn for a few steps and verify
the TDSE residual decreases.

Heavy-fidelity numbers (waveform comparison to the closed form
solution at multiple times) live in the GPU benchmark suite (not
shipped publicly; see ``docs/benchmarks.md``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.equations import tdse


@pytest.mark.timeout(60)
def test_tdse_loss_decreases():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x", "t"))
    spec = make_psi_components(name="psi")
    field = OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=32, base="gaussian", dtype=torch.float64,
    )
    # Free-particle TDSE on a small (x, t) grid.
    xs = torch.linspace(-3.0, 3.0, 33, dtype=torch.float64)
    ts = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    grid = torch.cartesian_prod(xs, ts)
    optim = torch.optim.Adam(field.parameters(), lr=5e-3)
    losses: list[float] = []
    for _ in range(50):
        optim.zero_grad()
        state = field(grid)
        out = tdse(state, hbar=1.0, mass=1.0)
        loss = (out.residual ** 2).sum(dim=-1).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


@pytest.mark.timeout(30)
def test_tdse_continuity_residual_finite():
    """Continuity residual diagnostic should be computable on a (x, t) state."""
    from omnibias.qpinn.torch.diagnostics import continuity_residual

    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x", "t"))
    spec = make_psi_components(name="psi")
    field = OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=16, base="gaussian", dtype=torch.float64,
    )
    grid = torch.randn(20, 2, dtype=torch.float64)
    state = field(grid)
    r = continuity_residual(state, hbar=1.0, mass=1.0)
    assert r.shape == (20,)
    assert torch.isfinite(r).all()
