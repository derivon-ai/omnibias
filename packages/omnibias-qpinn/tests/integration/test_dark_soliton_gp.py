# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke integration test: 1D Gross-Pitaevskii dark soliton.

The 1D defocusing Gross-Pitaevskii equation
``i hbar psi_t = -hbar^2/(2m) psi_xx + g |psi|^2 psi``
with ``g > 0`` admits a stationary dark-soliton solution
``psi(x, t) = sqrt(rho_0) tanh(x / sqrt(2 xi)) e^(-i mu t)`` where
``mu = g rho_0`` and ``xi = hbar / sqrt(2 m g rho_0)``.

This test checks the qpinn NLS pipeline runs end-to-end and the loss
decreases. The full quantitative comparison to the analytic profile
lives in the GPU benchmark suite (not shipped publicly; see
``docs/benchmarks.md``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.equations import nls


@pytest.mark.timeout(60)
def test_nls_loss_decreases_repulsive():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x", "t"))
    spec = make_psi_components(name="psi")
    field = OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=32, base="tanh", dtype=torch.float64,
    )
    xs = torch.linspace(-4.0, 4.0, 33, dtype=torch.float64)
    ts = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    grid = torch.cartesian_prod(xs, ts)
    optim = torch.optim.Adam(field.parameters(), lr=5e-3)
    losses: list[float] = []
    for _ in range(50):
        optim.zero_grad()
        state = field(grid)
        out = nls(state, g=1.0, hbar=1.0, mass=1.0)
        loss = (out.residual ** 2).sum(dim=-1).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


@pytest.mark.timeout(30)
def test_nls_density_diagnostic_consistent():
    """``nls`` output's ``mean_density`` diag should match the diag from
    direct computation."""
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x", "t"))
    spec = make_psi_components(name="psi")
    field = OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=16, base="gaussian", dtype=torch.float64,
    )
    coords = torch.randn(16, 2, dtype=torch.float64)
    state = field(coords)
    out = nls(state, g=1.0)
    psi_re = state.ops.value(state, "psi_re")
    psi_im = state.ops.value(state, "psi_im")
    expected_density = (psi_re ** 2 + psi_im ** 2).mean()
    assert abs(out.diag["mean_density"] - float(expected_density.detach())) < 1e-12
