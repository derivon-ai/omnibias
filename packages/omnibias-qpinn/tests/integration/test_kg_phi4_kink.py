# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke integration test: 1+1d Klein-Gordon with phi^4 self-interaction.

The phi^4 theory in 1+1 dimensions has the well-known kink solution
``phi_kink(x) = tanh(m x / sqrt(2))`` for the action
``S = int [1/2 (d_mu phi)^2 + (m^2/4)(phi^2 - 1)^2]`` (after rescaling).
We don't compare to the analytic profile here -- this is a smoke test
that the qpinn KG residual trains and the loss decreases.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn.torch.equations import klein_gordon


@pytest.mark.timeout(60)
def test_kg_phi4_loss_decreases():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x", "t"))
    spec = ComponentSpec(("phi",))
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
        out = klein_gordon(state, mass=1.0, lambda_phi4=0.5)
        loss = (out.residual ** 2).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
