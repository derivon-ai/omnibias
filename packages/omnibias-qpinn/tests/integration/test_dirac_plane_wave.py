# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke integration test: 1+1d massive Dirac equation.

Trains a tiny spinor field on the Dirac residual for a couple of
optimizer steps. The full plane-wave fidelity test (comparing to
``psi_p(x, t) = u(p) exp(-i p . x)`` for various p) lives in the
internal qpinn benchmark suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_spinor_components
from omnibias.qpinn.torch.equations import dirac


@pytest.mark.timeout(60)
def test_dirac_loss_decreases():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x", "t"))
    spec = make_spinor_components(name="spinor", n_components=4)
    field = OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=32, base="tanh", dtype=torch.float64,
    )
    xs = torch.linspace(-3.0, 3.0, 17, dtype=torch.float64)
    ts = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    grid = torch.cartesian_prod(xs, ts)
    optim = torch.optim.Adam(field.parameters(), lr=5e-3)
    losses: list[float] = []
    for _ in range(50):
        optim.zero_grad()
        state = field(grid)
        out = dirac(state, mass=1.0, representation="dirac")
        loss = (out.residual ** 2).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
