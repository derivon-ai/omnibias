# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke integration test: 2D Helmholtz scattering off a Gaussian bump.

Trains a small qpinn on the Helmholtz residual with a position-dependent
``k(x)`` and verifies the loss decreases. The full-fidelity scattering
benchmark (boundary integral comparison) lives in the internal qpinn
benchmark suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.equations import helmholtz


@pytest.mark.timeout(60)
def test_helmholtz_loss_decreases():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x", "y"))
    spec = make_psi_components(name="psi")
    field = OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=32, base="gaussian", dtype=torch.float64,
    )
    xs = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64)
    ys = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64)
    grid = torch.cartesian_prod(xs, ys)

    def k_callable(s):
        # Index of refraction with a Gaussian bump at origin.
        x = s.coords[..., 0]
        y = s.coords[..., 1]
        return 1.0 + 0.5 * torch.exp(-x * x - y * y)

    optim = torch.optim.Adam(field.parameters(), lr=5e-3)
    losses: list[float] = []
    for _ in range(50):
        optim.zero_grad()
        state = field(grid)
        out = helmholtz(state, k=k_callable)
        loss = (out.residual ** 2).sum(dim=-1).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
