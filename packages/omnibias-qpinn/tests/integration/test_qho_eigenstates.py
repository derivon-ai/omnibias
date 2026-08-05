# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke integration test: 1D quantum harmonic oscillator ground state.

We train a tiny :class:`OneLayerVectorField` on the TISE residual
with the harmonic potential ``V(x) = 1/2 x^2``, plus a soft norm
constraint, and verify that:

1. The loss decreases monotonically (or at least decreases overall).
2. The final residual is below a loose threshold.

This is **smoke**: full convergence to the known ground-state energy
``E_0 = 1/2`` is checked in the heavy GPU benchmark suite (not shipped
publicly; see ``docs/benchmarks.md``); here we just check the
pipeline works on CPU in a few seconds.

For the *exact* ground state ``psi_0(x) = pi^(-1/4) exp(-x^2/2)`` the
TISE residual with ``E = 1/2`` is identically zero, so the model has
something to converge to.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.cage import norm_loss
from omnibias.qpinn.torch.equations import tise


def _build_field():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x",))
    spec = make_psi_components(name="psi")
    return OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=32, base="gaussian", dtype=torch.float64,
    )


def _harmonic_V(state):
    return 0.5 * state.coords[..., 0] ** 2


@pytest.mark.timeout(60)
def test_qho_loss_decreases():
    """Training the TISE residual must reduce the loss substantially."""
    field = _build_field()
    coords = torch.linspace(-4.0, 4.0, 65, dtype=torch.float64).unsqueeze(-1)
    weights = torch.full((65,), 8.0 / 65, dtype=torch.float64)
    optim = torch.optim.Adam(field.parameters(), lr=5e-3)
    losses: list[float] = []
    for _ in range(50):
        optim.zero_grad()
        state = field(coords)
        out = tise(
            state, energy=0.5, potential=_harmonic_V,
            quadrature_weights=weights,
        )
        residual_loss = (out.residual ** 2).sum(dim=-1).mean()
        norm_pen = norm_loss(state, quadrature_weights=weights, target_norm=1.0)
        loss = residual_loss + 10.0 * norm_pen
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], (
        f"loss did not decrease: start={losses[0]:.4e} end={losses[-1]:.4e}"
    )
    # Loose threshold: the residual is bounded above 1e-3 in this short
    # training run; final loss should be well below the initial value.
    assert losses[-1] < losses[0] * 0.5


@pytest.mark.timeout(60)
def test_qho_final_energy_estimate_finite():
    """End-to-end pipeline check: build field, evaluate, get an energy estimate."""
    field = _build_field()
    coords = torch.linspace(-4.0, 4.0, 65, dtype=torch.float64).unsqueeze(-1)
    weights = torch.full((65,), 8.0 / 65, dtype=torch.float64)
    state = field(coords)
    out = tise(
        state, energy=0.5, potential=_harmonic_V,
        quadrature_weights=weights,
    )
    assert out.energy_estimate is not None
    assert torch.isfinite(out.energy_estimate)
