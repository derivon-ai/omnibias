# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias QPINN quickstart: harmonic-oscillator ground state (TISE).

Run:

    pip install omnibias-qpinn[torch]
    python docs/examples/qpinn_tise_qho.py

Trains a small one-layer field to recover the quantum harmonic
oscillator ground state psi_0(x) = pi^(-1/4) exp(-x^2/2) and its energy
E_0 = 1/2 (atomic units). Runs on CPU in well under a minute.
"""

from __future__ import annotations

import torch
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.cage import norm_loss
from omnibias.qpinn.torch.equations import tise


def main() -> None:
    torch.manual_seed(0)

    coord = CoordinateSpec(("x",))
    spec = make_psi_components(name="psi")
    field = OneLayerVectorField(
        coordinate_spec=coord,
        components=spec,
        hidden=32,
        base="gaussian",
        dtype=torch.float64,
    )

    # Trapezoidal quadrature on a [-4, 4] box.
    xs = torch.linspace(-4.0, 4.0, 401, dtype=torch.float64).unsqueeze(-1)
    ws = torch.full((401,), 8.0 / 401, dtype=torch.float64)

    def potential(state):  # V(x) = 1/2 x^2
        return 0.5 * state.coords[..., 0] ** 2

    optim = torch.optim.Adam(field.parameters(), lr=5e-3)
    for step in range(2000):
        optim.zero_grad()
        state = field(xs)
        out = tise(state, energy=0.5, potential=potential, quadrature_weights=ws)
        residual_loss = (out.residual**2).sum(dim=-1).mean()
        norm_pen = norm_loss(state, quadrature_weights=ws, target_norm=1.0)
        loss = residual_loss + 10.0 * norm_pen
        loss.backward()
        optim.step()
        if step % 500 == 0:
            e_est = float(out.energy_estimate.detach())
            print(f"step {step:5d}  loss={float(loss):.3e}  E_est={e_est:.6f}")

    print("target E_0 = 0.5")


if __name__ == "__main__":
    main()
