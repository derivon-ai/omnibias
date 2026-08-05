# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias PINN quickstart: 1D heat equation.

Run:

    pip install omnibias-torch
    python docs/examples/pinn_heat.py

Trains the closed-form-Laplacian ``PINNHeat`` reference architecture on
the 1D heat equation ``u_t = alpha * u_xx``. The spatial second
derivative is closed form (no autograd through the activation), so the
residual is bit-stable.
"""

from __future__ import annotations

import torch
from omnibias.torch.architectures import PINNHeat


def main() -> None:
    torch.manual_seed(0)

    net = PINNHeat(hidden=64, base="softplus", alpha=0.1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Interior collocation points on (x, t) in [0, 1] x [0, 0.1].
    n = 512
    for step in range(200):
        x = torch.rand(n)
        t = torch.rand(n) * 0.1

        _, residual = net(x, t)
        pde_loss = (residual**2).mean()

        # Initial condition u(x, 0) = sin(pi x).
        x0 = torch.linspace(0.0, 1.0, 64)
        u0, _ = net(x0, torch.zeros_like(x0))
        ic_loss = ((u0 - torch.sin(torch.pi * x0)) ** 2).mean()

        loss = pde_loss + ic_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step {step:4d}  pde={pde_loss.item():.3e}  ic={ic_loss.item():.3e}")

    print("done.")


if __name__ == "__main__":
    main()
