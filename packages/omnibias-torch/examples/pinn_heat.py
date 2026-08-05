# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""1D heat-equation PINN demo: ``u_t = alpha * u_xx``.

Solves the heat equation on ``(x, t) in [0, 1] x [0, T]`` with
- initial condition ``u(x, 0) = sin(pi x)``,
- Dirichlet boundary ``u(0, t) = u(1, t) = 0``,

whose analytic solution is ``u(x, t) = exp(-alpha pi^2 t) sin(pi x)``.

Run::

    python examples/pinn_heat.py

The demo trains for 2000 iterations on a fixed collocation grid and
reports the L^2 error against the analytic reference. Closed-form
derivatives via :class:`PINNHeat` (no autograd-through-derivative).
"""

from __future__ import annotations

import math
from time import perf_counter

import torch
from omnibias.torch.architectures import PINNHeat


def analytic_solution(x: torch.Tensor, t: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.exp(-alpha * math.pi**2 * t) * torch.sin(math.pi * x)


def main(
    n_collocation: int = 2000,
    n_boundary: int = 200,
    n_initial: int = 200,
    hidden: int = 64,
    base: str = "softplus",
    alpha: float = 0.1,
    iters: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)

    model = PINNHeat(hidden=hidden, base=base, alpha=alpha)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Collocation points (interior of the (x, t) box).
    x_in = torch.rand(n_collocation)
    t_in = torch.rand(n_collocation)

    # Boundary points x in {0, 1}, t uniform.
    x_bd = torch.cat([torch.zeros(n_boundary // 2), torch.ones(n_boundary - n_boundary // 2)])
    t_bd = torch.rand(n_boundary)

    # Initial condition x uniform, t = 0.
    x_ic = torch.rand(n_initial)
    t_ic = torch.zeros(n_initial)
    u_ic_target = torch.sin(math.pi * x_ic)

    print(f"PINNHeat: hidden={hidden}, base={base}, alpha={alpha}, iters={iters}")
    print(f"params: {sum(p.numel() for p in model.parameters())}")
    t0 = perf_counter()

    for it in range(1, iters + 1):
        optim.zero_grad()
        _, res_in = model(x_in, t_in)
        u_bd, _ = model(x_bd, t_bd)
        u_ic, _ = model(x_ic, t_ic)

        loss_pde = (res_in**2).mean()
        loss_bd = (u_bd**2).mean()
        loss_ic = ((u_ic - u_ic_target) ** 2).mean()
        loss = loss_pde + 10.0 * loss_bd + 10.0 * loss_ic

        loss.backward()
        optim.step()

        if it == 1 or it % 200 == 0:
            print(
                f"  iter {it:4d}  loss={loss.item():.4e}  "
                f"pde={loss_pde.item():.4e}  bd={loss_bd.item():.4e}  ic={loss_ic.item():.4e}"
            )

    elapsed = perf_counter() - t0
    print(f"trained in {elapsed:.2f}s")

    # Evaluate L^2 error on a fine grid.
    nx, nt = 64, 64
    xs = torch.linspace(0, 1, nx)
    ts = torch.linspace(0, 1, nt)
    XX, TT = torch.meshgrid(xs, ts, indexing="ij")
    with torch.no_grad():
        u_pred, _ = model(XX.flatten(), TT.flatten())
    u_true = analytic_solution(XX.flatten(), TT.flatten(), alpha=alpha)
    rel_l2 = ((u_pred - u_true) ** 2).mean().sqrt() / ((u_true**2).mean().sqrt() + 1e-12)
    print(f"relative L2 error vs analytic: {rel_l2.item():.4e}")


if __name__ == "__main__":
    main()
