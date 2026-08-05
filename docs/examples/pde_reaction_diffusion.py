# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.pinn.solver quickstart: a coupled PDE system, two ways.

Run:

    pip install "omnibias-pinn[torch] @ ."   # from a checkout, with torch extra
    python docs/examples/pde_reaction_diffusion.py

Part 1 -- steady, linear, scalar (Poisson). Solve ``Delta u = f`` on the unit
square by **exact-operator linear collocation**: the Laplacian is the closed-form
``sigma``-tower operator (no autograd through the activation), so for a linear
PDE the residual is affine in the readout weights and one least-squares solve
fits the field. We check it against a manufactured solution.

Part 2 -- time-dependent, COUPLED system (two species, reaction-diffusion). Solve
``u_t = Du u_xx + R_u`` and ``v_t = Dv v_xx + R_v`` on a periodic line by the
spectral **method-of-lines**. With a mass-preserving exchange reaction
``R_u = -k (u - v)``, ``R_v = +k (u - v)`` the total mass ``int (u + v) dx`` is
conserved -- a clean diagnostic that the coupled march is doing the right thing.

Everything is labelled honestly: the spatial operators are closed-form /
spectral; the time march (RK4) is numerical; the linear solve is numerical. No
`unproven_claim` is ever asserted.
"""

from __future__ import annotations

import math

import omnibias.pinn.solver as pde
import omnibias.pinn.solver.torch as pt
import torch
from omnibias.pinn.solver.torch import (
    SpectralGrid1D,
    grid_solution,
    method_of_lines,
    reaction_diffusion_semidiscrete,
)


def steady_poisson() -> None:
    """Part 1: closed-form linear collocation on a manufactured Poisson problem."""
    torch.set_default_dtype(torch.float64)

    # Manufactured solution u* = sin(pi x) sin(pi y)  =>  Delta u* = -2 pi^2 u*.
    def u_star(c: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * c[:, 0]) * torch.sin(math.pi * c[:, 1])

    def source(c: torch.Tensor) -> torch.Tensor:
        xp = pde.array_namespace(c)
        return -2.0 * math.pi**2 * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    domain = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    system = pde.poisson(domain, source=source, boundary=0.0)
    print(f"[poisson] classification: {system.classify()}")

    solution = pt.solve_least_squares(
        system,
        hidden=100,
        weight_init_scale=3.0,
        seed=0,
        collocation=pde.CollocationSpec(n_interior=20, n_boundary=20),
    )

    grid = torch.linspace(0.02, 0.98, 40)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    u = solution.evaluate(pts, "u")
    ustar = u_star(pts)
    rel_l2 = torch.linalg.norm(u - ustar) / torch.linalg.norm(ustar)
    print(f"[poisson] residual_norm={solution.residual_norm:.3e}  relL2={rel_l2:.3e}")


def coupled_reaction_diffusion() -> None:
    """Part 2: a coupled two-species system via the spectral method-of-lines."""
    torch.set_default_dtype(torch.float64)

    grid = SpectralGrid1D(n=64, length=2.0 * math.pi)
    x = grid.points()

    exchange = 0.5  # mass-preserving linear exchange rate k

    def reaction(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return -exchange * (u - v), exchange * (u - v)

    semi = reaction_diffusion_semidiscrete(
        grid, diffusivities=(0.1, 0.05), reaction=reaction
    )

    u0 = torch.cos(x) + 1.5
    v0 = torch.sin(2.0 * x) + 1.5
    state0 = torch.stack([u0, v0], dim=0)

    times = [0.05 * i for i in range(21)]  # t in [0, 1.0]
    snapshots, ts = method_of_lines(semi, state0, times, integrator="rk4")
    solution = grid_solution(snapshots, ts, grid, ("u", "v"))
    print(f"[reaction-diffusion] classification: coupled 2-species; steps={len(ts)}")

    dx = grid.length / grid.n
    mass0 = float((solution.at("u", 0) + solution.at("v", 0)).sum() * dx)
    massT = float((solution.final("u") + solution.final("v")).sum() * dx)
    print(
        f"[reaction-diffusion] total mass  t=0: {mass0:.6f}   "
        f"t={float(ts[-1]):.2f}: {massT:.6f}   drift={abs(massT - mass0):.2e}"
    )


def main() -> None:
    steady_poisson()
    coupled_reaction_diffusion()
    print("done.")


if __name__ == "__main__":
    main()
