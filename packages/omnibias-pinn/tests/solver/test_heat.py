# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Heat equation (parabolic, linear, IVP, scalar): collocation + MOL."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402


def test_heat_space_time_collocation() -> None:
    """u_t = D u_xx with u(x,0)=sin(pi x), u=0 at x=0,1 -> exp(-D pi^2 t) sin(pi x)."""
    diff = 0.1
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.3)))

    def ic(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 0])

    sys = pde.heat(dom, diffusivity=diff, initial=ic, boundary=0.0)
    sol = pt.solve_evolution(
        sys, hidden=120, weight_init_scale=3.0, seed=0,
        collocation=pde.CollocationSpec(n_interior=22, n_boundary=22),
    )
    assert sol.method == "least_squares"

    grid = np.linspace(0.02, 0.98, 30)
    ts = np.linspace(0.0, 0.3, 8)
    xx, tt = np.meshgrid(grid, ts, indexing="ij")
    pts = np.stack([xx.ravel(), tt.ravel()], axis=-1)
    u = sol.evaluate(pts, "u").detach().numpy()
    exact = np.exp(-diff * math.pi ** 2 * pts[:, 1]) * np.sin(math.pi * pts[:, 0])
    rel = np.linalg.norm(u - exact) / np.linalg.norm(exact)
    assert rel < 5e-3, f"heat collocation relL2 too large: {rel}"


def test_heat_method_of_lines_matches_exact_mode() -> None:
    diff = 0.1
    grid = pt.SpectralGrid1D(32, 2.0 * math.pi)
    x = grid.points()
    k0 = 2
    u0 = torch.sin(k0 * x)
    semi = pt.heat_semidiscrete(grid, diff)
    times = torch.linspace(0.0, 0.5, 21)
    exact = math.exp(-diff * k0 ** 2 * 0.5) * torch.sin(k0 * x)

    for integrator in ("rk4", "jet_taylor", "crank_nicolson"):
        snaps, _ = pt.method_of_lines(semi, u0, times, integrator=integrator, order=8)
        rel = torch.linalg.norm(snaps[-1] - exact) / torch.linalg.norm(exact)
        tol = 1e-9 if integrator in ("rk4", "jet_taylor") else 1e-4
        assert rel < tol, f"{integrator}: relerr {rel.item():.2e}"
