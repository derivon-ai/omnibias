# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Advection-diffusion (linear, IVP, coupled system): MOL + space-time collocation."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402


def test_advection_diffusion_transports_and_decays_a_mode() -> None:
    """u_t + a u_x = D u_xx -> exp(-D k^2 t) sin(k (x - a t)) (single-field MOL)."""
    a = 1.0
    diff = 0.05
    k0 = 2
    grid = pt.SpectralGrid1D(32, 2.0 * math.pi)
    x = grid.points()
    u0 = torch.sin(k0 * x)
    semi = pt.advection_diffusion_semidiscrete(grid, a, diff)
    end = 0.5
    snaps, _ = pt.method_of_lines(
        semi, u0, torch.linspace(0.0, end, 51), integrator="jet_taylor", order=8
    )
    exact = math.exp(-diff * k0 ** 2 * end) * torch.sin(k0 * (x - a * end))
    rel = torch.linalg.norm(snaps[-1] - exact) / torch.linalg.norm(exact)
    assert rel < 1e-9, f"advection-diffusion relerr {rel.item():.2e}"


def test_coupled_advection_diffusion_space_time_collocation() -> None:
    """The coupled 2-field system solves via space-time collocation (linear->lstsq)."""
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.3)))

    def ic_u(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 0])

    def ic_v(c):
        xp = pde.array_namespace(c)
        return xp.sin(2.0 * math.pi * c[:, 0])

    sys = pde.advection_diffusion(
        dom, velocity=0.5, diffusivities=(0.1, 0.1), coupling=0.3,
        initial=(ic_u, ic_v), boundary=(0.0, 0.0),
    )
    assert sys.classify().arity is pde.Arity.SYSTEM
    sol = pt.solve_evolution(
        sys, hidden=120, weight_init_scale=3.0, seed=0,
        collocation=pde.CollocationSpec(n_interior=22, n_boundary=22),
    )
    assert sol.method == "least_squares"
    assert sol.residual_norm < 0.05

    # the fitted field honours the initial condition
    xs = np.linspace(0.02, 0.98, 40)
    pts0 = np.stack([xs, np.zeros_like(xs)], axis=-1)
    u_ic = sol.evaluate(pts0, "u").detach().numpy()
    v_ic = sol.evaluate(pts0, "v").detach().numpy()
    rel_u = np.linalg.norm(u_ic - np.sin(math.pi * xs)) / np.linalg.norm(np.sin(math.pi * xs))
    rel_v = np.linalg.norm(v_ic - np.sin(2 * math.pi * xs)) / np.linalg.norm(
        np.sin(2 * math.pi * xs)
    )
    assert rel_u < 0.1 and rel_v < 0.15, f"IC fit u={rel_u:.3f} v={rel_v:.3f}"
