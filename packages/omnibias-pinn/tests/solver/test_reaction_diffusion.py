# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reaction-diffusion (parabolic, nonlinear, IVP, coupled system) via spectral MOL."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402


def test_reaction_diffusion_conserves_total_mass() -> None:
    """With reaction (-u v, +u v) and periodic diffusion, int(u+v) is conserved."""
    grid = pt.SpectralGrid1D(32, 2.0 * math.pi)
    x = grid.points()
    u0 = torch.stack(
        [1.0 + 0.2 * torch.sin(x), 0.5 + 0.1 * torch.cos(x)], dim=0
    )

    def reaction(u, v):
        return (-u * v, u * v)

    semi = pt.reaction_diffusion_semidiscrete(grid, (0.2, 0.1), reaction)
    times = torch.linspace(0.0, 0.5, 101)
    snaps, _ = pt.method_of_lines(semi, u0, times, integrator="rk4")

    assert torch.isfinite(snaps[-1]).all()
    mass0 = float((u0[0] + u0[1]).sum())
    mass_t = float((snaps[-1, 0] + snaps[-1, 1]).sum())
    assert abs(mass_t - mass0) < 1e-9, f"mass drift {abs(mass_t - mass0):.2e}"
    # the coupling / diffusion must actually evolve the state
    assert float(torch.linalg.norm(snaps[-1] - u0)) > 1e-3


def test_reaction_diffusion_is_a_coupled_nonlinear_system() -> None:
    dom = pde.Domain(("x", "t"), ((0.0, 2.0), (0.0, 0.5)), periodic=(True, False))
    sys = pde.reaction_diffusion(dom, reaction=lambda u, v: (u * v, -u * v))
    cls = sys.classify()
    assert cls.arity is pde.Arity.SYSTEM
    assert cls.linearity is pde.Linearity.NONLINEAR
    assert len(sys.residuals) == 2
