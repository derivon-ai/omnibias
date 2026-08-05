# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Viscous Burgers (parabolic, nonlinear, IVP, scalar) via spectral MOL.

The nonlinear ``u u_x`` term's time-jet is the Cauchy product computed by
``omnibias.torch.jet_mv.jet_multiply`` inside the jet-Taylor integrator.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver.torch as pt  # noqa: E402


def test_burgers_jet_taylor_matches_fine_reference() -> None:
    nu = 0.1
    grid = pt.SpectralGrid1D(32, 2.0 * math.pi)
    x = grid.points()
    u0 = torch.sin(x)
    semi = pt.burgers_semidiscrete(grid, nu)
    end = 0.3

    reference, _ = pt.method_of_lines(
        semi, u0, torch.linspace(0.0, end, 3001), integrator="rk4"
    )
    u_ref = reference[-1]

    coarse = torch.linspace(0.0, end, 61)
    snaps_jet, _ = pt.method_of_lines(semi, u0, coarse, integrator="jet_taylor", order=6)
    snaps_rk4, _ = pt.method_of_lines(semi, u0, coarse, integrator="rk4")

    rel_jet = torch.linalg.norm(snaps_jet[-1] - u_ref) / torch.linalg.norm(u_ref)
    rel_rk4 = torch.linalg.norm(snaps_rk4[-1] - u_ref) / torch.linalg.norm(u_ref)
    assert rel_jet < 1e-6, f"jet-Taylor Burgers relerr {rel_jet.item():.2e}"
    assert rel_rk4 < 1e-6, f"rk4 Burgers relerr {rel_rk4.item():.2e}"


def test_burgers_jet_step_is_high_order_in_time() -> None:
    """One jet-Taylor step of order p has local error ~ O(dt^{p+1})."""
    nu = 0.1
    grid = pt.SpectralGrid1D(32, 2.0 * math.pi)
    x = grid.points()
    u0 = torch.sin(x)
    semi = pt.burgers_semidiscrete(grid, nu)

    def local_error(dt: float, order: int) -> float:
        fine, _ = pt.method_of_lines(
            semi, u0, torch.linspace(0.0, dt, 401), integrator="rk4"
        )
        step = semi.jet_step(u0, dt, order)
        return float(torch.linalg.norm(step - fine[-1]))

    order = 4
    e1 = local_error(0.02, order)
    e2 = local_error(0.01, order)
    rate = math.log(e1 / e2) / math.log(2.0)
    assert rate > order + 0.5, f"jet-Taylor local order too low: {rate:.2f}"
