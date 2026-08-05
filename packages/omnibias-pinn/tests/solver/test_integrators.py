# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Time-integrator order-of-accuracy and baseline behaviour (torch)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver.torch as pt  # noqa: E402


def _heat_setup():
    grid = pt.SpectralGrid1D(32, 2.0 * math.pi)
    x = grid.points()
    diff = 0.1
    k0 = 2
    u0 = torch.sin(k0 * x)
    semi = pt.heat_semidiscrete(grid, diff)

    def exact_step(dt: float) -> torch.Tensor:
        return math.exp(-diff * k0 ** 2 * dt) * torch.sin(k0 * x)

    return semi, u0, exact_step


def test_jet_taylor_local_order_matches_p_plus_one() -> None:
    semi, u0, exact_step = _heat_setup()
    for order, expected in ((2, 3.0), (4, 5.0)):
        e1 = float(torch.linalg.norm(semi.jet_step(u0, 0.2, order) - exact_step(0.2)))
        e2 = float(torch.linalg.norm(semi.jet_step(u0, 0.1, order) - exact_step(0.1)))
        rate = math.log(e1 / e2) / math.log(2.0)
        assert abs(rate - expected) < 0.4, f"order {order}: rate {rate:.2f}"


def test_higher_jet_order_is_more_accurate() -> None:
    semi, u0, exact_step = _heat_setup()
    dt = 0.2
    err = [
        float(torch.linalg.norm(semi.jet_step(u0, dt, order) - exact_step(dt)))
        for order in (2, 4, 6)
    ]
    assert err[0] > err[1] > err[2]


def test_rk4_local_order_is_five() -> None:
    semi, u0, exact_step = _heat_setup()
    e1 = float(torch.linalg.norm(pt.rk4_step(semi.rhs, u0, 0.2) - exact_step(0.2)))
    e2 = float(torch.linalg.norm(pt.rk4_step(semi.rhs, u0, 0.1) - exact_step(0.1)))
    rate = math.log(e1 / e2) / math.log(2.0)
    assert abs(rate - 5.0) < 0.5, f"rk4 local rate {rate:.2f}"


def test_implicit_baselines_are_stable_for_large_steps() -> None:
    """Implicit Euler / Crank-Nicolson stay bounded at a step where explicit blows up."""
    semi, u0, exact_step = _heat_setup()
    big_dt = 5.0  # far beyond the explicit stability limit
    for scheme in ("implicit_euler", "crank_nicolson"):
        u = pt.implicit_linear_step(semi.grid, semi.symbol, u0, big_dt, scheme=scheme)
        assert torch.isfinite(u).all()
        assert torch.linalg.norm(u) <= torch.linalg.norm(u0) + 1e-9
