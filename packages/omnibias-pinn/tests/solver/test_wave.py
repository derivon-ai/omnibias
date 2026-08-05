# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave equation (hyperbolic, linear, IVP, scalar) via space-time collocation."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402


def test_wave_space_time_collocation() -> None:
    """u_tt = c^2 u_xx, u(x,0)=sin(pi x), u_t(x,0)=0 -> cos(c pi t) sin(pi x)."""
    speed = 1.0
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.5)))

    def ic(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 0])

    sys = pde.wave(dom, speed=speed, initial=ic, initial_velocity=0.0, boundary=0.0)
    assert sys.classify().pde_type is pde.PDEType.HYPERBOLIC

    sol = pt.solve_evolution(
        sys, hidden=140, weight_init_scale=3.0, seed=0,
        collocation=pde.CollocationSpec(n_interior=24, n_boundary=24),
    )
    grid = np.linspace(0.02, 0.98, 30)
    ts = np.linspace(0.0, 0.5, 8)
    xx, tt = np.meshgrid(grid, ts, indexing="ij")
    pts = np.stack([xx.ravel(), tt.ravel()], axis=-1)
    u = sol.evaluate(pts, "u").detach().numpy()
    exact = np.cos(speed * math.pi * pts[:, 1]) * np.sin(math.pi * pts[:, 0])
    rel = np.linalg.norm(u - exact) / np.linalg.norm(exact)
    assert rel < 1e-2, f"wave collocation relL2 too large: {rel}"
