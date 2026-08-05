# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Poisson (elliptic, linear, BVP, scalar) via exact-operator linear collocation."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402


def _manufactured():
    def usol(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    def source(c):
        return -2.0 * math.pi ** 2 * usol(c)

    return usol, source


def test_poisson_least_squares_matches_manufactured() -> None:
    torch.set_default_dtype(torch.float64)
    _, source = _manufactured()
    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    sys = pde.poisson(dom, source=source, boundary=0.0)

    sol = pt.solve_least_squares(
        sys,
        hidden=100,
        weight_init_scale=3.0,
        seed=0,
        collocation=pde.CollocationSpec(n_interior=20, n_boundary=20),
    )
    assert sol.method == "least_squares"

    grid = np.linspace(0.02, 0.98, 40)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    u = sol.evaluate(pts, "u").detach().numpy()
    ustar = np.sin(math.pi * pts[:, 0]) * np.sin(math.pi * pts[:, 1])
    rel_l2 = np.linalg.norm(u - ustar) / np.linalg.norm(ustar)
    assert rel_l2 < 4e-2, f"relative L2 error too large: {rel_l2}"


def test_solve_steady_auto_dispatches_least_squares_for_linear() -> None:
    torch.set_default_dtype(torch.float64)
    _, source = _manufactured()
    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    sys = pde.poisson(dom, source=source, boundary=0.0)
    sol = pde.torch.solve_steady(
        sys, hidden=64, weight_init_scale=2.5,
        collocation=pde.CollocationSpec(n_interior=16, n_boundary=16),
    )
    assert sol.method == "least_squares"
    assert sol.residual_norm < 0.2


def test_least_squares_rejects_nonlinear_system() -> None:
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.5)), periodic=(True, False))
    burg = pde.burgers(dom, viscosity=0.1, initial=0.0)
    with pytest.raises(ValueError):
        pt.solve_least_squares(burg)
