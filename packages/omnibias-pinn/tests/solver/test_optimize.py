# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke test for the general residual-minimisation driver (Adam + L-BFGS)."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402


def test_solve_optimize_reduces_poisson_residual() -> None:
    def source(c):
        xp = pde.array_namespace(c)
        return -2.0 * math.pi ** 2 * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    sys = pde.poisson(dom, source=source, boundary=0.0)

    sol = pt.solve_optimize(
        sys, hidden=32, seed=0,
        collocation=pde.CollocationSpec(n_interior=14, n_boundary=14),
        adam_iters=80, iters=20, condition_weight=20.0,
    )
    assert sol.method == "optimize:lbfgs"
    assert math.isfinite(sol.residual_norm)

    grid = np.linspace(0.05, 0.95, 20)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    u = sol.evaluate(pts, "u").detach().numpy()
    ue = np.sin(math.pi * pts[:, 0]) * np.sin(math.pi * pts[:, 1])
    rel = np.linalg.norm(u - ue) / np.linalg.norm(ue)
    assert rel < 0.3, f"optimize Poisson relL2 {rel}"
