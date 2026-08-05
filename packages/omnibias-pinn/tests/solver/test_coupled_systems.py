# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Coupled-system headline: a coupled elliptic system via exact linear collocation.

Multiple fields, vector residual assembled from the closed-form operator surface,
solved as one least-squares problem.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402


def test_coupled_elliptic_manufactured_solution() -> None:
    r"""Solve  Delta u - k(u - v) = f_u,  Delta v - k(v - u) = f_v  with a
    manufactured (u*, v*) and Dirichlet data, as a single coupled least-squares.
    """
    k = 0.5

    def u_star(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    def v_star(c):
        x, y = c[:, 0], c[:, 1]
        return 16.0 * x * (1.0 - x) * y * (1.0 - y)

    def lap_u_star(c):
        return -2.0 * math.pi ** 2 * u_star(c)

    def lap_v_star(c):
        x, y = c[:, 0], c[:, 1]
        return 16.0 * (-2.0 * y * (1.0 - y) - 2.0 * x * (1.0 - x))

    def f_u(c):
        return lap_u_star(c) - k * (u_star(c) - v_star(c))

    def f_v(c):
        return lap_v_star(c) - k * (v_star(c) - u_star(c))

    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    res_u = lambda s: (  # noqa: E731
        s.ops.laplacian(s, "u") - k * (s.ops.value(s, "u") - s.ops.value(s, "v")) - f_u(s.coords)
    )
    res_v = lambda s: (  # noqa: E731
        s.ops.laplacian(s, "v") - k * (s.ops.value(s, "v") - s.ops.value(s, "u")) - f_v(s.coords)
    )
    sys = pde.make_system(
        domain=dom,
        fields=["u", "v"],
        residuals=[res_u, res_v],
        boundary=[
            pde.BoundaryCondition("u", "dirichlet", u_star),
            pde.BoundaryCondition("v", "dirichlet", v_star),
        ],
        linearity=pde.Linearity.LINEAR,
        name="coupled_elliptic",
    )
    assert sys.classify().arity is pde.Arity.SYSTEM

    sol = pt.solve_least_squares(
        sys, hidden=110, weight_init_scale=3.0, seed=0,
        collocation=pde.CollocationSpec(n_interior=22, n_boundary=22),
    )

    grid = np.linspace(0.03, 0.97, 28)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    u = sol.evaluate(pts, "u").detach().numpy()
    v = sol.evaluate(pts, "v").detach().numpy()
    ue = np.sin(math.pi * pts[:, 0]) * np.sin(math.pi * pts[:, 1])
    ve = 16.0 * pts[:, 0] * (1 - pts[:, 0]) * pts[:, 1] * (1 - pts[:, 1])
    rel_u = np.linalg.norm(u - ue) / np.linalg.norm(ue)
    rel_v = np.linalg.norm(v - ve) / np.linalg.norm(ve)
    assert rel_u < 4e-2, f"coupled u relL2 {rel_u}"
    assert rel_v < 4e-2, f"coupled v relL2 {rel_v}"
