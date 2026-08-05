# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Brachistochrone: the cycloid is the least-time path.

Run:

    pip install omnibias-variational[torch]
    python docs/examples/variational_brachistochrone.py

A bead sliding without friction from rest minimises the descent time
``T[y] proportional to int sqrt((1 + y'^2) / y) dx``. The minimiser is a cycloid
``x = R(tau - sin tau)``, ``y = R(1 - cos tau)``. This script shows both faces of
the variational statement on omnibias:

- *indirect*: the cycloid zeroes the Euler-Lagrange residual of the descent-time
  Lagrangian (it is an extremal), while a straight chord between the same
  endpoints does not;
- *direct*: evaluating the descent-time functional with ``action`` gives a
  smaller value for the cycloid than for the chord.

``y'`` and ``y''`` are closed form (the omnibias sigma-tower); the Lagrangian
partials are autodiff; the functional is Gauss-Legendre quadrature.
"""

from __future__ import annotations

import math

import torch
from _variational_fields import TrajectoryField, column
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as var

R = 1.0
TAU0, TAU1 = 0.5, 1.7 * math.pi  # a cycloid arc, away from the singular cusp

# Descent-time Lagrangian L(y, y') = sqrt((1 + y'^2) / y)  (release from y = 0).
LAG = Lagrangian(lambda q, qd, t: torch.sqrt((1.0 + qd[..., 0] ** 2) / q[..., 0]), dof=("y",))


def cycloid(tau: torch.Tensor):
    """(x, y, y'=dy/dx, y''=d2y/dx2) of the cycloid at parameter tau."""
    x = R * (tau - torch.sin(tau))
    y = R * (1.0 - torch.cos(tau))
    yp = torch.sin(tau) / (1.0 - torch.cos(tau))
    ypp = -1.0 / (R * (1.0 - torch.cos(tau)) ** 2)
    return x, y, yp, ypp


def tau_of_x(x_target: torch.Tensor) -> torch.Tensor:
    """Invert x(tau) = R(tau - sin tau) by bisection (x is monotone in tau)."""
    lo = torch.full_like(x_target, TAU0)
    hi = torch.full_like(x_target, TAU1)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        over = R * (mid - torch.sin(mid)) > x_target
        hi = torch.where(over, mid, hi)
        lo = torch.where(over, lo, mid)
    return 0.5 * (lo + hi)


def const_specs(y, yp):
    """Precomputed per-node y(x), y'(x); y'' = 0 for the straight chord."""
    return {"y": (lambda t: y, lambda t: yp, lambda t: torch.zeros_like(y))}


def main() -> None:
    torch.set_default_dtype(torch.float64)

    # ---- indirect: the cycloid is an extremal; the chord is not -----------
    tau = torch.linspace(TAU0, TAU1, 9)
    x, y, yp, ypp = cycloid(tau)
    cyc = TrajectoryField({"y": (lambda t: y, lambda t: yp, lambda t: ypp)})(column(x))
    el_cyc = var.euler_lagrange_residual(cyc, LAG).abs().max().item()

    (x0, y0), (x1, y1) = (x[0], y[0]), (x[-1], y[-1])
    slope = (y1 - y0) / (x1 - x0)
    y_line = y0 + slope * (x - x0)
    line = TrajectoryField(const_specs(y_line, slope * torch.ones_like(x)))(column(x))
    el_line = var.euler_lagrange_residual(line, LAG).abs().max().item()

    print(f"max |EL residual|:  cycloid = {el_cyc:.2e}   straight chord = {el_line:.2e}")
    assert el_cyc < 1e-9
    assert el_line > 1e-2

    # ---- direct: the cycloid has the smaller descent-time functional ------
    rule = gauss_legendre([(float(x0), float(x1))], 64)
    xq = quadrature_nodes(rule, like=torch.zeros(1))[:, 0]

    tq = tau_of_x(xq)
    _, yq, ypq, _ = cycloid(tq)
    cyc_q = TrajectoryField(const_specs(yq, ypq))(column(xq))
    line_yq = y0 + slope * (xq - x0)
    line_q = TrajectoryField(const_specs(line_yq, slope * torch.ones_like(xq)))(column(xq))

    j_cyc = float(var.action(cyc_q, LAG, rule=rule))
    j_line = float(var.action(line_q, LAG, rule=rule))
    j_cyc_exact = math.sqrt(2.0 * R) * (TAU1 - TAU0)  # closed form for the cycloid
    print(f"descent-time functional:  cycloid = {j_cyc:.6f}   chord = {j_line:.6f}")
    print(f"cycloid vs closed form {j_cyc_exact:.6f}:  |diff| = {abs(j_cyc - j_cyc_exact):.2e}")
    assert j_cyc < j_line
    assert abs(j_cyc - j_cyc_exact) < 1e-4
    print("\nOK: the cycloid extremizes and minimizes the descent time.")


if __name__ == "__main__":
    main()
