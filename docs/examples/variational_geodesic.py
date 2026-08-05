# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Geodesics as least action: great circles on the sphere.

Run:

    pip install "omnibias-variational[torch,geometry]"
    python docs/examples/variational_geodesic.py

A geodesic extremizes the arc-length / kinetic action ``S = int 1/2 g_ij
qdot^i qdot^j dt``. Its Euler-Lagrange equation is exactly the geodesic
equation, so this script:

- builds the metric Lagrangian of the round sphere ``ds^2 = dtheta^2 +
  sin^2(theta) dphi^2`` with ``metric_lagrangian`` and checks its generic
  Euler-Lagrange residual against ``omnibias.geometry``'s ``geodesic_rhs``;
- confirms the equator is a geodesic (residual ~ 0) whereas a curve that bulges
  off it is not;
- uses ``geodesic_action`` to show the equator is *shorter* than the bulging
  curve between the same two endpoints.

Trajectory derivatives are closed form; the metric derivatives inside
``geodesic_rhs`` are exact forward-mode autodiff of the analytic metric.
"""

from __future__ import annotations

import math

import torch
from _variational_fields import TrajectoryField, column
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.torch.ops.basic import stack_components, vector_derivative
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.variational.torch import ops as var

try:
    from omnibias.geometry._core.manifold import ManifoldSpec, MetricSpec
    from omnibias.geometry.torch.ops.connection import geodesic_rhs, metric
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This example needs omnibias-geometry: "
        'pip install "omnibias-variational[torch,geometry]"'
    ) from exc

DOF = ("theta", "phi")
DELTA = 1.2  # longitude swept


def sphere() -> ManifoldSpec:
    def g_point(x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros((), dtype=x.dtype)
        o = torch.ones((), dtype=x.dtype)
        s2 = torch.sin(x[0]) ** 2
        return torch.stack([torch.stack([o, z]), torch.stack([z, s2])])

    return ManifoldSpec("S2", 2, MetricSpec(g_point, 2, name="round_sphere"))


def equator_specs():
    return {
        "theta": (lambda t: (math.pi / 2) * torch.ones_like(t),
                  lambda t: torch.zeros_like(t), lambda t: torch.zeros_like(t)),
        "phi": (lambda t: t, lambda t: torch.ones_like(t), lambda t: torch.zeros_like(t)),
    }


def bulging_specs():
    # theta bulges off the equator but returns; same endpoints as the equator.
    a, w = 0.3, math.pi / DELTA
    return {
        "theta": (lambda t: (math.pi / 2) + a * torch.sin(w * t),
                  lambda t: a * w * torch.cos(w * t),
                  lambda t: -a * w**2 * torch.sin(w * t)),
        "phi": (lambda t: t, lambda t: torch.ones_like(t), lambda t: torch.zeros_like(t)),
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    manifold = sphere()
    lag = var.metric_lagrangian(manifold, dof=DOF)

    # ---- EL of the metric Lagrangian matches geodesic_rhs -----------------
    t = torch.linspace(0.15, 1.75, 5)
    state = TrajectoryField(bulging_specs())(column(t))
    el = var.euler_lagrange_residual(state, lag)
    q = stack_components(state, DOF)
    qdot = vector_derivative(state, DOF, axis="t", order=1)
    qddot = vector_derivative(state, DOF, axis="t", order=2)
    lowered = torch.einsum("bkm,bm->bk", metric(q, manifold), qddot - geodesic_rhs(q, qdot, manifold))
    print(f"max |EL - g(qddot - geodesic_rhs)| = {(el - lowered).abs().max().item():.2e}")
    assert (el - lowered).abs().max().item() < 1e-9

    # ---- the equator is a geodesic; the bulging curve is not --------------
    eq = TrajectoryField(equator_specs())(column(t))
    el_eq = var.euler_lagrange_residual(eq, lag).abs().max().item()
    el_bulge = el.abs().max().item()
    print(f"max |EL residual|:  equator = {el_eq:.2e}   bulging curve = {el_bulge:.2e}")
    assert el_eq < 1e-10
    assert el_bulge > 1e-2

    # ---- the geodesic is the shorter path (least length) ------------------
    rule = gauss_legendre([(0.0, DELTA)], 24)
    nodes = quadrature_nodes(rule, like=torch.zeros(1))[:, 0]
    len_eq = float(var.geodesic_action(TrajectoryField(equator_specs())(column(nodes)),
                                       manifold, rule=rule, dof=DOF))
    len_bulge = float(var.geodesic_action(TrajectoryField(bulging_specs())(column(nodes)),
                                          manifold, rule=rule, dof=DOF))
    print(f"arc length over [0, {DELTA}]:  equator = {len_eq:.6f}   bulging = {len_bulge:.6f}")
    assert abs(len_eq - DELTA) < 1e-10   # on the equator ds = dphi = dt
    assert len_bulge > len_eq
    print("\nOK: the equator is a geodesic and the shortest path between its endpoints.")


if __name__ == "__main__":
    main()
