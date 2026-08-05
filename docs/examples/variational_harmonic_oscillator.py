# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Least action for the harmonic oscillator.

Run:

    pip install omnibias-variational[torch]
    python docs/examples/variational_harmonic_oscillator.py

Hamilton's principle: the physical path makes the action ``S[q] = int L dt``
stationary. For ``L = 1/2 q_dot^2 - 1/2 w^2 q^2`` the extremal is ``q*(t) =
cos(w t)``. This script checks the three faces of the principle on omnibias:

1. the Euler-Lagrange residual vanishes on ``q*`` (``q_dot``/``q_ddot`` closed
   form from the sigma-tower, Lagrangian partials by autodiff);
2. the energy ``E = 1/2 q_dot^2 + 1/2 w^2 q^2`` is conserved along ``q*``;
3. ``S`` is genuinely minimized at ``q*``: perturbing by ``eps * sin(pi t / T)``
   (which pins both endpoints) raises the action, with ``dS/deps = 0`` at eps=0.
"""

from __future__ import annotations

import math

import torch
from _variational_fields import TrajectoryField
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as var

W = 1.0        # angular frequency
TF = 1.0       # horizon (< pi/W, so the action is a *minimum*, not a saddle)

LAG = Lagrangian(
    lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * W**2 * (q**2).sum(-1),
    dof=("q",),
)


def perturbed_specs(eps: float):
    """q(t) = cos(w t) + eps sin(pi t / T), plus its first two derivatives."""
    a = math.pi / TF
    return {
        "q": (
            lambda t: torch.cos(W * t) + eps * torch.sin(a * t),
            lambda t: -W * torch.sin(W * t) + eps * a * torch.cos(a * t),
            lambda t: -(W**2) * torch.cos(W * t) - eps * a**2 * torch.sin(a * t),
        )
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)

    # ---- 1. Euler-Lagrange residual vanishes on the true path -------------
    t_dense = torch.linspace(0.0, TF, 64).reshape(-1, 1)
    state = TrajectoryField(perturbed_specs(0.0))(t_dense)
    el = var.euler_lagrange_residual(state, LAG)
    print(f"max |Euler-Lagrange residual| on q* = {el.abs().max().item():.2e}")
    assert el.abs().max().item() < 1e-10

    # ---- 2. Energy is conserved along the true path -----------------------
    energy = var.energy(state, LAG)
    print(f"energy: mean={energy.mean().item():.6f}  std={energy.std().item():.2e}")
    assert energy.std().item() < 1e-10
    assert abs(energy.mean().item() - 0.5 * W**2) < 1e-10

    # ---- 3. The action is minimized at eps = 0 ----------------------------
    rule = gauss_legendre([(0.0, TF)], 48)
    nodes = quadrature_nodes(rule, like=torch.zeros(1))

    def action_of(eps: float) -> float:
        s = TrajectoryField(perturbed_specs(eps))(nodes)
        return float(var.action(s, LAG, rule=rule))

    s0 = action_of(0.0)
    print(f"\naction S(eps):   (extremal S(0) = {s0:.6f})")
    print("   eps      S(eps)      S(eps) - S(0)")
    worst = 0.0
    for eps in (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4):
        s = action_of(eps)
        print(f"  {eps:+.2f}   {s:+.6f}     {s - s0:+.3e}")
        if eps != 0.0:
            assert s > s0  # every endpoint-preserving perturbation costs action
            worst = max(worst, s - s0)

    h = 1e-4
    dS = (action_of(h) - action_of(-h)) / (2 * h)
    print(f"\ndS/deps at eps=0 = {dS:.2e}  (stationary)   max rise = {worst:.3e}")
    assert abs(dS) < 1e-8
    print("\nOK: q* extremizes and minimizes the action; energy is conserved.")


if __name__ == "__main__":
    main()
