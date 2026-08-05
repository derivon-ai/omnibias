# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Higher-order least action: the functional (Euler-Poisson) derivative.

Run:

    pip install omnibias-variational[torch]
    python docs/examples/variational_higher_order.py

Some actions depend on more than the velocity. For a Lagrangian of order ``n``,
``L(q, q', ..., q^(n), t)``, stationarity ``delta S = 0`` is the arbitrary-order
**Euler-Poisson** equation

    delta S / delta q = sum_{k=0}^{n} (-1)^k d^k/dt^k ( dL / dq^(k) ) = 0,

which ``functional_derivative`` evaluates with the outer ``d^k/dt^k`` supplied
**closed form** by the sigma-tower (up to order ``2n``) and the ``L``-partials by
autodiff. This script checks two second-order (``n = 2``) problems:

1. the Pais-Uhlenbeck oscillator ``L = 1/2 q''^2 - 1/2 (w1^2 + w2^2) q'^2 +
   1/2 w1^2 w2^2 q^2``, whose Euler-Poisson equation is the fourth-order
   ``q'''' + (w1^2 + w2^2) q'' + w1^2 w2^2 q = 0`` -- zero on the normal mode
   ``cos(w1 t)``;
2. the static Euler-Bernoulli beam ``L = 1/2 EI y''^2 - rho y``, whose
   Euler-Poisson equation is ``EI y'''' - rho``.
"""

from __future__ import annotations

import torch
from _variational_fields import TrajectoryField, column
from omnibias.fields.torch.ops.basic import stack_components, vector_derivative
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as var

W1, W2 = 1.0, 1.7      # Pais-Uhlenbeck normal-mode frequencies
EI, RHO, KBEAM = 2.3, 0.7, 1.1  # beam stiffness, load, mode number


def cos_specs(omega: float):
    """cos(omega t) and its first four derivatives (order 2 needs q up to 4)."""
    return {
        "q": (
            lambda t: torch.cos(omega * t),
            lambda t: -omega * torch.sin(omega * t),
            lambda t: -(omega**2) * torch.cos(omega * t),
            lambda t: omega**3 * torch.sin(omega * t),
            lambda t: omega**4 * torch.cos(omega * t),
        )
    }


def sin_specs(k: float):
    """sin(k x) and its first four derivatives, for the beam deflection y(x)."""
    return {
        "y": (
            lambda t: torch.sin(k * t),
            lambda t: k * torch.cos(k * t),
            lambda t: -(k**2) * torch.sin(k * t),
            lambda t: -(k**3) * torch.cos(k * t),
            lambda t: k**4 * torch.sin(k * t),
        )
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    t = column(torch.linspace(0.1, 1.9, 48))

    pu = Lagrangian(
        lambda q, qd, qdd, t: (
            0.5 * (qdd**2).sum(-1)
            - 0.5 * (W1**2 + W2**2) * (qd**2).sum(-1)
            + 0.5 * W1**2 * W2**2 * (q**2).sum(-1)
        ),
        dof=("q",),
        order=2,
    )

    # ---- 1a. Euler-Poisson vanishes on the normal mode cos(w1 t) ----------
    state = TrajectoryField(cos_specs(W1))(t)
    fd = var.functional_derivative(state, pu)
    print(f"Pais-Uhlenbeck  delta S/delta q on cos(w1 t): max |.| = {fd.abs().max():.2e}")
    assert fd.abs().max() < 1e-9
    # euler_lagrange_residual dispatches to the same (negated) operator:
    el = var.euler_lagrange_residual(state, pu)
    assert (fd + el).abs().max() < 1e-12

    # ---- 1b. Off a mode it IS the fourth-order operator -------------------
    off = TrajectoryField(cos_specs(0.53))(t)  # 0.53 is neither normal mode
    fd_off = var.functional_derivative(off, pu)[:, 0]
    q = stack_components(off, ("q",))[:, 0]
    q2 = vector_derivative(off, ("q",), axis="t", order=2)[:, 0]
    q4 = vector_derivative(off, ("q",), axis="t", order=4)[:, 0]
    manual = q4 + (W1**2 + W2**2) * q2 + W1**2 * W2**2 * q
    print(f"                off-mode == q'''' + (w1^2+w2^2)q'' + w1^2 w2^2 q: "
          f"max diff = {(fd_off - manual).abs().max():.2e}")
    assert (fd_off - manual).abs().max() < 1e-10

    # ---- 2. Static Euler-Bernoulli beam: EI y'''' - rho -------------------
    beam = Lagrangian(
        lambda y, yp, ypp, t: 0.5 * EI * (ypp**2).sum(-1) - RHO * y.sum(-1),
        dof=("y",),
        order=2,
    )
    ys = TrajectoryField(sin_specs(KBEAM))(t)
    fd_beam = var.functional_derivative(ys, beam)[:, 0]
    y4 = vector_derivative(ys, ("y",), axis="t", order=4)[:, 0]
    manual_beam = EI * y4 - RHO
    print(f"Euler-Bernoulli delta S/delta y == EI y'''' - rho: "
          f"max diff = {(fd_beam - manual_beam).abs().max():.2e}")
    assert (fd_beam - manual_beam).abs().max() < 1e-10

    print("\nOK: the Euler-Poisson functional derivative reproduces the "
          "fourth-order equations of motion (outer d/dt closed form).")


if __name__ == "__main__":
    main()
