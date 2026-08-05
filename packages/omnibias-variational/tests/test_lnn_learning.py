# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learning a Lagrangian from data (a tiny Lagrangian Neural Network).

``lagrangian_dynamics_loss`` is differentiable w.r.t. the parameters closed over
by ``lagrangian.fn`` (here two scalar log-parameters through softplus). Fitting a
structured ``L = 1/2 a qdot^2 - 1/2 k q^2`` to harmonic-oscillator accelerations
drives the loss to ~0 and recovers the identifiable ratio ``k / a = w^2`` (the
overall scale of ``L`` is a gauge freedom that does not affect the dynamics).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as tv

DT = torch.float64
W = 1.3


def test_lnn_recovers_frequency_ratio() -> None:
    torch.manual_seed(0)
    q = torch.randn(128, 1, dtype=DT)
    qd = torch.randn(128, 1, dtype=DT)
    t = torch.zeros(128, 1, dtype=DT)
    qddot_target = -(W**2) * q  # unit-mass harmonic oscillator

    log_a = torch.zeros((), dtype=DT, requires_grad=True)
    log_k = torch.zeros((), dtype=DT, requires_grad=True)

    def make_lagrangian() -> Lagrangian:
        a = F.softplus(log_a)
        k = F.softplus(log_k)
        return Lagrangian(
            lambda q, qd, t: 0.5 * a * (qd**2).sum(-1) - 0.5 * k * (q**2).sum(-1),
            dof=("q",),
        )

    opt = torch.optim.Adam([log_a, log_k], lr=0.05)
    initial = tv.lagrangian_dynamics_loss(
        make_lagrangian(), q, qd, qddot_target, t,
    ).detach().item()
    loss = torch.tensor(float("nan"))
    for _ in range(400):
        opt.zero_grad()
        loss = tv.lagrangian_dynamics_loss(make_lagrangian(), q, qd, qddot_target, t)
        loss.backward()
        opt.step()

    final = loss.detach().item()
    with torch.no_grad():
        a = F.softplus(log_a).item()
        k = F.softplus(log_k).item()
    assert final < initial          # training reduced the loss
    assert final < 1e-6             # and essentially fits the accelerations
    assert abs(k / a - W**2) < 0.02  # the identifiable ratio is recovered


def test_lnn_loss_reduction_modes() -> None:
    lag = Lagrangian(
        lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * (q**2).sum(-1), dof=("q",),
    )
    q = torch.tensor([[0.5], [1.0]], dtype=DT)
    qd = torch.zeros(2, 1, dtype=DT)
    t = torch.zeros(2, 1, dtype=DT)
    target = torch.zeros(2, 1, dtype=DT)  # true qddot = -q; target 0 => residual q
    mean = tv.lagrangian_dynamics_loss(lag, q, qd, target, t, reduction="mean")
    total = tv.lagrangian_dynamics_loss(lag, q, qd, target, t, reduction="sum")
    # acceleration = -q, so squared error is q^2: [0.25, 1.0].
    assert abs(float(mean) - 0.625) < 1e-12
    assert abs(float(total) - 1.25) < 1e-12
