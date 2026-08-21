# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Implicit DEQ Newton, PyTorch (theory 08-08)."""

from __future__ import annotations

import math
import statistics

import pytest
import torch
from omnibias.core.implicit import DEQConfig, DEQNotContractive, DEQSolverUnknown
from omnibias.torch.implicit import deq_solve, deq_vjp


def test_section5_scalar_fixed_point() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = DEQConfig(solver="newton", tol=1e-12, max_iter=20)
    result = deq_solve(torch.tensor([[0.2]]), torch.tensor([0.5]), "tanh", config=cfg)
    u = float(result.u.reshape(-1)[0])
    z = 0.2 * u + 0.5
    assert abs(u - float(torch.tanh(torch.tensor(z)))) <= 1e-12
    assert result.residual <= 1e-10
    assert 0.542 < u < 0.544
    assert result.spectral_radius_bound < 1.0


def test_g1_ift_matches_central_fd() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = DEQConfig(solver="newton", tol=1e-14, max_iter=30)
    w = torch.tensor([[0.2]])
    x = torch.tensor([0.5])
    g = torch.tensor([1.0])
    vjp = float(deq_vjp(w, x, "tanh", g, config=cfg).reshape(-1)[0])
    eps = 1e-6
    up = float(deq_solve(w + eps, x, "tanh", config=cfg).u.reshape(-1)[0])
    um = float(deq_solve(w - eps, x, "tanh", config=cfg).u.reshape(-1)[0])
    fd = (up - um) / (2.0 * eps)
    rel = abs(vjp - fd) / max(abs(fd), 1e-12)
    assert rel <= 1e-8
    assert deq_solve(w, x, "tanh", config=cfg).residual <= 1e-10


def test_contraction_raises() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = DEQConfig(require_contraction=True, max_iter=2)
    with pytest.raises(DEQNotContractive, match="unroll"):
        deq_solve(torch.tensor([[3.0]]), torch.tensor([0.1]), "tanh", config=cfg)


def test_anderson_raises() -> None:
    torch.set_default_dtype(torch.float64)
    with pytest.raises(DEQSolverUnknown, match="newton"):
        deq_solve(
            torch.tensor([[0.2]]),
            torch.tensor([0.5]),
            "tanh",
            config=DEQConfig(solver="anderson"),
        )


def test_g1_width2_ift_matches_central_fd() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = DEQConfig(solver="newton", tol=1e-14, max_iter=40)
    w = torch.tensor([[0.15, -0.05], [0.08, 0.12]])
    x = torch.tensor([0.4, -0.2])
    g = torch.tensor([1.0, 0.0])
    vjp = deq_vjp(w, x, "tanh", g, config=cfg)
    eps = 1e-6
    fd = torch.empty_like(w)
    for i in range(2):
        for j in range(2):
            wp = w.clone()
            wm = w.clone()
            wp[i, j] = wp[i, j] + eps
            wm[i, j] = wm[i, j] - eps
            up = deq_solve(wp, x, "tanh", config=cfg).u[0]
            um = deq_solve(wm, x, "tanh", config=cfg).u[0]
            fd[i, j] = (up - um) / (2.0 * eps)
    rel = (vjp - fd).abs() / fd.abs().clamp_min(1e-12)
    assert float(rel.max()) <= 1e-8
    assert deq_solve(w, x, "tanh", config=cfg).residual <= 1e-10


def test_g2_implicit_residual_skill() -> None:
    torch.set_default_dtype(torch.float64)
    from omnibias.torch.implicit import deq_du_dW
    from omnibias.torch.optim import gauss_newton_direction

    xs = torch.linspace(0.0, 1.0, 21)
    target = torch.sin(math.pi * xs)
    x_inj = torch.stack([xs, torch.ones_like(xs)], dim=-1)
    cfg = DEQConfig(tol=1e-12, max_iter=30)
    zero_mse = float((target**2).mean())
    mses: list[float] = []
    for seed in range(5):
        gen = torch.Generator().manual_seed(seed)
        w = 0.15 * torch.randn(2, 2, generator=gen)
        sol = deq_solve(w, x_inj, "tanh", config=cfg)
        best = float(((sol.u[:, 0] - target) ** 2).mean())
        for _ in range(15):
            res = sol.u[:, 0] - target
            jac = deq_du_dW(w, x_inj, "tanh", config=cfg, u=sol.u)
            j = jac[:, 0, :, :].reshape(xs.numel(), 4)
            trial = w + gauss_newton_direction(j, res, 1e-3).reshape(2, 2)
            trial_sol = deq_solve(trial, x_inj, "tanh", config=cfg)
            trial_mse = float(((trial_sol.u[:, 0] - target) ** 2).mean())
            if trial_mse <= best:
                w, sol, best = trial, trial_sol, trial_mse
        mses.append(best)
    assert all(m < zero_mse for m in mses)
    assert statistics.mean(mses) <= 0.20
    assert max(mses) <= 0.25


def test_iterate_reaches_same_point() -> None:
    torch.set_default_dtype(torch.float64)
    newton = deq_solve(
        torch.tensor([[0.2]]),
        torch.tensor([0.5]),
        "tanh",
        config=DEQConfig(solver="newton", tol=1e-12),
    )
    it = deq_solve(
        torch.tensor([[0.2]]),
        torch.tensor([0.5]),
        "tanh",
        config=DEQConfig(solver="iterate", tol=1e-12, max_iter=80),
    )
    assert abs(float(newton.u.reshape(-1)[0]) - float(it.u.reshape(-1)[0])) <= 1e-8
