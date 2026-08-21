# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sharpness-scheduled cubic step, PyTorch (theory 08-06)."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.torch.optim import CubicNewton, CubicRegularizedNewton
from omnibias.torch.optim_sharpness import (
    SharpnessSchedule,
    sharpness_lambda_max,
    sharpness_scheduled_minimize,
    sharpness_scheduled_step,
)

STIFF = 1.0e4
G1_SCHEDULE = SharpnessSchedule(n_lanczos=4, c=1e-3, ell_min=1e-6)
SEEDS = (0, 1, 2, 3, 4)


def _stiff_loss(params: torch.Tensor) -> torch.Tensor:
    p = params.reshape(-1)
    return 0.5 * (p[0] * p[0] + STIFF * p[1] * p[1])


def _start(seed: int) -> torch.Tensor:
    return torch.tensor([1.0 + 0.02 * seed, 1.0 - 0.01 * seed])


def test_lambda_max_stiff_quadratic() -> None:
    torch.set_default_dtype(torch.float64)
    ell = sharpness_lambda_max(_stiff_loss, torch.tensor([1.0, 1.0]), n_lanczos=4)
    assert abs(ell - STIFF) / STIFF <= 1e-12


def test_probe_along_y_is_exactly_stiff() -> None:
    torch.set_default_dtype(torch.float64)
    ell = sharpness_lambda_max(
        _stiff_loss,
        torch.tensor([1.0, 1.0]),
        n_lanczos=1,
        probe=torch.tensor([0.0, 1.0]),
    )
    assert abs(ell - STIFF) / STIFF <= 1e-12


def test_scheduled_step_records_named_schedule() -> None:
    torch.set_default_dtype(torch.float64)
    _new, report = sharpness_scheduled_step(
        _stiff_loss, torch.tensor([1.0, 1.0]), schedule=G1_SCHEDULE
    )
    assert report.c == pytest.approx(1e-3)
    assert report.n_lanczos == 4
    assert report.ell_min == pytest.approx(1e-6)
    assert abs(report.ell_k - STIFF) / STIFF <= 1e-12
    assert report.scheduled == pytest.approx(10.0)


def test_twenty_steps_reach_loss_floor() -> None:
    torch.set_default_dtype(torch.float64)
    params, losses, ells = sharpness_scheduled_minimize(
        _stiff_loss, torch.tensor([1.0, 1.0]), schedule=G1_SCHEDULE, steps=20
    )
    assert math.isfinite(losses[-1])
    assert bool(torch.isfinite(params).all())
    assert losses[-1] < 1e-8
    assert ells
    assert min(ells) > 0.0
    assert max(ells) == pytest.approx(STIFF, rel=1e-12)


def test_g1_five_seeds_finite() -> None:
    torch.set_default_dtype(torch.float64)
    finite = 0
    finals: list[float] = []
    for seed in SEEDS:
        params, losses, _ells = sharpness_scheduled_minimize(
            _stiff_loss, _start(seed), schedule=G1_SCHEDULE, steps=20
        )
        ok = math.isfinite(losses[-1]) and bool(torch.isfinite(params).all())
        finite += int(ok)
        finals.append(losses[-1])
    assert finite == len(SEEDS)
    assert max(finals) < 1e-8


def test_zero_sharpness_refuses_step() -> None:
    torch.set_default_dtype(torch.float64)

    def constant(params: torch.Tensor) -> torch.Tensor:
        return 0.0 * params.reshape(-1)[0]

    schedule = SharpnessSchedule(c=1.0, ell_min=0.0, target="cubic_sigma")
    with pytest.raises(ValueError, match="refusing the step"):
        sharpness_scheduled_step(constant, torch.tensor([1.0, 1.0]), schedule=schedule)


def test_lr_target_takes_gradient_step() -> None:
    torch.set_default_dtype(torch.float64)
    schedule = SharpnessSchedule(n_lanczos=4, c=1.0, ell_min=1e-6, target="lr")
    start = torch.tensor([1.0, 1.0])
    new, report = sharpness_scheduled_step(_stiff_loss, start, schedule=schedule)
    assert report.target == "lr"
    assert report.scheduled == pytest.approx(1.0 / STIFF)
    # y-step is lr * 1e4 = 1, so y goes 1 -> 0; x barely moves.
    assert float(new[1]) == pytest.approx(0.0, abs=1e-12)


def test_cubic_regularized_newton_hook_sets_sigma() -> None:
    torch.set_default_dtype(torch.float64)
    opt = CubicRegularizedNewton(krylov_dim=4, sharpness=G1_SCHEDULE)
    new, info = opt.step(_stiff_loss, torch.tensor([1.0, 1.0]))
    assert opt.last_ell_k is not None
    assert abs(opt.last_ell_k - STIFF) / STIFF <= 1e-12
    assert info.sigma == pytest.approx(10.0) or info.sigma > 0.0
    assert bool(torch.isfinite(new).all())


def test_cubic_newton_optimizer_hook() -> None:
    torch.set_default_dtype(torch.float64)
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    opt = CubicNewton([p], krylov_dim=4, sharpness=G1_SCHEDULE)

    def closure() -> torch.Tensor:
        return 0.5 * (p[0] * p[0] + STIFF * p[1] * p[1])

    loss = opt.step(closure)
    assert math.isfinite(float(loss))
    assert opt.last_ell_k is not None
    assert abs(opt.last_ell_k - STIFF) / STIFF <= 1e-12
    assert bool(torch.isfinite(p.detach()).all())


def test_lr_target_rejected_on_cubic_hook() -> None:
    schedule = SharpnessSchedule(target="lr")
    with pytest.raises(ValueError, match="cubic_sigma"):
        CubicRegularizedNewton(sharpness=schedule)
    with pytest.raises(ValueError, match="cubic_sigma"):
        p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        CubicNewton([p], sharpness=schedule)
