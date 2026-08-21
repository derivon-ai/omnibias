# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Block exact search, PyTorch (theory 08-07)."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.torch.optim_block_search import (
    block_exact_search,
    block_exact_sweep,
    last_linear_block,
    ombu_bias_block,
)

SEEDS = (0, 1, 2, 3, 4)


def _section5_loss(params: torch.Tensor) -> torch.Tensor:
    p = params.reshape(-1)
    hidden = torch.tensor([1.0, 0.5], dtype=p.dtype, device=p.device)
    return (p[0] * hidden[0] + p[1] * hidden[1] - 1.0) ** 2


def test_g2_exact_quadratic() -> None:
    torch.set_default_dtype(torch.float64)
    v = torch.tensor([0.0, 0.0])
    new, result = block_exact_search(
        _section5_loss,
        v,
        mask=(True, False),
        exact_quadratic=True,
    )
    assert result.fell_back is False
    assert abs(result.step - 1.0) <= 1e-12
    assert abs(float(new[0]) - 1.0) <= 1e-12
    assert abs(float(new[1])) <= 1e-12
    assert result.actual_value is not None
    assert abs(result.actual_value) <= 1e-12
    assert abs(result.model_value) <= 1e-12


def test_g1_last_linear_beats_adam() -> None:
    torch.set_default_dtype(torch.float64)
    wins = 0
    skills = 0
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        hidden = torch.randn(12, 3, generator=g)
        v_true = torch.tensor([0.4, -0.3, 0.7])
        target = hidden @ v_true

        def loss(params: torch.Tensor, h: torch.Tensor = hidden, y: torch.Tensor = target) -> torch.Tensor:
            r = h @ params.reshape(-1) - y
            return 0.5 * torch.dot(r, r)

        v0 = torch.zeros(3)
        l0 = float(loss(v0))
        new, _report = block_exact_search(
            loss, v0, spec=last_linear_block(3), exact_quadratic=True
        )
        block_loss = float(loss(new))
        p = torch.nn.Parameter(v0.clone())
        opt = torch.optim.Adam([p], lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            value = loss(p)
            value.backward()
            opt.step()
        adam_loss = float(loss(p.detach()))
        wins += int(block_loss <= adam_loss + 1e-12)
        skills += int(block_loss < l0)
    assert wins == len(SEEDS)
    assert skills == len(SEEDS)


def test_g3_never_worse_random_blocks() -> None:
    torch.set_default_dtype(torch.float64)
    for seed in range(20):
        g = torch.Generator().manual_seed(seed + 11)
        q = torch.randn(4, generator=g)
        diag = torch.tensor([1.0, 4.0, 0.5, 2.0])

        def loss(params: torch.Tensor, scale: torch.Tensor = diag) -> torch.Tensor:
            p = params.reshape(-1)
            return 0.5 * torch.dot(p * scale, p * scale)

        mask = [(int(torch.randint(0, 2, (1,), generator=g)) == 1) for _ in range(4)]
        if not any(mask):
            mask[0] = True
        start = q.clone()
        l0 = float(loss(start))
        new, result = block_exact_search(loss, start, mask=mask, exact_quadratic=True)
        actual = float(loss(new))
        assert math.isfinite(actual)
        assert actual <= l0 + 1e-12
        if result.actual_value is not None:
            assert result.actual_value <= l0 + 1e-12


def test_sweep_and_named_ombu_mask() -> None:
    torch.set_default_dtype(torch.float64)

    def loss(params: torch.Tensor) -> torch.Tensor:
        p = params.reshape(-1)
        return (p[1] - 2.0) ** 2

    start = torch.zeros(3)
    spec = ombu_bias_block(n_channels=1, k_biases=3, slot=1)
    new, reports = block_exact_sweep(loss, start, spec=spec, exact_quadratic=True)
    assert len(reports) == 1
    assert abs(float(new[1]) - 2.0) <= 1e-12
    assert reports[0].fell_back is False


def test_empty_mask_raises() -> None:
    torch.set_default_dtype(torch.float64)
    with pytest.raises(ValueError, match="empty"):
        block_exact_search(_section5_loss, torch.zeros(2), mask=(False, False))
