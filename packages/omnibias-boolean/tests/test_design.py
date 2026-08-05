# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable design losses (torch): spectral objectives reduce under SGD."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.boolean._core.truth_table import pm1_values  # noqa: E402
from omnibias.boolean.torch.ops import design, spectrum  # noqa: E402


def test_target_spectrum_loss_decreases() -> None:
    # Drive a free 2-bit function toward the Walsh spectrum of XOR.
    target = spectrum.walsh_coeffs(
        torch.tensor(pm1_values((0, 1, 1, 0)), dtype=torch.float64)
    ).detach()
    vals = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([vals], lr=0.2)
    first = None
    last = None
    for _ in range(200):
        opt.zero_grad()
        loss = design.target_spectrum_loss(vals, target, basis="walsh")
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if first is None:
            first = last
    assert last is not None and first is not None
    assert last < first
    assert last < 1e-3


def test_degree_penalty_orders_functions() -> None:
    # A degree-1 dictator has less high-order energy than degree-2 XOR.
    dictator = torch.tensor(pm1_values((0, 0, 1, 1)), dtype=torch.float64)  # f = x1
    xor = torch.tensor(pm1_values((0, 1, 1, 0)), dtype=torch.float64)
    assert float(design.degree_penalty(dictator)) < float(design.degree_penalty(xor))


def test_influence_penalty_is_differentiable() -> None:
    vals = torch.rand(8, dtype=torch.float64, requires_grad=True)
    design.influence_penalty(vals).backward()
    assert vals.grad is not None
