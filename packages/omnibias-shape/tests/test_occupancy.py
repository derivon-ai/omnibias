# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soft occupancy: hardening, separability, and closed-form box derivatives (torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.shape.torch import ops as shape  # noqa: E402


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _axes(m: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.arange(m, dtype=torch.float64), torch.arange(n, dtype=torch.float64)


def test_soft_interval_in_unit_interval_and_hardens():
    t = torch.linspace(-5, 15, 200)
    soft = shape.soft_interval(t, center=5.0, side=6.0, beta=8.0)
    # Membership is in (0, 1); far from the box it underflows to exactly 0.0.
    assert torch.all(soft >= 0.0) and torch.all(soft < 1.0)
    assert torch.all(soft[(t >= 3.0) & (t <= 7.0)] > 0.0)
    hard = shape.soft_interval(t, center=5.0, side=6.0, beta=200.0)
    inside = (t >= 2.0) & (t <= 8.0)
    assert torch.all(hard[inside] > 0.98)
    assert torch.all(hard[~inside][(t[~inside] < 1.0) | (t[~inside] > 9.0)] < 0.02)


def test_soft_box_separable_and_hardens_to_area():
    axes = _axes(11, 11)
    centers = torch.tensor([[5.0, 5.0]])
    occ = shape.soft_box(axes, centers, side=5.0, beta=100.0)[0]
    # 5x5 square centred at (5,5) covers integer rows/cols 3..7 -> 25 cells.
    assert abs(float(occ.sum()) - 25.0) < 0.5
    assert float(occ[5, 5]) > 0.99
    assert float(occ[0, 0]) < 0.01


def test_soft_box_grad_matches_autodiff():
    axes = _axes(7, 8)
    torch.manual_seed(0)
    centers = torch.rand(3, 2, requires_grad=True) * 6.0 + 1.0
    cf = shape.soft_box_grad(axes, centers.detach(), side=3.0, beta=2.0)

    def occ_sum(c: torch.Tensor) -> torch.Tensor:
        return shape.soft_box(axes, c, side=3.0, beta=2.0).sum()

    auto = torch.autograd.functional.jacobian(occ_sum, centers)
    assert torch.allclose(cf.sum(dim=(-1, -2)), auto, atol=1e-10)


def test_soft_box_hessian_matches_autodiff_block_diagonal():
    axes = _axes(6, 6)
    torch.manual_seed(1)
    centers = torch.rand(2, 2, requires_grad=True) * 4.0 + 1.0
    cf = shape.soft_box_hessian(axes, centers.detach(), side=3.0, beta=2.5)

    def occ_sum(c: torch.Tensor) -> torch.Tensor:
        return shape.soft_box(axes, c, side=3.0, beta=2.5).sum()

    auto = torch.autograd.functional.hessian(occ_sum, centers).reshape(4, 4)
    # occ is separable per shape -> cross-shape Hessian blocks are exactly zero.
    for k in range(2):
        block = cf[k].sum(dim=(-1, -2))  # (2, 2)
        assert torch.allclose(block, auto[2 * k : 2 * k + 2, 2 * k : 2 * k + 2], atol=1e-10)
    assert abs(float(auto[0:2, 2:4].abs().max())) < 1e-10


def test_soft_disk_and_polytope_shapes():
    axes = _axes(9, 9)
    disk = shape.soft_disk(axes, torch.tensor([[4.0, 4.0]]), radius=2.0, beta=50.0)
    assert disk.shape == (1, 9, 9)
    assert float(disk[0, 4, 4]) > 0.99 and float(disk[0, 0, 0]) < 0.01
    normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    offsets = torch.tensor([6.0, -2.0, 6.0, -2.0])
    poly = shape.soft_polytope(axes, normals, offsets, beta=50.0)
    assert poly.shape == (9, 9)
    assert float(poly[4, 4]) > 0.9
