# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 06-05 obligation 3: mlp_jet Poisson residual matches nested AD."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from omnibias.torch.jet import jet_to_tower, mlp_jet  # noqa: E402


@pytest.fixture(autouse=True)
def _float64() -> None:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def test_mlp_jet_uxx_matches_nested_ad() -> None:
    torch.manual_seed(0)
    hidden = 8
    w = torch.randn(hidden, 1) * 0.5
    beta = torch.randn(hidden) * 0.1
    c = torch.randn(hidden) * 0.1
    b = torch.zeros(1)
    layers = ((w, beta, "tanh"), (c.reshape(1, -1), b, None))
    x = torch.linspace(0.0, 1.0, 20).reshape(-1, 1)
    tower = jet_to_tower(mlp_jet(x, torch.ones_like(x), layers, 2))
    xx = x.detach().clone().requires_grad_(True)
    z = xx @ w.T + beta
    u = (torch.tanh(z) * c).sum(-1) + b.reshape(())
    du = torch.autograd.grad(u.sum(), xx, create_graph=True)[0]
    d2 = torch.autograd.grad(du.sum(), xx)[0]
    assert float((tower[0].reshape(-1) - u.detach()).abs().max()) <= 1e-14
    assert float((tower[2].reshape(-1) - d2.reshape(-1).detach()).abs().max()) <= 1e-14
    f = (math.pi**2) * torch.sin(math.pi * x).reshape(-1)
    r_jet = -tower[2].reshape(-1) - f
    r_ad = -d2.reshape(-1).detach() - f
    assert float((r_jet - r_ad).abs().max()) <= 1e-14
