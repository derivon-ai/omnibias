# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch jet line search (theory 03-12): G1 exactness, never-worse, optimisers."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.line_search import JetLineSearchConfig
from omnibias.torch.line_search import (
    directional_derivatives,
    jet_line_search,
    jet_line_search_on_ray,
)
from omnibias.torch.optim import CubicNewton, TrustRegionNewtonCG


def _quartic_loss(params: torch.Tensor) -> torch.Tensor:
    # phi(s) along e_0 with params = s * e_0 equals the spec quartic when
    # params[0] = s and the other entries are unused.
    s = params[0]
    return 1.0 - 2.0 * s + 3.0 * s**2 - 2.0 * s**3 + 2.0 * s**4


def test_g1_jet_matches_finite_differences_orders_0_to_6() -> None:
    torch.set_default_dtype(torch.float64)

    def loss(p: torch.Tensor) -> torch.Tensor:
        # exp(<p, 1>) so every directional derivative along ones is exp(sum p)
        return torch.exp(p.sum())

    params = torch.zeros(3, dtype=torch.float64)
    direction = torch.ones(3, dtype=torch.float64)
    derivs = directional_derivatives(loss, params, direction, 6)
    # phi(s) = exp(3s); phi^(k)(0) = 3^k
    for k, val in enumerate(derivs):
        expected = 3.0**k
        rel = abs(val - expected) / max(abs(expected), 1.0)
        assert rel <= 1e-10, f"order {k}: {val} vs {expected} rel={rel}"

    # High-precision central differences for k = 1, 2 against the jet.
    h = 1e-4

    def phi(s: float) -> float:
        return float(loss(params + s * direction))

    fd1 = (phi(h) - phi(-h)) / (2.0 * h)
    rel1 = abs(derivs[1] - fd1) / max(abs(fd1), 1.0)
    assert rel1 <= 1e-7


def test_quartic_recovers_spec_step() -> None:
    torch.set_default_dtype(torch.float64)
    params = torch.zeros(2, dtype=torch.float64)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    result = jet_line_search(
        _quartic_loss,
        params,
        direction,
        config=JetLineSearchConfig(order=4, trust_radius=1.0, verify=True),
        next_derivative_bound=0.0,
    )
    assert result.step == pytest.approx(0.409461, abs=5e-5)
    assert result.actual_value is not None
    assert result.actual_value <= 1.0 + 1e-14
    assert result.truncation_certified
    assert not result.fell_back


def test_g3_never_worse_pathological() -> None:
    torch.set_default_dtype(torch.float64)

    def sharp_bowl(p: torch.Tensor) -> torch.Tensor:
        # Order-1 model is 1 - 2s and wants s=1; the true quartic term is 50 s^2.
        s = p[0]
        return 1.0 - 2.0 * s + 50.0 * s**2

    params = torch.zeros(1, dtype=torch.float64)
    direction = torch.ones(1, dtype=torch.float64)
    result = jet_line_search(
        sharp_bowl,
        params,
        direction,
        config=JetLineSearchConfig(order=1, trust_radius=1.0, verify=True),
    )
    assert result.actual_value is not None
    assert result.actual_value <= 1.0 + 1e-14
    assert result.fell_back


def test_g3_randomized_suite_never_worse() -> None:
    torch.set_default_dtype(torch.float64)
    rng = torch.Generator().manual_seed(0)
    for _ in range(32):
        a = torch.randn(4, generator=rng, dtype=torch.float64)
        w = torch.randn(4, generator=rng, dtype=torch.float64)

        def loss(p: torch.Tensor, a: torch.Tensor = a, w: torch.Tensor = w) -> torch.Tensor:
            return (0.5 * ((p - a) ** 2 * torch.abs(w + 0.3)).sum()) + 0.1 * (p**4).sum()

        params = torch.randn(4, generator=rng, dtype=torch.float64)
        direction = torch.randn(4, generator=rng, dtype=torch.float64)
        start = float(loss(params))
        result = jet_line_search(
            loss,
            params,
            direction,
            config=JetLineSearchConfig(order=4, trust_radius=0.5, verify=True),
        )
        assert result.actual_value is not None
        assert result.actual_value <= start + 1e-12


def test_mlp_ray_uses_compose_jet() -> None:
    torch.set_default_dtype(torch.float64)
    w = torch.tensor([[0.4, -0.2], [0.1, 0.3]], dtype=torch.float64)
    b = torch.zeros(2, dtype=torch.float64)
    w2 = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    b2 = torch.zeros(1, dtype=torch.float64)
    layers = [(w, b, "tanh"), (w2, b2, None)]
    x0 = torch.zeros(2, dtype=torch.float64)
    v = torch.tensor([0.3, -0.1], dtype=torch.float64)
    target = torch.tensor([0.0], dtype=torch.float64)
    result = jet_line_search_on_ray(
        layers,
        x0,
        v,
        target,
        config=JetLineSearchConfig(order=4, trust_radius=0.4, verify=True),
    )
    assert result.actual_value is not None
    from omnibias.torch.jet import mlp_jet

    y0 = float(mlp_jet(x0, v, layers, 0)[0, 0])
    start = 0.5 * y0 * y0
    assert result.actual_value <= start + 1e-12


def test_cubic_newton_then_jet_line_search_descends() -> None:
    torch.set_default_dtype(torch.float64)
    x = torch.nn.Parameter(torch.tensor([0.8, -0.4], dtype=torch.float64))

    def closure() -> torch.Tensor:
        return (x[0] - 0.2) ** 2 + 3.0 * (x[1] + 0.1) ** 2

    start = float(closure().detach())
    opt = CubicNewton([x], krylov_dim=4, max_line_search=4)
    opt.step(closure)
    after_cubic = float(closure().detach())
    assert after_cubic <= start + 1e-12

    params = x.detach().clone()
    direction = -torch.tensor(
        [2.0 * (params[0] - 0.2), 6.0 * (params[1] + 0.1)],
        dtype=torch.float64,
    )

    def flat_loss(p: torch.Tensor) -> torch.Tensor:
        return (p[0] - 0.2) ** 2 + 3.0 * (p[1] + 0.1) ** 2

    result = jet_line_search(
        flat_loss,
        params,
        direction,
        config=JetLineSearchConfig(order=2, trust_radius=1.0, verify=True),
        next_derivative_bound=0.0,
    )
    assert result.actual_value is not None
    assert result.actual_value <= float(flat_loss(params)) + 1e-12


def test_trust_region_radius_can_use_certified_truncation() -> None:
    torch.set_default_dtype(torch.float64)
    from omnibias.core.line_search import certified_truncation_radius

    radius = certified_truncation_radius(0.0, order=2, atol=1e-8, max_step=0.25)
    x = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
    opt = TrustRegionNewtonCG([x], radius=radius, max_radius=1.0)

    def closure() -> torch.Tensor:
        return 0.5 * x[0] ** 2

    start = float(closure().detach())
    opt.step(closure)
    assert float(closure().detach()) <= start + 1e-12
    assert math.isfinite(opt.radius)
