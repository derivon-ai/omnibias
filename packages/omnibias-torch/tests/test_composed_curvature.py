# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch composed curvature (theory 08-02): G1 FD, G2 escape, G4."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.composed_curvature import ComposedCurvatureConfig
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.optim_composed import (
    composed_block_hessian,
    composed_curvature_step,
    scalar_nest_hessian,
)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def test_g1_compose_jet_and_hvp_match_fd() -> None:
    torch.set_default_dtype(torch.float64)
    w = torch.tensor(0.2)
    v = torch.tensor(0.1)
    closed = scalar_nest_hessian(w, v, 1.0)

    def residual(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(prev * 1.0)
        return (curr * h - 1.0).reshape(1)

    cfg = ComposedCurvatureConfig(n_directions=2, allow_full=True)
    h_slice, h_joint, cross = composed_block_hessian(
        residual,
        w,
        v,
        directions=(
            (torch.tensor(1.0), torch.tensor(0.0)),
            (torch.tensor(0.0), torch.tensor(1.0)),
        ),
        config=cfg,
    )
    assert _rel(float(h_joint[0, 0]), closed[0]) <= 1e-10
    assert _rel(float(h_joint[0, 1]), closed[1]) <= 1e-10
    assert _rel(float(h_joint[1, 1]), closed[2]) <= 1e-10
    assert _rel(float(h_slice[0, 0]), closed[2]) <= 1e-10
    assert _rel(float(cross[0, 0]), closed[1]) <= 1e-10


def test_g4_raises_without_allow_full() -> None:
    torch.set_default_dtype(torch.float64)

    def residual(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        return (curr * torch.tanh(prev) - 1.0).reshape(1)

    with pytest.raises(ValueError, match="allow_full"):
        composed_block_hessian(
            residual,
            torch.tensor(0.2),
            torch.tensor(0.1),
            config=ComposedCurvatureConfig(n_directions=2, allow_full=False),
        )


def _poisson_pair(xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pi = math.pi
    target = torch.sin(pi * xs)
    force = (pi**2) * target
    return target, force


def test_g2_poisson_slice_min_is_joint_saddle() -> None:
    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(-1.0, 1.0, 41)
    target, force = _poisson_pair(xs)
    spec = get_activation("tanh")
    fp = spec.fastpath
    assert fp is not None
    w0 = torch.tensor(3.0)

    def feature_xx(w: torch.Tensor) -> torch.Tensor:
        z = w * xs
        return (w**2) * fp(z, 2)

    a = feature_xx(w0)
    v0 = torch.tensor(-float(torch.dot(a, force)) / float(torch.dot(a, a)))

    def residual(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        return curr * feature_xx(prev) + force

    cfg = ComposedCurvatureConfig(
        n_directions=2,
        allow_full=True,
        include_residual_hess=True,
        escape_tol=0.0,
    )
    h_slice, h_joint, _ = composed_block_hessian(
        residual,
        w0,
        v0,
        directions=(
            (torch.tensor(1.0), torch.tensor(0.0)),
            (torch.tensor(0.0), torch.tensor(1.0)),
        ),
        config=cfg,
    )
    evals = torch.linalg.eigvalsh(h_joint)
    assert float(h_slice[0, 0]) > 0.0
    assert float(evals[0]) < 0.0

    def loss(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        r = residual(prev, curr)
        return 0.5 * torch.dot(r, r)

    eps = 1e-7
    g_v = (float(loss(w0, v0 + eps)) - float(loss(w0, v0 - eps))) / (2.0 * eps)
    assert abs(g_v) < 1e-6

    new_w, new_v, report = composed_curvature_step(residual, w0, v0, config=cfg)
    assert report.lambda_min_slice >= 0.0
    assert report.lambda_min_joint < 0.0
    assert report.escaped
    assert float(loss(new_w, new_v)) < float(loss(w0, v0))

    mse_zero = float(torch.mean(target**2))
    field0 = v0 * torch.tanh(w0 * xs)
    mse0 = float(torch.mean((field0 - target) ** 2))
    skill = 1.0 - mse0 / mse_zero
    assert skill > 0.0


def test_gn_only_does_not_claim_escape_on_the_psd_model() -> None:
    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(-1.0, 1.0, 41)
    _, force = _poisson_pair(xs)
    spec = get_activation("tanh")
    fp = spec.fastpath
    assert fp is not None
    w0 = torch.tensor(3.0)

    def residual(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        z = prev * xs
        return curr * (prev**2) * fp(z, 2) + force

    a = residual(w0, torch.tensor(1.0)) - force
    v0 = torch.tensor(-float(torch.dot(a, force)) / float(torch.dot(a, a)))
    cfg = ComposedCurvatureConfig(
        n_directions=2,
        allow_full=True,
        include_residual_hess=False,
    )
    _, h_joint, _ = composed_block_hessian(
        residual,
        w0,
        v0,
        directions=(
            (torch.tensor(1.0), torch.tensor(0.0)),
            (torch.tensor(0.0), torch.tensor(1.0)),
        ),
        config=cfg,
    )
    evals = torch.linalg.eigvalsh(h_joint)
    assert float(evals[0]) >= -1e-8
