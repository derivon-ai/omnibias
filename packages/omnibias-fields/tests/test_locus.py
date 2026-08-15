# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equality-locus tensor twins G5/G6 (theory 01-09)."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.locus import EqualitySystem, UnitTerm
from omnibias.core.locus import newton_project as newton_core
from omnibias.fields.locus.torch import newton_project, newton_project_unrolled


def _case_b() -> EqualitySystem:
    return EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(2, -2.0, (0.0, 1.0), 0.0),
        )
    )


def test_g5_ift_matches_unrolled() -> None:
    sys = _case_b()
    x0 = torch.tensor([0.0, 0.20], dtype=torch.float64)
    w = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    x_ift = newton_project(sys, x0, weights=w, max_iter=8, tol=1e-14)
    loss = x_ift[1]
    loss.backward()
    g_ift = w.grad.detach().clone()

    w2 = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    x_un = newton_project_unrolled(sys, x0, weights=w2, max_iter=8, tol=1e-14)
    x_un[1].backward()
    g_un = w2.grad.detach().clone()
    rel = torch.norm(g_ift - g_un) / torch.norm(g_un).clamp_min(1e-30)
    assert float(rel.detach()) <= 1e-8


def test_g6_torch_jax_newton_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.fields.locus.jax import newton_project as newton_jax

    jax.config.update("jax_enable_x64", True)
    sys = _case_b()
    x0_t = torch.tensor([0.0, 0.20], dtype=torch.float64)
    x_t = newton_project(sys, x0_t)
    x_j = newton_jax(sys, jnp.asarray([0.0, 0.20], dtype=jnp.float64))
    assert x_t.detach().cpu().numpy() == pytest.approx(x_j.tolist(), rel=0, abs=0)
    core = newton_core(sys, (0.0, 0.20))
    assert x_t[1].item() == pytest.approx(core.point[1], rel=0, abs=0)
    assert math.isfinite(float(x_t[1].detach()))
