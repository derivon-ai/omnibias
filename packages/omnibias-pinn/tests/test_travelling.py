# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Travelling-wave / soliton field twins (theory 02-09). Tanh algebra, not a collapse."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.tanh_method import classical_pdes, published_ansatz
from omnibias.pinn.travelling.torch import SolitonField


def test_g3_burgers_residual() -> None:
    torch.set_default_dtype(torch.float64)
    pde = classical_pdes()["burgers"]
    ans = published_ansatz("burgers")
    field = SolitonField((ans,), dtype=torch.float64)
    x = torch.tensor([-1.0, 0.0, 0.5], dtype=torch.float64)
    t = torch.zeros_like(x)
    res = field.exact_residual(x, t, pde)
    mag = field(x, t).abs().max().clamp_min(1e-16)
    assert float(res.abs().max().detach()) <= 1e-14 * float(mag.detach())


def test_g5_heat_is_not_a_soliton() -> None:
    from omnibias.core.tanh_method import PDESpec, PDETerm, TermKind, solve_ansatz

    heat = PDESpec("heat", (PDETerm(TermKind.U_T, 1), PDETerm(TermKind.U_XX, -1)))
    assert solve_ansatz(heat) == ()


def test_g6_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.pinn.travelling.jax import soliton_apply

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    ans = published_ansatz("kdv")
    x = torch.tensor([0.1, -0.3], dtype=torch.float64)
    t = torch.tensor([0.0, 0.2], dtype=torch.float64)
    u_t = SolitonField((ans,), dtype=torch.float64)(x, t)
    u_j = soliton_apply((ans,), jnp.asarray(x.numpy()), jnp.asarray(t.numpy()))
    assert u_t.detach().cpu().numpy() == pytest.approx(u_j.tolist(), rel=0, abs=0)
    assert math.isfinite(float(u_t[0].detach()))
