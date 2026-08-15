# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named linearizing transforms (theory 02-13). No 03-11 search."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.transforms_pde import (
    TransformKind,
    cole_hopf_from_heat_phi,
    named_cole_hopf,
    permutability,
    verify_transform,
)
from omnibias.pinn.transform.torch import ColeHopfField


def test_g1_cole_hopf_exact() -> None:
    t = named_cole_hopf(nu=1.0)
    assert t.kind is TransformKind.COLE_HOPF
    assert verify_transform(t)
    u = cole_hopf_from_heat_phi(0.0, 0.0, nu=1.0)
    assert abs(u + 2.0) <= 1e-15


def test_g2_cole_hopf_field_heat_residual() -> None:
    torch.set_default_dtype(torch.float64)
    field = ColeHopfField(nu=1.0, dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.zeros_like(x)
    res = field.linear_residual(x, t)
    mag = field.phi(x, t).abs().max().clamp_min(1e-16)
    assert float(res.abs().max().detach()) <= 1e-14 * float(mag.detach())
    u = field(x, t)
    assert abs(float(u[0].detach()) + 2.0) <= 1e-12


def test_g5_wrong_transform_rejected() -> None:
    from omnibias.core.tanh_method import PDESpec, PDETerm, TermKind
    from omnibias.core.transforms_pde import LinearizingTransform

    heat = PDESpec("heat", (PDETerm(TermKind.U_T, 1), PDETerm(TermKind.U_XX, -1)))
    bogus = LinearizingTransform(TransformKind.MIURA, heat, heat, {})
    assert verify_transform(bogus) is False


def test_backlund_permutability_finite() -> None:
    u3 = permutability(0.0, 1.0, -1.0, a1=2.0, a2=1.0)
    assert math.isfinite(u3)


def test_g6_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.pinn.transform.jax import cole_hopf_apply

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    x = torch.tensor([0.2, -0.1], dtype=torch.float64)
    t = torch.tensor([0.0, 0.3], dtype=torch.float64)
    u_t = ColeHopfField(nu=1.0, dtype=torch.float64)(x, t)
    u_j = cole_hopf_apply(jnp.asarray(x.numpy()), jnp.asarray(t.numpy()), nu=1.0, k=1.0)
    assert u_t.detach().cpu().numpy() == pytest.approx(u_j.tolist(), rel=0, abs=0)
