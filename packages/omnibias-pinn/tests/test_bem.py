# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""BEM-Net G1/G5 (theory 02-06). Off-surface exact; BC approximated."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.pinn.bem._core import (
    KernelSpec,
    Surface,
    half_plane_dtn,
    pde_residual_off_surface,
    poisson_pair_dictionary,
    single_layer,
)
from omnibias.pinn.bem.torch import BEMNet


def _ulp(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_g1_off_surface_residual() -> None:
    torch.set_default_dtype(torch.float64)
    surface = Surface("circle", radius=1.0, n_quad=16)
    kernel = KernelSpec("laplace", dimension=2)
    net = BEMNet(surface, kernel, dtype=torch.float64)
    with torch.no_grad():
        net.log_density.copy_(torch.randn(16, dtype=torch.float64))
    pts = torch.tensor([[2.0, 0.0], [0.0, -2.5], [1.5, 1.5]], dtype=torch.float64)
    res = net.pde_residual(pts)
    mag = net.evaluate(pts).abs().max().clamp_min(1e-16)
    assert float(res.abs().max().detach()) <= 1e-13 * float(mag.detach())
    dens = [float(v) for v in net.density().detach().tolist()]
    for pt in pts.tolist():
        val = single_layer((pt[0], pt[1]), surface, dens, kernel)
        assert math.isfinite(val)
        assert pde_residual_off_surface((pt[0], pt[1]), surface, dens, kernel) == 0.0


def test_g5_half_plane_dtn_ulp() -> None:
    dictionary, coeffs = poisson_pair_dictionary(scale=1.0)
    y = 0.3
    got = half_plane_dtn(dictionary, coeffs, y)
    # Analytic: Q'(y) for Q = y/(y^2+a^2), a=1 -> (1-y^2)/(y^2+1)^2
    expect = (1.0 - y * y) / (y * y + 1.0) ** 2
    assert _ulp(got, expect) <= 4.0


def test_g4_mollifier_order_smoke() -> None:
    surface = Surface("circle", radius=1.0, n_quad=24)
    exact = KernelSpec("laplace", dimension=2)
    x = (2.0, 0.0)
    dens = [1.0] * 24
    u0 = single_layer(x, surface, dens, exact)
    errs = []
    for eps in (0.2, 0.1, 0.05):
        moll = KernelSpec("laplace", dimension=2, regularization=eps)
        u = single_layer(x, surface, dens, moll)
        errs.append(abs(u - u0))
    # Error shrinks as eps halves (order at least 1).
    assert errs[1] < errs[0]
    assert errs[2] < errs[1]


def test_g6_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.pinn.bem.jax import bem_evaluate

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    surface = Surface("circle", radius=1.0, n_quad=8)
    kernel = KernelSpec("laplace", dimension=2)
    dens = [0.5, -0.2, 0.1, 0.0, 0.3, -0.1, 0.2, 0.05]
    x = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    net = BEMNet(surface, kernel, dtype=torch.float64)
    with torch.no_grad():
        net.log_density.copy_(torch.tensor(dens, dtype=torch.float64))
    u_t = net.evaluate(x)
    u_j = bem_evaluate(jnp.asarray(x.numpy()), surface, dens, kernel)
    assert u_t.detach().cpu().numpy() == pytest.approx(u_j.tolist(), rel=0, abs=0)
