# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Weak-form VPINN tests (theory 02-04 G1/G2/G4/G5)."""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.core.scan import BankSpec
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.weak import (
    TestFunctionSpace,
    WeakForm,
    boundary_bound,
    eval_test,
    exact_moment,
)
from omnibias.fields.weak.torch import weak_loss, weak_residual


def _space() -> TestFunctionSpace:
    return TestFunctionSpace(
        BankSpec.uniform(0.4, 0.6, 3, scales=(10.0,)),
        orders=(2,),
        base="tanh",
        window=(0.0, 1.0),
    )


def test_g1_exact_moment_vs_gauss() -> None:
    space = TestFunctionSpace(
        BankSpec.uniform(0.3, 0.7, 3, scales=(2.0,)),
        orders=(2,),
        base="tanh",
        window=(0.0, 1.0),
    )
    exact = exact_moment(space, 0, 0)
    hi = gauss_legendre(((0.0, 1.0),), 48)
    lo = gauss_legendre(((0.0, 1.0),), 2)
    v_hi = np.array([eval_test(space, 0, float(x), deriv=0) for x in hi.nodes[:, 0]])
    v_lo = np.array([eval_test(space, 0, float(x), deriv=0) for x in lo.nodes[:, 0]])
    g_hi = float(np.dot(hi.weights, v_hi))
    g_lo = float(np.dot(lo.weights, v_lo))
    rel = abs(exact - g_hi) / max(abs(g_hi), 1e-16)
    assert rel <= 1e-13
    assert abs(g_lo - exact) > abs(g_hi - exact)


def test_g2_boundary_bound_changes_loss() -> None:
    torch.set_default_dtype(torch.float64)
    space = _space()
    op = WeakForm(diffusion=(1.0,), source=(2.0,))
    field = (0.0, 1.0, -1.0)  # x - x^2
    on = float(weak_loss(field, space, operator=op, include_boundary_bound=True))
    off = float(weak_loss(field, space, operator=op, include_boundary_bound=False))
    assert on != off
    bound = boundary_bound(space, deriv_bound=1.0)
    assert bound.hi > 0.0


def test_g4_weak_condition_vs_strong() -> None:
    space = _space()
    n = space.size
    quad = gauss_legendre(((0.0, 1.0),), 32)
    k = np.zeros((n, n), dtype=np.float64)
    a = np.zeros((n, n), dtype=np.float64)
    xs = np.linspace(0.15, 0.85, n)
    for i in range(n):
        for j in range(n):
            vi = np.array([eval_test(space, i, float(x), deriv=1) for x in quad.nodes[:, 0]])
            vj = np.array([eval_test(space, j, float(x), deriv=1) for x in quad.nodes[:, 0]])
            k[i, j] = float(np.dot(quad.weights, vi * vj))
            a[i, j] = -eval_test(space, j, float(xs[i]), deriv=2)
    cond_k = float(np.linalg.cond(k))
    cond_a = float(np.linalg.cond(a))
    assert cond_a / cond_k >= 10.0


def test_g5_torch_jax_exact_residual_parity() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.fields.weak.jax import weak_residual as jax_weak_residual

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    space = _space()
    op = WeakForm(diffusion=(1.0,), source=(2.0,))
    field = (0.0, 1.0, -1.0)
    t_res, t_terms = weak_residual(field, space, operator=op)
    j_res, j_terms = jax_weak_residual(field, space, operator=op)
    t = t_res.detach().cpu().numpy()
    j = np.asarray(j_res)
    assert np.allclose(t, j, rtol=0, atol=1e-15)
    assert all(term.path == "exact" for term in t_terms)
    assert all(term.path == "exact" for term in j_terms)
    del jnp


def test_path_recording_quadrature_for_callable_source() -> None:
    torch.set_default_dtype(torch.float64)
    space = _space()
    op = WeakForm(diffusion=(1.0,))
    quad = gauss_legendre(((0.0, 1.0),), 8)

    def source(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x.reshape(-1)).reshape(x.shape)

    _res, terms = weak_residual(
        (0.0, 1.0, -1.0), space, operator=op, source=source, quadrature=quad
    )
    load_paths = {t.path for t in terms if t.name.startswith("load")}
    stiff_paths = {t.path for t in terms if t.name.startswith("stiffness")}
    assert "quadrature" in load_paths
    assert "exact" in stiff_paths
