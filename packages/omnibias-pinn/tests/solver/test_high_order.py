# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""High-order operator exactness: the closed-form tower vs nested autodiff.

This is the package's differentiator, tested crisply: the closed-form
arbitrary-order operators (a single ``sigma`` evaluation per order) match
brute-force nested autograd to machine precision -- with no nested autograd of
our own.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
from omnibias.pinn.solver.torch.fields import build_field  # noqa: E402


def _autograd_nth_derivative(field, coords, name, order, axis=0):
    x = coords.clone().requires_grad_(True)
    state = field(x)
    d = state.ops.value(state, name)
    for _ in range(order):
        (d,) = torch.autograd.grad(d.sum(), x, create_graph=True)
        d = d[:, axis]
    return d


def _autograd_laplacian(u, x):
    (g,) = torch.autograd.grad(u.sum(), x, create_graph=True)
    lap = x.new_zeros(x.shape[0])
    for i in range(x.shape[1]):
        (gi,) = torch.autograd.grad(g[:, i].sum(), x, create_graph=True)
        lap = lap + gi[:, i]
    return lap


def test_arbitrary_order_1d_derivative_is_exact() -> None:
    torch.set_default_dtype(torch.float64)
    dom = pde.Domain(("x",), ((-1.0, 1.0),))
    sys = pde.poisson(dom)  # scalar field "u" on a 1-D domain
    field = build_field(sys, hidden=12, activation="tanh", seed=3, weight_init_scale=1.5)
    coords = torch.linspace(-0.8, 0.8, 11).reshape(-1, 1)

    for order in (1, 2, 4, 6):
        state = field(coords)
        closed = state.ops.derivative(state, "u", axis="x", order=order)
        reference = _autograd_nth_derivative(field, coords, "u", order)
        assert torch.allclose(closed, reference, atol=1e-8, rtol=1e-7), (
            f"order {order}: max abs diff "
            f"{(closed - reference).abs().max().item():.2e}"
        )


def test_biharmonic_2d_matches_laplacian_of_laplacian() -> None:
    torch.set_default_dtype(torch.float64)
    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    sys = pde.poisson(dom)
    field = build_field(sys, hidden=10, activation="tanh", seed=7, weight_init_scale=2.0)
    coords = torch.rand(9, 2)

    state = field(coords)
    closed = state.ops.biharmonic(state, "u")

    x = coords.clone().requires_grad_(True)
    st = field(x)
    u = st.ops.value(st, "u")
    lap1 = _autograd_laplacian(u, x)
    lap2 = _autograd_laplacian(lap1, x)
    assert torch.allclose(closed, lap2, atol=1e-8, rtol=1e-7), (
        f"biharmonic max abs diff {(closed - lap2).abs().max().item():.2e}"
    )


def test_polylaplacian_equals_high_order_1d_derivative() -> None:
    torch.set_default_dtype(torch.float64)
    dom = pde.Domain(("x",), ((-1.0, 1.0),))
    sys = pde.poisson(dom)
    field = build_field(sys, hidden=8, activation="tanh", seed=1, weight_init_scale=1.5)
    coords = torch.linspace(-0.7, 0.7, 7).reshape(-1, 1)
    state = field(coords)
    # in 1-D, Delta^k u == d^{2k} u / dx^{2k}
    for k in (1, 2, 3):
        poly = state.ops.polylaplacian(state, "u", k=k)
        pure = state.ops.derivative(state, "u", axis="x", order=2 * k)
        assert torch.allclose(poly, pure, atol=1e-10)
