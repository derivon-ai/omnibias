# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the torch :class:`OneLayerVectorField` and the basic ops it wires."""

from __future__ import annotations

import pytest
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.fields import OneLayerVectorField


@pytest.fixture
def small_field():
    torch.manual_seed(0)
    coord_spec = CoordinateSpec(("x", "y", "t"))
    comp_spec = ComponentSpec(
        ("u", "v", "p"),
        groups={"velocity": ("u", "v")},
    )
    field = OneLayerVectorField(
        coordinate_spec=coord_spec,
        components=comp_spec,
        hidden=8,
        base="tanh",
        weight_init_scale=0.5,
        dtype=torch.float64,
    )
    return field


@pytest.fixture
def small_coords():
    torch.manual_seed(0)
    return torch.randn(7, 3, dtype=torch.float64)


def test_field_evaluate_returns_state(small_field, small_coords):
    state = small_field(small_coords)
    assert state.coords is small_coords
    assert state.field is small_field
    assert state.components.names == ("u", "v", "p")
    assert state.coordinate_spec.axes == ("x", "y", "t")


def test_field_value_via_attribute_dsl(small_field, small_coords):
    state = small_field(small_coords)
    u_val = state.u.value
    v_val = state.v.value
    p_val = state.p.value
    assert u_val.shape == (7,)
    assert v_val.shape == (7,)
    assert p_val.shape == (7,)
    assert torch.isfinite(u_val).all()
    assert torch.isfinite(v_val).all()
    assert torch.isfinite(p_val).all()

    # Should match the reference forward pass.
    z = small_field._pre_activations(small_coords)         # (B, H)
    sigma_z = small_field._sigma(z)                        # (B, H)
    expected_u = small_field.value(sigma_z, "u")
    assert torch.allclose(u_val, expected_u, rtol=1e-15, atol=1e-15)


def test_value_op_matches_view(small_field, small_coords):
    state = small_field(small_coords)
    direct = ops.value(state, "u")
    via_view = state.u.value
    assert torch.allclose(direct, via_view, rtol=1e-15, atol=1e-15)


def test_first_partial_via_dsl(small_field, small_coords):
    state = small_field(small_coords)
    du_dx = state.u.dx
    du_dy = state.u.dy
    du_dt = state.u.dt
    assert du_dx.shape == (7,)

    # Cross-check against autograd.
    coords_grad = small_coords.detach().clone().requires_grad_(True)
    state_grad = small_field(coords_grad)
    u_val = state_grad.u.value
    g_full, = torch.autograd.grad(u_val.sum(), coords_grad, create_graph=False)
    assert torch.allclose(du_dx, g_full[:, 0], rtol=1e-10, atol=1e-12)
    assert torch.allclose(du_dy, g_full[:, 1], rtol=1e-10, atol=1e-12)
    assert torch.allclose(du_dt, g_full[:, 2], rtol=1e-10, atol=1e-12)


def test_gradient_is_spatial_only(small_field, small_coords):
    state = small_field(small_coords)
    g = state.u.grad
    # Default is spatial only -> 2 axes (x, y).
    assert g.shape == (7, 2)
    g_full = ops.gradient(state, "u", axes=("x", "y", "t"))
    assert g_full.shape == (7, 3)


def test_laplacian_against_autograd(small_field, small_coords):
    state = small_field(small_coords)
    L = state.u.lap
    assert L.shape == (7,)

    coords_grad = small_coords.detach().clone().requires_grad_(True)
    state_grad = small_field(coords_grad)
    u_val = state_grad.u.value
    g, = torch.autograd.grad(u_val.sum(), coords_grad, create_graph=True)
    L_ref = torch.zeros_like(u_val)
    for axis in (0, 1):  # spatial only
        gi, = torch.autograd.grad(
            g[:, axis].sum(), coords_grad, create_graph=True,
        )
        L_ref = L_ref + gi[:, axis]
    assert torch.allclose(L, L_ref, rtol=1e-9, atol=1e-12)


def test_hessian_against_autograd(small_field, small_coords):
    state = small_field(small_coords)
    H = state.u.hess  # (B, D, D)
    assert H.shape == (7, 3, 3)

    coords_grad = small_coords.detach().clone().requires_grad_(True)
    state_grad = small_field(coords_grad)
    u_val = state_grad.u.value
    g, = torch.autograd.grad(u_val.sum(), coords_grad, create_graph=True)
    rows = []
    for j in range(3):
        gj, = torch.autograd.grad(g[:, j].sum(), coords_grad, create_graph=True)
        rows.append(gj)
    H_ref = torch.stack(rows, dim=-1)  # (B, D_in, D_out): row index is the "outer" partial
    # H[b, i, j] = d^2 u / dx_i dx_j; autograd-of-grad gives g_ij = d/dx_j (du/dx_i),
    # so H_ref above is shape (B, i, j) when rows[j] is gj = d/dx_j (g) = d^2 u / dx_i dx_j
    # for each i. We assembled it as stack(rows, dim=-1) which yields (B, i, j) as required.
    assert torch.allclose(H, H_ref, rtol=1e-9, atol=1e-12)
    # Symmetry sanity.
    assert torch.allclose(H, H.transpose(-1, -2), rtol=1e-12, atol=1e-12)


def test_biharmonic_against_polylaplacian_k_2(small_field, small_coords):
    state = small_field(small_coords)
    B = state.u.biharm
    P = ops.polylaplacian(state, "u", k=2)
    assert torch.allclose(B, P, rtol=1e-15, atol=1e-15)


def test_polylaplacian_k_1_equals_laplacian(small_field, small_coords):
    state = small_field(small_coords)
    L = state.u.lap
    P = ops.polylaplacian(state, "u", k=1)
    assert torch.allclose(L, P, rtol=1e-15, atol=1e-15)


def test_divergence_2d(small_field, small_coords):
    state = small_field(small_coords)
    div = ops.divergence(state, ("u", "v"))
    expected = state.u.dx + state.v.dy
    assert torch.allclose(div, expected, rtol=1e-15, atol=1e-15)


def test_curl_2d_returns_scalar_in_one_slot(small_field, small_coords):
    state = small_field(small_coords)
    c = ops.curl(state, ("u", "v"))
    assert c.shape == (7, 1)
    expected = (state.v.dx - state.u.dy).unsqueeze(-1)
    assert torch.allclose(c, expected, rtol=1e-15, atol=1e-15)


def test_strain_rate_is_symmetric(small_field, small_coords):
    state = small_field(small_coords)
    S = ops.strain_rate(state, ("u", "v"))
    assert S.shape == (7, 2, 2)
    assert torch.allclose(S, S.transpose(-1, -2), rtol=1e-15, atol=1e-15)


def test_self_advection_2d(small_field, small_coords):
    state = small_field(small_coords)
    adv = state.velocity.advect()
    assert adv.shape == (7, 2)
    u, v = state.u, state.v
    expected = torch.stack([
        u.value * u.dx + v.value * u.dy,
        u.value * v.dx + v.value * v.dy,
    ], dim=-1)
    assert torch.allclose(adv, expected, rtol=1e-12, atol=1e-12)


def test_material_derivative_2d(small_field, small_coords):
    state = small_field(small_coords)
    Dt = state.velocity.material_derivative()
    expected = state.velocity.dt + state.velocity.advect()
    assert torch.allclose(Dt, expected, rtol=1e-12, atol=1e-12)


def test_sigma_cache_is_reused(small_field, small_coords):
    state = small_field(small_coords)
    # Touch several ops; each distinct order should be cached once.
    _ = state.u.value
    _ = state.u.dx
    _ = state.u.lap
    _ = state.u.biharm
    _ = state.u.hess
    orders = state.sigma_cache.orders()
    # value -> 0, dx -> 1, lap -> 2, hess -> 2 (already cached), biharm -> 4.
    assert orders == (0, 1, 2, 4)


def test_p_laplacian_equals_laplacian_at_p_2(small_field, small_coords):
    state = small_field(small_coords)
    L = state.u.lap
    P = ops.p_laplacian(state, "u", p=2.0)
    assert torch.allclose(L, P, rtol=1e-12, atol=1e-12)


def test_did_you_mean_on_typo(small_field, small_coords):
    state = small_field(small_coords)
    with pytest.raises(AttributeError) as ei:
        _ = state.velocty
    assert "velocity" in str(ei.value)


def test_repr_field(small_field):
    r = repr(small_field)
    assert "OneLayerVectorField" in r
    assert "hidden=8" in r
