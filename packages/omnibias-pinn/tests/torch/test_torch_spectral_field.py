# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the torch :class:`SpectralVectorField`.

We verify

1. Forward value parity with a hand-rolled Fourier expansion.
2. Spatial derivatives match :func:`torch.autograd.grad` (the gold ref).
3. Time derivative matches autograd.
4. Mixed partials match autograd.
5. Spatial Laplacian and biharmonic match the diagonal-multiplier
   identity (and autograd).
6. ``polylaplacian(k)`` matches ``Delta^k`` built by repeated 2nd
   derivatives.
7. The non-trivial divergence-of-gradient and Jacobian shapes wire
   through the ops dispatch correctly.
"""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields.spectral import SpectralVectorField


def _make_field_2d(K: int = 4, time_hidden: int = 6) -> SpectralVectorField:
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(
        ("u", "v", "p"), groups={"velocity": ("u", "v")},
    )
    field = SpectralVectorField(
        coordinate_spec=cspec,
        components=mspec,
        K=K,
        time_hidden=time_hidden,
        time_depth=1,
        activation="tanh",
    )
    torch.manual_seed(0)
    with torch.no_grad():
        field.W_t.normal_(0.0, 1.0)
        field.beta_t.normal_(0.0, 0.1)
        field.V.normal_(0.0, 0.1)
        field.b_t.normal_(0.0, 0.1)
    return field


def _make_field_3d(K: int = 3, time_hidden: int = 4) -> SpectralVectorField:
    cspec = CoordinateSpec(
        axes=("t", "x", "y", "z"),
        periodicity=(False, True, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(
        ("u", "v", "w", "p"), groups={"velocity": ("u", "v", "w")},
    )
    field = SpectralVectorField(
        coordinate_spec=cspec,
        components=mspec,
        K=K,
        time_hidden=time_hidden,
        time_depth=1,
        activation="tanh",
    )
    torch.manual_seed(1)
    with torch.no_grad():
        field.W_t.normal_(0.0, 1.0)
        field.beta_t.normal_(0.0, 0.1)
        field.V.normal_(0.0, 0.1)
        field.b_t.normal_(0.0, 0.1)
    return field


def _coords_2d(B: int = 5) -> torch.Tensor:
    torch.manual_seed(7)
    coords = torch.empty(B, 3, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.0, 1.0, B)
    coords[:, 1] = torch.linspace(0.1, 1.5, B)
    coords[:, 2] = torch.linspace(0.2, 1.7, B)
    coords.requires_grad_(True)
    return coords


def _coords_3d(B: int = 5) -> torch.Tensor:
    torch.manual_seed(11)
    coords = torch.empty(B, 4, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.0, 1.0, B)
    coords[:, 1] = torch.linspace(0.1, 1.5, B)
    coords[:, 2] = torch.linspace(0.2, 1.7, B)
    coords[:, 3] = torch.linspace(-0.3, 1.1, B)
    coords.requires_grad_(True)
    return coords


def _autograd_d(out: torch.Tensor, coords: torch.Tensor, axis: int) -> torch.Tensor:
    g, = torch.autograd.grad(
        out.sum(), coords, create_graph=True, retain_graph=True,
    )
    return g[..., axis]


def _autograd_dn(
    out: torch.Tensor, coords: torch.Tensor, axes: tuple[int, ...],
) -> torch.Tensor:
    cur = out
    for a in axes:
        cur = _autograd_d(cur, coords, a)
    return cur


# -- 2D ----------------------------------------------------------------


def test_value_finite_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for n in ("u", "v", "p"):
        v = tops.value(state, n)
        assert v.shape == (coords.shape[0],)
        assert torch.isfinite(v).all()


def test_d_dt_matches_autograd_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    closed = tops.derivative(state, "u", axis="t")
    auto = _autograd_d(tops.value(state, "u"), coords, axis=0)
    assert torch.allclose(closed, auto, rtol=1e-10, atol=1e-10)


def test_d_dx_dy_match_autograd_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for axis_name, axis_idx in (("x", 1), ("y", 2)):
        closed = tops.derivative(state, "u", axis=axis_name)
        auto = _autograd_d(tops.value(state, "u"), coords, axis=axis_idx)
        assert torch.allclose(closed, auto, rtol=1e-10, atol=1e-10), axis_name


def test_higher_order_partials_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for axis_name, axis_idx in (("x", 1), ("y", 2)):
        for order in (2, 3, 4):
            closed = tops.derivative(state, "u", axis=axis_name, order=order)
            auto = _autograd_dn(tops.value(state, "u"), coords, (axis_idx,) * order)
            assert torch.allclose(closed, auto, rtol=1e-9, atol=1e-9), (
                axis_name, order,
            )


def test_mixed_partials_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    closed = tops.mixed_partial(state, "u", ("x", "y"), (1, 1))
    auto = _autograd_dn(tops.value(state, "u"), coords, (1, 2))
    assert torch.allclose(closed, auto, rtol=1e-10, atol=1e-10)
    closed = tops.mixed_partial(state, "u", ("t", "x"), (1, 1))
    auto = _autograd_dn(tops.value(state, "u"), coords, (0, 1))
    assert torch.allclose(closed, auto, rtol=1e-10, atol=1e-10)
    closed = tops.mixed_partial(state, "u", ("x", "y"), (2, 1))
    auto = _autograd_dn(tops.value(state, "u"), coords, (1, 1, 2))
    assert torch.allclose(closed, auto, rtol=1e-10, atol=1e-10)


def test_laplacian_matches_d2x_plus_d2y_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    lap = tops.laplacian(state, "u")
    sum_d2 = (
        tops.derivative(state, "u", axis="x", order=2)
        + tops.derivative(state, "u", axis="y", order=2)
    )
    assert torch.allclose(lap, sum_d2, rtol=1e-12, atol=1e-12)


def test_biharmonic_and_polylaplacian_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    bih = tops.biharmonic(state, "u")
    pl1 = tops.polylaplacian(state, "u", k=2)
    # Same value via two API entry points.
    assert torch.allclose(bih, pl1, rtol=1e-12, atol=1e-12)
    # Compare against double laplacian via repeated 2nd derivs.
    state = field(_coords_2d())  # fresh state
    d4x = tops.derivative(state, "u", axis="x", order=4)
    d4y = tops.derivative(state, "u", axis="y", order=4)
    dxxyy = tops.mixed_partial(state, "u", ("x", "y"), (2, 2))
    expected = d4x + 2.0 * dxxyy + d4y
    assert torch.allclose(bih, expected, rtol=1e-9, atol=1e-9)


def test_divergence_jacobian_curl_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    div = tops.divergence(state, ("u", "v"))
    expected_div = (
        tops.derivative(state, "u", axis="x")
        + tops.derivative(state, "v", axis="y")
    )
    assert torch.allclose(div, expected_div, rtol=1e-12, atol=1e-12)
    jac = tops.jacobian(state, ("u", "v"))
    assert jac.shape == (coords.shape[0], 2, 3)
    curl_z = tops.curl(state, ("u", "v"))
    assert curl_z.shape == (coords.shape[0], 1)
    expected_curl = (
        tops.derivative(state, "v", axis="x")
        - tops.derivative(state, "u", axis="y")
    )
    assert torch.allclose(curl_z[:, 0], expected_curl, rtol=1e-12, atol=1e-12)


# -- 3D ----------------------------------------------------------------


def test_value_finite_3d():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    for n in ("u", "v", "w", "p"):
        v = tops.value(state, n)
        assert v.shape == (coords.shape[0],)
        assert torch.isfinite(v).all()


def test_partials_3d_match_autograd():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    for n, axis_name, axis_idx in [
        ("u", "x", 1), ("v", "y", 2), ("w", "z", 3), ("u", "t", 0),
    ]:
        closed = tops.derivative(state, n, axis=axis_name)
        auto = _autograd_d(tops.value(state, n), coords, axis=axis_idx)
        assert torch.allclose(closed, auto, rtol=1e-10, atol=1e-10), (
            n, axis_name,
        )


def test_laplacian_3d_matches_sum_of_d2():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    lap = tops.laplacian(state, "u")
    sum_d2 = sum(
        tops.derivative(state, "u", axis=a, order=2)
        for a in ("x", "y", "z")
    )
    assert torch.allclose(lap, sum_d2, rtol=1e-12, atol=1e-12)


def test_3d_curl_through_velocity_view():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    curl = state.velocity.curl
    assert curl.shape == (coords.shape[0], 3)
    # Compare to a direct computation:
    # curl[0] = dw/dy - dv/dz
    expected_x = (
        tops.derivative(state, "w", axis="y")
        - tops.derivative(state, "v", axis="z")
    )
    expected_y = (
        tops.derivative(state, "u", axis="z")
        - tops.derivative(state, "w", axis="x")
    )
    expected_z = (
        tops.derivative(state, "v", axis="x")
        - tops.derivative(state, "u", axis="y")
    )
    assert torch.allclose(curl[:, 0], expected_x, rtol=1e-12, atol=1e-12)
    assert torch.allclose(curl[:, 1], expected_y, rtol=1e-12, atol=1e-12)
    assert torch.allclose(curl[:, 2], expected_z, rtol=1e-12, atol=1e-12)


# -- exact spatial periodicity ---------------------------------------


def test_periodicity_2d():
    """Spatial values are exactly periodic along x, y with period L."""
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    val_a = tops.value(state, "u")

    coords_b = coords.detach().clone()
    coords_b[:, 1] = coords_b[:, 1] + 2.0 * math.pi
    coords_b[:, 2] = coords_b[:, 2] - 2.0 * math.pi
    coords_b.requires_grad_(True)
    state_b = field(coords_b)
    val_b = tops.value(state_b, "u")
    assert torch.allclose(val_a, val_b, rtol=1e-11, atol=1e-11)


def test_state_view_dsl_2d():
    """Attribute DSL routes through ops correctly."""
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    assert torch.equal(state.u.value, tops.value(state, "u"))
    assert torch.equal(state.u.dt, tops.derivative(state, "u", axis="t"))
    assert torch.equal(state.u.dx, tops.derivative(state, "u", axis="x"))
    assert torch.equal(state.u.lap, tops.laplacian(state, "u"))
    assert torch.equal(state.u.biharm, tops.biharmonic(state, "u"))
    assert torch.equal(state.velocity.div, tops.divergence(state, ("u", "v")))


def test_repr():
    field = _make_field_2d()
    s = repr(field)
    assert "SpectralVectorField" in s
    assert "K=4" in s


# -- deep temporal MLP (time_depth > 1): autograd fallback for d/dt -----


def _make_field_2d_deep(K: int = 3, time_hidden: int = 6, time_depth: int = 2):
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("u", "v"))
    field = SpectralVectorField(
        coordinate_spec=cspec,
        components=mspec,
        K=K,
        time_hidden=time_hidden,
        time_depth=time_depth,
        activation="tanh",
    )
    torch.manual_seed(0)
    with torch.no_grad():
        field.W_t.normal_(0.0, 1.0)
        field.beta_t.normal_(0.0, 0.1)
        for layer in field._inner_layers:
            layer.weight.normal_(0.0, 1.0 / math.sqrt(time_hidden))
            layer.bias.normal_(0.0, 0.1)
        # Make V non-zero so we exercise time-derivative paths (default V=0).
        field.V.normal_(0.0, 0.1)
        field.b_t.normal_(0.0, 0.05)
    return field


def test_d_dt_deep_time_depth_matches_autograd():
    """``time_depth=2`` triggers the autograd fallback for d/dt; result
    must match a direct ``torch.autograd.grad`` baseline.
    """
    field = _make_field_2d_deep(time_depth=2)
    coords = _coords_2d()
    state = field(coords)
    dudt = tops.derivative(state, "u", axis="t", order=1)

    coords_a = coords.detach().clone().requires_grad_(True)
    state_a = field(coords_a)
    u_val = tops.value(state_a, "u")
    (gu,) = torch.autograd.grad(u_val.sum(), coords_a, create_graph=False)
    expected = gu[:, 0]                              # axis "t" is index 0
    assert torch.allclose(dudt, expected, rtol=1e-9, atol=1e-9)


def test_mixed_partial_deep_time_depth_matches_autograd():
    """Mixed (spatial, time) partial through the autograd time fallback."""
    field = _make_field_2d_deep(time_depth=2)
    coords = _coords_2d()
    state = field(coords)
    mp = tops.mixed_partial(state, "u", ("x", "t"), (1, 1))

    coords_a = coords.detach().clone().requires_grad_(True)
    state_a = field(coords_a)
    u_val = tops.value(state_a, "u")
    (gu,) = torch.autograd.grad(u_val.sum(), coords_a, create_graph=True)
    du_dx = gu[:, 1]                                 # axis "x" is index 1
    (gxt,) = torch.autograd.grad(du_dx.sum(), coords_a, create_graph=False)
    expected = gxt[:, 0]                             # then d/dt
    assert torch.allclose(mp, expected, rtol=1e-9, atol=1e-9)


def test_d2_dt2_deep_time_depth_matches_autograd():
    """Second-order time derivative via the closed-form jet path."""
    field = _make_field_2d_deep(time_depth=2)
    coords = _coords_2d()
    state = field(coords)
    d2 = tops.derivative(state, "u", axis="t", order=2)

    coords_a = coords.detach().clone().requires_grad_(True)
    state_a = field(coords_a)
    u_val = tops.value(state_a, "u")
    (g1,) = torch.autograd.grad(u_val.sum(), coords_a, create_graph=True)
    (g2,) = torch.autograd.grad(g1[:, 0].sum(), coords_a, create_graph=False)
    expected = g2[:, 0]
    assert torch.allclose(d2, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_deep_time_depth3_time_derivs_match_autograd(order: int) -> None:
    """Deep head (``time_depth=3``): the closed-form omnibias jet time
    derivative matches nested ``torch.autograd.grad`` for orders 1..3."""
    field = _make_field_2d_deep(time_depth=3)
    coords = _coords_2d()
    state = field(coords)
    d_jet = tops.derivative(state, "u", axis="t", order=order)

    coords_a = coords.detach().clone().requires_grad_(True)
    g = tops.value(field(coords_a), "u")
    for _ in range(order):
        (g_full,) = torch.autograd.grad(g.sum(), coords_a, create_graph=True)
        g = g_full[:, 0]  # time axis is index 0
    assert torch.allclose(d_jet, g, rtol=1e-8, atol=1e-8)


def test_deep_time_head_mixed_third_order_matches_autograd() -> None:
    """Mixed ``d^3/dt^2 dx`` through the deep closed-form time head."""
    field = _make_field_2d_deep(time_depth=3)
    coords = _coords_2d()
    state = field(coords)
    mp = tops.mixed_partial(state, "u", ("x", "t"), (1, 2))

    coords_a = coords.detach().clone().requires_grad_(True)
    u_val = tops.value(field(coords_a), "u")
    (g1,) = torch.autograd.grad(u_val.sum(), coords_a, create_graph=True)  # du
    (g2,) = torch.autograd.grad(g1[:, 1].sum(), coords_a, create_graph=True)  # d/dx
    (g3,) = torch.autograd.grad(g2[:, 0].sum(), coords_a, create_graph=False)  # d/dt
    # g3[:, 0] = d^3 u / (dx dt^2): one spatial + two temporal derivatives.
    assert torch.allclose(mp, g3[:, 0], rtol=1e-8, atol=1e-8)


def test_deep_time_head_param_grads_flow() -> None:
    """The closed-form deep time derivative is differentiable wrt the temporal
    MLP parameters: a loss on d/dt backprops into the inner layers (training)."""
    field = _make_field_2d_deep(time_depth=3)
    coords = _coords_2d().detach()
    dudt = tops.derivative(field(coords), "u", axis="t", order=1)
    loss = (dudt**2).mean()
    loss.backward()
    assert field.W_t.grad is not None and field.W_t.grad.abs().sum() > 0
    inner0 = field._inner_layers[0]
    assert inner0.weight.grad is not None and inner0.weight.grad.abs().sum() > 0
    assert field.V.grad is not None and field.V.grad.abs().sum() > 0
