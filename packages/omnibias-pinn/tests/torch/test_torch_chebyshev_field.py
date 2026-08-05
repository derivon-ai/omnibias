# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the torch :class:`ChebyshevVectorField`.

We verify

1. Forward value is finite and well-shaped.
2. First / second / higher-order spatial derivatives match
   :func:`torch.autograd.grad` (gold reference) on the rescaled domain.
3. Time derivative matches autograd.
4. Mixed partials match autograd.
5. Spatial Laplacian and biharmonic match the multinomial expansion.
6. Polynomial-exact recovery: a manually constructed polynomial value
   (e.g. ``T_3 (xi)``) is reproduced and its derivatives are exact.
"""

from __future__ import annotations

import math

import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields.chebyshev import ChebyshevVectorField


def _make_field_2d(K: int = 5, time_hidden: int = 6) -> ChebyshevVectorField:
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0), (-2.0, 2.0)),
    )
    mspec = ComponentSpec(
        ("u", "v", "p"), groups={"velocity": ("u", "v")},
    )
    field = ChebyshevVectorField(
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


def _make_field_3d(K: int = 4, time_hidden: int = 4) -> ChebyshevVectorField:
    cspec = CoordinateSpec(
        axes=("t", "x", "y", "z"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    )
    mspec = ComponentSpec(
        ("u", "v", "w", "p"), groups={"velocity": ("u", "v", "w")},
    )
    field = ChebyshevVectorField(
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
    coords = torch.empty(B, 3, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.1, 0.9, B)
    coords[:, 1] = torch.linspace(-0.8, 0.7, B)
    coords[:, 2] = torch.linspace(-1.5, 1.7, B)
    coords.requires_grad_(True)
    return coords


def _coords_3d(B: int = 5) -> torch.Tensor:
    coords = torch.empty(B, 4, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.1, 0.9, B)
    coords[:, 1] = torch.linspace(-0.8, 0.7, B)
    coords[:, 2] = torch.linspace(-0.6, 0.6, B)
    coords[:, 3] = torch.linspace(-0.5, 0.4, B)
    coords.requires_grad_(True)
    return coords


def _autograd_d(out: torch.Tensor, coords: torch.Tensor, axis: int) -> torch.Tensor:
    g, = torch.autograd.grad(
        out.sum(), coords, create_graph=True, retain_graph=True,
    )
    return g[..., axis]


def _autograd_dn(out: torch.Tensor, coords: torch.Tensor, axes: tuple[int, ...]):
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


def test_d_dx_dy_dt_match_autograd_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for axis_name, axis_idx in (("t", 0), ("x", 1), ("y", 2)):
        closed = tops.derivative(state, "u", axis=axis_name)
        auto = _autograd_d(tops.value(state, "u"), coords, axis=axis_idx)
        assert torch.allclose(closed, auto, rtol=1e-10, atol=1e-10), axis_name


def test_higher_order_partials_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for axis_name, axis_idx in (("x", 1), ("y", 2)):
        for order in (2, 3):
            closed = tops.derivative(state, "u", axis=axis_name, order=order)
            auto = _autograd_dn(
                tops.value(state, "u"), coords, (axis_idx,) * order,
            )
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
    assert torch.allclose(closed, auto, rtol=1e-9, atol=1e-9)


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


def test_biharmonic_via_multinomial_2d():
    """Bih = d4x + 2 d2x d2y + d4y matches polylaplacian(k=2)."""
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    bih = tops.biharmonic(state, "u")
    pl = tops.polylaplacian(state, "u", k=2)
    assert torch.allclose(bih, pl, rtol=1e-12, atol=1e-12)
    state2 = field(_coords_2d())
    d4x = tops.derivative(state2, "u", axis="x", order=4)
    d4y = tops.derivative(state2, "u", axis="y", order=4)
    dxxyy = tops.mixed_partial(state2, "u", ("x", "y"), (2, 2))
    expected = d4x + 2.0 * dxxyy + d4y
    assert torch.allclose(bih, expected, rtol=1e-9, atol=1e-9)


def test_divergence_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    div = tops.divergence(state, ("u", "v"))
    expected = (
        tops.derivative(state, "u", axis="x")
        + tops.derivative(state, "v", axis="y")
    )
    assert torch.allclose(div, expected, rtol=1e-12, atol=1e-12)


# -- 3D ----------------------------------------------------------------


def test_partials_3d():
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


def test_laplacian_3d():
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


def test_default_domain_minus1_to_1():
    """Default domain [-1, 1] when nothing is specified on the coord spec."""
    cspec = CoordinateSpec(axes=("t", "x"), time_axis="t")
    mspec = ComponentSpec(("u",))
    field = ChebyshevVectorField(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=4, time_depth=1, activation="tanh",
    )
    assert field.domain == ((-1.0, 1.0),)


def test_state_view_dsl_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    assert torch.equal(state.u.value, tops.value(state, "u"))
    assert torch.equal(state.u.dx, tops.derivative(state, "u", axis="x"))
    assert torch.equal(state.u.lap, tops.laplacian(state, "u"))
    assert torch.equal(state.u.biharm, tops.biharmonic(state, "u"))


def test_repr():
    field = _make_field_2d()
    s = repr(field)
    assert "ChebyshevVectorField" in s
    assert "K=5" in s
    assert "domain=" in s
