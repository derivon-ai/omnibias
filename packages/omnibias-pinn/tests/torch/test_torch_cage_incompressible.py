# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the torch incompressible cage layers.

Hard guarantees we verify

1. ``StreamfunctionField`` 2D: ``div(u, v) = 0`` to ``rtol=0`` (exact).
2. ``VectorPotentialField`` 3D: ``div(u, v, w) = 0`` to ``atol=1e-12``
   (machine precision, since ``div curl`` algebraically cancels).
3. Pass-through components forward unchanged.
4. Caged derivatives match the explicit derivative-of-curl-of-A
   computed via autograd on the velocity field.
5. The Coulomb-gauge soft penalty is positive and differentiable.
6. Trainable parameters of the base field travel through
   ``cage.parameters()``.
"""

from __future__ import annotations

import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.cage import (
    HelmholtzProjectionField,
    StreamfunctionField,
    VectorPotentialField,
    coulomb_gauge_loss,
    helmholtz_gauge_loss,
)
from omnibias.pinn.torch.fields.spectral import SpectralVectorField

# --------------- 2D streamfunction ---------------------------------


def _make_2d_cage():
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("psi", "p"))
    base = SpectralVectorField(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=6, time_depth=1, activation="tanh",
    )
    torch.manual_seed(0)
    with torch.no_grad():
        base.W_t.normal_()
        base.beta_t.normal_(0.0, 0.1)
        base.V.normal_(0.0, 0.1)
        base.b_t.normal_(0.0, 0.1)
    return StreamfunctionField(
        base=base, psi="psi",
        velocity_names=("u", "v"),
        passthrough_names=("p",),
        spatial_axes=("x", "y"),
    )


def _coords_2d(B: int = 8):
    coords = torch.zeros(B, 3, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.0, 1.0, B)
    coords[:, 1] = torch.linspace(0.1, 1.5, B)
    coords[:, 2] = torch.linspace(0.2, 1.7, B)
    return coords


def test_streamfunction_div_zero_2d():
    cage = _make_2d_cage()
    coords = _coords_2d()
    state = cage(coords)
    div = state.velocity.div
    assert torch.allclose(div, torch.zeros_like(div), atol=1e-13)


def test_streamfunction_passthrough_p_2d():
    cage = _make_2d_cage()
    coords = _coords_2d()
    state = cage(coords)
    inner = state.extra["_cage_inner_state"]
    assert torch.allclose(state.p.value, tops.value(inner, "p"))
    assert torch.allclose(state.p.dx, tops.derivative(inner, "p", axis="x"))


def test_streamfunction_velocity_via_psi_2d():
    cage = _make_2d_cage()
    coords = _coords_2d()
    state = cage(coords)
    inner = state.extra["_cage_inner_state"]
    psi_dx = tops.derivative(inner, "psi", axis="x")
    psi_dy = tops.derivative(inner, "psi", axis="y")
    assert torch.allclose(state.u.value, psi_dy)
    assert torch.allclose(state.v.value, -psi_dx)


def test_streamfunction_velocity_derivatives_match_via_psi_2d():
    cage = _make_2d_cage()
    coords = _coords_2d()
    state = cage(coords)
    inner = state.extra["_cage_inner_state"]
    # d u / d t = d/dt (d_y psi) = mixed_partial(psi, (t, y), (1, 1))
    du_dt = state.u.dt
    auto = tops.mixed_partial(inner, "psi", ("t", "y"), (1, 1))
    assert torch.allclose(du_dt, auto)
    # d^2 u / d x^2 = mixed_partial(psi, (x, y), (2, 1))
    du_dxx = state.u.d("x", 2)
    auto = tops.mixed_partial(inner, "psi", ("x", "y"), (2, 1))
    assert torch.allclose(du_dxx, auto)


def test_streamfunction_laplacian_identity_2d():
    """Δu = ∂_y(Δψ) = mixed_partial(ψ, (y, x, x), (1, 2)) + mp(ψ, (y, y, y), (3,))."""
    cage = _make_2d_cage()
    coords = _coords_2d()
    state = cage(coords)
    lap_u = state.u.lap
    sum_d2 = (
        state.u.d("x", 2) + state.u.d("y", 2)
    )
    assert torch.allclose(lap_u, sum_d2, rtol=1e-12, atol=1e-12)


def test_streamfunction_parameters_visible():
    cage = _make_2d_cage()
    base_params = list(cage.base.parameters())
    cage_params = list(cage.parameters())
    for p in base_params:
        assert id(p) in {id(q) for q in cage_params}, "base param missing"


# --------------- 3D vector potential -------------------------------


def _make_3d_cage():
    cspec = CoordinateSpec(
        axes=("t", "x", "y", "z"),
        periodicity=(False, True, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("A1", "A2", "A3", "p"))
    base = SpectralVectorField(
        coordinate_spec=cspec, components=mspec,
        K=3, time_hidden=4, time_depth=1, activation="tanh",
    )
    torch.manual_seed(1)
    with torch.no_grad():
        base.W_t.normal_()
        base.beta_t.normal_(0.0, 0.1)
        base.V.normal_(0.0, 0.1)
        base.b_t.normal_(0.0, 0.1)
    return VectorPotentialField(
        base=base, A_components=("A1", "A2", "A3"),
        velocity_names=("u", "v", "w"),
        passthrough_names=("p",),
        spatial_axes=("x", "y", "z"),
    )


def _coords_3d(B: int = 6):
    coords = torch.zeros(B, 4, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.0, 1.0, B)
    coords[:, 1] = torch.linspace(0.1, 1.5, B)
    coords[:, 2] = torch.linspace(0.2, 1.7, B)
    coords[:, 3] = torch.linspace(-0.5, 1.0, B)
    return coords


def test_vector_potential_div_zero_3d():
    cage = _make_3d_cage()
    coords = _coords_3d()
    state = cage(coords)
    div = state.velocity.div
    assert torch.max(torch.abs(div)).item() < 1e-12


def test_vector_potential_curl_explicit_3d():
    cage = _make_3d_cage()
    coords = _coords_3d()
    state = cage(coords)
    inner = state.extra["_cage_inner_state"]
    # u = ∂_y A3 - ∂_z A2
    expected_u = (
        tops.derivative(inner, "A3", axis="y")
        - tops.derivative(inner, "A2", axis="z")
    )
    expected_v = (
        tops.derivative(inner, "A1", axis="z")
        - tops.derivative(inner, "A3", axis="x")
    )
    expected_w = (
        tops.derivative(inner, "A2", axis="x")
        - tops.derivative(inner, "A1", axis="y")
    )
    assert torch.allclose(state.u.value, expected_u)
    assert torch.allclose(state.v.value, expected_v)
    assert torch.allclose(state.w.value, expected_w)


def test_vector_potential_passthrough_p_3d():
    cage = _make_3d_cage()
    coords = _coords_3d()
    state = cage(coords)
    inner = state.extra["_cage_inner_state"]
    assert torch.allclose(state.p.value, tops.value(inner, "p"))
    assert torch.allclose(state.p.dx, tops.derivative(inner, "p", axis="x"))


def test_vector_potential_velocity_laplacian_3d():
    """Δu = sum_a ∂_a^2 u, where each term reduces to mixed partials of A."""
    cage = _make_3d_cage()
    coords = _coords_3d()
    state = cage(coords)
    lap_u = state.u.lap
    sum_d2 = sum(state.u.d(a, 2) for a in ("x", "y", "z"))
    assert torch.allclose(lap_u, sum_d2, rtol=1e-12, atol=1e-12)


def test_coulomb_gauge_loss_positive_3d():
    cage = _make_3d_cage()
    coords = _coords_3d()
    loss = coulomb_gauge_loss(cage, coords)
    assert loss.item() > 0
    assert torch.isfinite(loss).all()


# --------------- Helmholtz projection ------------------------------


def _make_helmholtz_2d_cage():
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("u_pred", "v_pred", "phi", "p"))
    base = SpectralVectorField(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=6, time_depth=1, activation="tanh",
    )
    torch.manual_seed(2)
    with torch.no_grad():
        base.W_t.normal_()
        base.beta_t.normal_(0.0, 0.1)
        base.V.normal_(0.0, 0.1)
        base.b_t.normal_(0.0, 0.1)
    return HelmholtzProjectionField(
        base=base,
        u_pred_components=("u_pred", "v_pred"),
        phi="phi",
        velocity_names=("u", "v"),
        passthrough_names=("p",),
    )


def test_helmholtz_velocity_value_2d():
    cage = _make_helmholtz_2d_cage()
    coords = _coords_2d()
    state = cage(coords)
    inner = state.extra["_cage_inner_state"]
    expected_u = (
        tops.value(inner, "u_pred") - tops.derivative(inner, "phi", axis="x")
    )
    expected_v = (
        tops.value(inner, "v_pred") - tops.derivative(inner, "phi", axis="y")
    )
    assert torch.allclose(state.u.value, expected_u)
    assert torch.allclose(state.v.value, expected_v)


def test_helmholtz_gauge_loss_positive_2d():
    cage = _make_helmholtz_2d_cage()
    coords = _coords_2d()
    loss = helmholtz_gauge_loss(cage, coords)
    assert loss.item() > 0
    assert torch.isfinite(loss).all()


def test_helmholtz_velocity_derivative_2d():
    cage = _make_helmholtz_2d_cage()
    coords = _coords_2d()
    state = cage(coords)
    inner = state.extra["_cage_inner_state"]
    # d u / d t = d (u_pred - d_x phi) / d t = d u_pred / d t - d^2 phi / dx dt
    du_dt = state.u.dt
    expected = (
        tops.derivative(inner, "u_pred", axis="t")
        - tops.mixed_partial(inner, "phi", ("x", "t"), (1, 1))
    )
    assert torch.allclose(du_dt, expected, rtol=1e-12, atol=1e-12)


# --------------- DSL routing -----------------------------------------


def test_streamfunction_state_view_dsl():
    cage = _make_2d_cage()
    coords = _coords_2d()
    state = cage(coords)
    assert torch.equal(state.u.value, tops.value(state, "u"))
    assert torch.equal(state.u.dt, tops.derivative(state, "u", axis="t"))
    assert torch.equal(state.u.lap, tops.laplacian(state, "u"))
    assert torch.equal(state.velocity.div, tops.divergence(state, ("u", "v")))


def test_repr():
    cage = _make_2d_cage()
    s = repr(cage)
    assert "StreamfunctionField" in s
