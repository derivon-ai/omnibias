# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for conservation cages on the torch backend."""

from __future__ import annotations

import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.cage import (
    HardBoundaryField,
    MassFluxPotentialField,
    StreamfunctionField,
    energy_conserving_advection,
    enstrophy_conserving_advection,
)
from omnibias.pinn.torch.fields.spectral import SpectralVectorField

# --- skew-symmetric advection -------------------------------------


def _make_div_free_state():
    """A streamfunction-cage state -> u, v are exactly divergence-free."""
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
    cage = StreamfunctionField(
        base=base, psi="psi", velocity_names=("u", "v"),
        passthrough_names=("p",),
    )
    coords = torch.zeros(8, 3, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.0, 1.0, 8)
    coords[:, 1] = torch.linspace(0.1, 1.5, 8)
    coords[:, 2] = torch.linspace(0.2, 1.7, 8)
    return cage(coords)


def _make_compressible_state():
    """A general spectral state (not divergence-free)."""
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")})
    field = SpectralVectorField(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=6, time_depth=1, activation="tanh",
    )
    torch.manual_seed(1)
    with torch.no_grad():
        field.W_t.normal_()
        field.beta_t.normal_(0.0, 0.1)
        field.V.normal_(0.0, 0.1)
        field.b_t.normal_(0.0, 0.1)
    coords = torch.zeros(8, 3, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.0, 1.0, 8)
    coords[:, 1] = torch.linspace(0.1, 1.5, 8)
    coords[:, 2] = torch.linspace(0.2, 1.7, 8)
    return field(coords)


def test_skew_advection_div_free_equals_standard():
    """For divergence-free u, skew = standard advection."""
    state = _make_div_free_state()
    standard = tops.advection(state, velocity=("u", "v"))
    skew = energy_conserving_advection(state, velocity=("u", "v"))
    assert torch.allclose(standard, skew, rtol=1e-12, atol=1e-12)


def test_skew_advection_general_formula():
    """For general u: skew - standard = 0.5 * div(u) * u."""
    state = _make_compressible_state()
    standard = tops.advection(state, velocity=("u", "v"))
    skew = energy_conserving_advection(state, velocity=("u", "v"))
    div_u = tops.divergence(state, ("u", "v"))
    u_v = tops.stack_components(state, ("u", "v"))
    expected_correction = 0.5 * div_u.unsqueeze(-1) * u_v
    assert torch.allclose(skew - standard, expected_correction,
                          rtol=1e-12, atol=1e-12)


def test_enstrophy_advection_div_free_equals_standard():
    """For div-free u, enstrophy advection = standard scalar advection."""
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("psi", "omega"))
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
    cage = StreamfunctionField(
        base=base, psi="psi", velocity_names=("u", "v"),
        passthrough_names=("omega",),
    )
    coords = torch.zeros(8, 3, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.0, 1.0, 8)
    coords[:, 1] = torch.linspace(0.1, 1.5, 8)
    coords[:, 2] = torch.linspace(0.2, 1.7, 8)
    state = cage(coords)
    std = tops.advection(state, velocity=("u", "v"), scalar="omega")
    skew = enstrophy_conserving_advection(
        state, velocity=("u", "v"), vorticity="omega",
    )
    assert torch.allclose(std, skew, rtol=1e-12, atol=1e-12)


# --- HardBoundary -------------------------------------------------


def test_hard_boundary_zero_value_on_boundary():
    """If g = 0, u(boundary) = d(boundary) * f(boundary) = 0."""
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    )
    mspec = ComponentSpec(("u",))
    base = SpectralVectorField(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=4, time_depth=1, activation="tanh",
    )
    torch.manual_seed(0)
    with torch.no_grad():
        base.W_t.normal_()
        base.beta_t.normal_(0.0, 0.1)
        base.V.normal_(0.0, 0.1)
        base.b_t.normal_(0.0, 0.1)

    def distance(coords: torch.Tensor) -> torch.Tensor:
        x, y = coords[..., 1], coords[..., 2]
        return (1.0 - x ** 2) * (1.0 - y ** 2)

    cage = HardBoundaryField(
        base=base, distance_fn=distance,
        boundary_value_fn=None,
        bounded_names=("u",),
    )

    boundary_coords = torch.tensor([
        [0.5, 1.0, 0.3],     # x = 1
        [0.5, -1.0, 0.7],    # x = -1
        [0.5, 0.2, 1.0],     # y = 1
        [0.5, -0.7, -1.0],   # y = -1
    ], dtype=torch.float64)
    state = cage(boundary_coords)
    val = state.u.value
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-13)


def test_hard_boundary_nonzero_value():
    """u = g + d*f recovers g exactly on boundary."""
    cspec = CoordinateSpec(
        axes=("t", "x"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0)),
    )
    mspec = ComponentSpec(("u",))
    base = SpectralVectorField(
        coordinate_spec=cspec, components=mspec,
        K=3, time_hidden=4, time_depth=1, activation="tanh",
    )
    torch.manual_seed(0)
    with torch.no_grad():
        base.W_t.normal_()
        base.beta_t.normal_(0.0, 0.1)
        base.V.normal_(0.0, 0.1)
        base.b_t.normal_(0.0, 0.1)

    def distance(coords: torch.Tensor) -> torch.Tensor:
        return 1.0 - coords[..., 1] ** 2

    def g(coords: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": torch.sin(coords[..., 0]) * coords[..., 1]}

    cage = HardBoundaryField(
        base=base, distance_fn=distance,
        boundary_value_fn=g,
        bounded_names=("u",),
    )
    boundary_coords = torch.tensor([
        [0.0, 1.0],
        [0.5, 1.0],
        [0.5, -1.0],
    ], dtype=torch.float64)
    state = cage(boundary_coords)
    expected = torch.tensor([
        torch.sin(torch.tensor(0.0, dtype=torch.float64)) * 1.0,
        torch.sin(torch.tensor(0.5, dtype=torch.float64)) * 1.0,
        torch.sin(torch.tensor(0.5, dtype=torch.float64)) * (-1.0),
    ], dtype=torch.float64)
    assert torch.allclose(state.u.value, expected, atol=1e-13)


def test_hard_boundary_derivative_via_leibniz():
    """Verify d(u)/dx = dg/dx + (dd/dx)*f + d*(df/dx) at an interior pt."""
    cspec = CoordinateSpec(
        axes=("t", "x"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0)),
    )
    mspec = ComponentSpec(("u",))
    base = SpectralVectorField(
        coordinate_spec=cspec, components=mspec,
        K=3, time_hidden=4, time_depth=1, activation="tanh",
    )
    torch.manual_seed(0)
    with torch.no_grad():
        base.W_t.normal_()
        base.beta_t.normal_(0.0, 0.1)
        base.V.normal_(0.0, 0.1)
        base.b_t.normal_(0.0, 0.1)

    def distance(coords: torch.Tensor) -> torch.Tensor:
        return 1.0 - coords[..., 1] ** 2

    def g(coords: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": torch.sin(coords[..., 0]) * coords[..., 1]}

    cage = HardBoundaryField(
        base=base, distance_fn=distance,
        boundary_value_fn=g,
        bounded_names=("u",),
    )
    coords = torch.tensor([
        [0.5, 0.3],
        [0.7, -0.4],
    ], dtype=torch.float64)
    state = cage(coords)
    inner = state.extra["_cage_inner_state"]
    # d/dx u = d/dx g + (-2x) * f + (1 - x^2) * df/dx
    coords_g = coords.detach().requires_grad_(True)
    g_x = g(coords_g)["u"]
    dg_dx, = torch.autograd.grad(g_x.sum(), coords_g, create_graph=True)
    dg_dx = dg_dx[..., 1]
    f_val = tops.value(inner, "u")
    df_dx = tops.derivative(inner, "u", axis="x")
    x = coords[..., 1]
    expected = dg_dx + (-2.0 * x) * f_val + (1.0 - x ** 2) * df_dx
    assert torch.allclose(state.u.dx, expected, rtol=1e-10, atol=1e-10)


# --- MassFluxPotential ---------------------------------------------


def test_mass_flux_potential_div_zero():
    cspec = CoordinateSpec(
        axes=("t", "x", "y", "z"),
        periodicity=(False, True, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("Psi1", "Psi2", "Psi3", "rho"))
    base = SpectralVectorField(
        coordinate_spec=cspec, components=mspec,
        K=3, time_hidden=4, time_depth=1, activation="tanh",
    )
    torch.manual_seed(0)
    with torch.no_grad():
        base.W_t.normal_()
        base.beta_t.normal_(0.0, 0.1)
        base.V.normal_(0.0, 0.1)
        base.b_t.normal_(0.0, 0.1)
    cage = MassFluxPotentialField(base=base)
    coords = torch.zeros(6, 4, dtype=torch.float64)
    coords[:, 0] = torch.linspace(0.0, 1.0, 6)
    coords[:, 1] = torch.linspace(0.1, 1.5, 6)
    coords[:, 2] = torch.linspace(0.2, 1.7, 6)
    coords[:, 3] = torch.linspace(-0.5, 1.0, 6)
    state = cage(coords)
    # The divergence here is div(rhou, rhov, rhow), which should be zero.
    div = tops.divergence(state, ("rhou", "rhov", "rhow"))
    assert torch.max(torch.abs(div)).item() < 1e-12
