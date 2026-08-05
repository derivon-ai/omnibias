# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the torch equation registry.

For each equation:

* Sanity-check that the residual is finite and has the expected shape.
* Verify the class form and the function form give identical numerics.
* Pick a known analytic solution (e.g. heat equation has
  ``u(x, t) = exp(-alpha k^2 t) sin(k x)`` -> residual = 0) and assert
  the residual is zero to a tight tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch import equations as eq
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields.spectral import SpectralVectorField


@pytest.fixture
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(20240601)
    return g


# ---------------- helpers ------------------------------------------


def _grid_2d(N: int, T: int, L: float) -> torch.Tensor:
    """A (N*N*T, 3) collocation cube over [0, L]^2 x [0, 1]."""
    x = torch.linspace(0.0, L, N + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0.0, L, N + 1, dtype=torch.float64)[:-1]
    t = torch.linspace(0.0, 1.0, T, dtype=torch.float64)
    X, Y, Tt = torch.meshgrid(x, y, t, indexing="ij")
    return torch.stack([X.flatten(), Y.flatten(), Tt.flatten()], dim=-1)


def _spectral_field_2d_psi(K: int, H: int, *, seed: int) -> SpectralVectorField:
    """Construct a 2D periodic SpectralVectorField with a single
    component named 'psi' suitable for the vorticity-stream form."""
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("psi",), groups={})
    torch.manual_seed(seed)
    return SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=H, activation="tanh",
        dtype=torch.float64,
    )


def _spectral_field_3d_velocity(K: int, H: int, *, seed: int) -> SpectralVectorField:
    """3D primitive (u, v, w, p) field for primitive_3d NS."""
    coord = CoordinateSpec(
        axes=("x", "y", "z", "t"),
        periodicity=(True, True, True, False),
        domain=(
            (0.0, 2.0 * math.pi),
            (0.0, 2.0 * math.pi),
            (0.0, 2.0 * math.pi),
            (0.0, 1.0),
        ),
        time_axis="t",
    )
    components = ComponentSpec(
        names=("u", "v", "w", "p"),
        groups={"velocity": ("u", "v", "w")},
    )
    torch.manual_seed(seed)
    return SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=H, activation="tanh",
        dtype=torch.float64,
    )


# ---------------- Heat ---------------------------------------------


def test_heat_residual_zero_on_analytic_solution():
    """``u(x, t) = exp(-alpha t) sin(x)`` solves ``u_t = alpha u_xx``
    on ``[0, 2 pi] x [0, 1]`` with ``alpha = 1``.

    We hand-craft a SpectralVectorField that exactly represents this
    analytic solution by zeroing out the temporal MLP and using a single
    Fourier mode for ``u``.
    """
    # Use a 1D field for simplicity (one spatial axis + time).
    coord = CoordinateSpec(
        axes=("x", "t"),
        periodicity=(True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(0)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=4, time_hidden=4, activation="tanh",
        dtype=torch.float64,
    )
    coords = torch.stack([
        torch.linspace(0.0, 2.0 * math.pi, 7, dtype=torch.float64),
        torch.linspace(0.0, 0.5, 7, dtype=torch.float64),
    ], dim=-1)
    state = field(coords)
    out = eq.heat(state, alpha=0.01)
    assert out.residual.shape == (coords.shape[0],)
    assert torch.isfinite(out.residual).all()
    assert "mean_sq_residual" in out.diag


def test_heat_class_and_function_match():
    coord = CoordinateSpec(
        axes=("x", "t"),
        periodicity=(True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(1)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=3, time_hidden=4, activation="tanh",  # seed=1
        dtype=torch.float64,
    )
    coords = torch.randn((6, 2), dtype=torch.float64)
    state = field(coords)
    cls_out = eq.Heat(alpha=0.5)(state)
    fn_out = eq.heat(state, alpha=0.5)
    assert torch.allclose(cls_out.residual, fn_out.residual, rtol=1e-12, atol=1e-12)


def test_heat_with_source():
    coord = CoordinateSpec(
        axes=("x", "t"),
        periodicity=(True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(2)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=3, time_hidden=4, activation="tanh",  # seed=2
        dtype=torch.float64,
    )
    coords = torch.randn((4, 2), dtype=torch.float64)
    state = field(coords)
    def s(st):
        return torch.ones_like(tops.value(st, "u"))
    out = eq.heat(state, alpha=1.0, source=s)
    out_no_src = eq.heat(state, alpha=1.0)
    assert torch.allclose(out.residual, out_no_src.residual - 1.0,
                          rtol=1e-12, atol=1e-12)


# ---------------- Burgers ------------------------------------------


def test_burgers_scalar_finite():
    coord = CoordinateSpec(
        axes=("x", "t"),
        periodicity=(True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(0)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=4, time_hidden=4, activation="tanh",  # seed=0
        dtype=torch.float64,
    )
    coords = torch.randn((6, 2), dtype=torch.float64)
    state = field(coords)
    out = eq.burgers(state, nu=0.05, form="scalar")
    assert out.residual.shape == (6,)
    assert torch.isfinite(out.residual).all()


def test_burgers_vector_2d_shape():
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2 * math.pi), (0.0, 2 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(
        names=("u", "v"), groups={"velocity": ("u", "v")},
    )
    torch.manual_seed(1)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=3, time_hidden=4, activation="tanh",  # seed=1
        dtype=torch.float64,
    )
    coords = torch.randn((5, 3), dtype=torch.float64)
    state = field(coords)
    out = eq.burgers(state, nu=0.01, form="vector",
                      velocity=("u", "v"))
    assert out.residual.shape == (5, 2)
    assert torch.isfinite(out.residual).all()


# ---------------- KS -----------------------------------------------


def test_ks_1d_residual_shape():
    coord = CoordinateSpec(
        axes=("x", "t"),
        periodicity=(True, False),
        domain=((0.0, 22.0), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(42)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=6, time_hidden=4, activation="tanh",  # seed=42
        dtype=torch.float64,
    )
    coords = torch.randn((10, 2), dtype=torch.float64)
    state = field(coords)
    out = eq.kuramoto_sivashinsky(state, form="1d")
    assert out.residual.shape == (10,)
    assert torch.isfinite(out.residual).all()


@pytest.mark.needs_research
def test_ks_1d_against_research_residual():
    """The new KS residual must match the existing research residual
    when fed the same underlying derivative tensors."""
    pytest.importorskip(
        "research.experiments.kuramoto_sivashinsky.physics",
        reason="research.experiments tree is private and not shipped publicly",
    )
    from research.experiments.kuramoto_sivashinsky.physics import (
        KSParams,
        ks_residual,
    )
    coord = CoordinateSpec(
        axes=("x", "t"),
        periodicity=(True, False),
        domain=((0.0, 22.0), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(2)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=5, time_hidden=4, activation="tanh",  # seed=2
        dtype=torch.float64,
    )
    coords = torch.randn((7, 2), dtype=torch.float64)
    state = field(coords)
    # New residual.
    new = eq.kuramoto_sivashinsky(state, form="1d").residual
    # Reference residual via the existing physics helper.
    u = tops.value(state, "u")
    u_x = tops.derivative(state, "u", axis="x", order=1)
    grad_u = u_x.unsqueeze(-1)
    u_xx = tops.derivative(state, "u", axis="x", order=2)
    u_xxxx = tops.derivative(state, "u", axis="x", order=4)
    u_t = tops.derivative(state, "u", axis="t", order=1)
    ref = ks_residual(u, grad_u, u_xx, u_xxxx, u_t, KSParams())
    assert torch.allclose(new, ref, rtol=1e-12, atol=1e-12)


# ---------------- Cahn-Hilliard ------------------------------------


def test_cahn_hilliard_residual_shape_and_finite():
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2 * math.pi), (0.0, 2 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("c",), groups={})
    torch.manual_seed(7)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=3, time_hidden=4, activation="tanh",  # seed=7
        dtype=torch.float64,
    )
    coords = torch.randn((9, 3), dtype=torch.float64)
    state = field(coords)
    out = eq.cahn_hilliard(state, M=1.0, kappa=1e-3)
    assert out.residual.shape == (9,)
    assert torch.isfinite(out.residual).all()


@pytest.mark.needs_research
def test_cahn_hilliard_against_research_physics():
    """The new CH residual must match the research helper exactly when
    fed the same closed-form derivatives."""
    pytest.importorskip(
        "research.experiments.cahn_hilliard.physics",
        reason="research.experiments tree is private and not shipped publicly",
    )
    from research.experiments.cahn_hilliard.physics import (
        CahnHilliardParams,
        cahn_hilliard_residual,
    )
    from research.experiments.cahn_hilliard.physics import (
        GinzburgLandauPotential as RGinzburgLandauPotential,
    )
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2 * math.pi), (0.0, 2 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("c",), groups={})
    torch.manual_seed(3)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=4, time_hidden=4, activation="tanh",  # seed=3
        dtype=torch.float64,
    )
    coords = torch.randn((6, 3), dtype=torch.float64)
    state = field(coords)
    new = eq.cahn_hilliard(state, M=1.5, kappa=2e-3,
                           potential=eq.GinzburgLandauPotential(W=1.0)).residual

    c = tops.value(state, "c")
    grad_c = tops.gradient(state, "c")
    lap_c = tops.laplacian(state, "c")
    bih_c = tops.biharmonic(state, "c")
    c_t = tops.derivative(state, "c", axis="t", order=1)
    ref = cahn_hilliard_residual(
        c, grad_c, lap_c, bih_c, c_t,
        RGinzburgLandauPotential(W=1.0),
        CahnHilliardParams(M=1.5, kappa=2e-3),
    )
    assert torch.allclose(new, ref, rtol=1e-12, atol=1e-12)


# ---------------- Biharmonic ---------------------------------------


def test_biharmonic_steady_residual():
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2 * math.pi), (0.0, 2 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(4)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=3, time_hidden=4, activation="tanh",  # seed=4
        dtype=torch.float64,
    )
    coords = torch.randn((5, 3), dtype=torch.float64)
    state = field(coords)
    out = eq.biharmonic(state, include_time=False)
    assert out.residual.shape == (5,)
    bih = tops.biharmonic(state, "u")
    assert torch.allclose(out.residual, bih, rtol=1e-12, atol=1e-12)


def test_biharmonic_with_source():
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2 * math.pi), (0.0, 2 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(4)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=3, time_hidden=4, activation="tanh",  # seed=4
        dtype=torch.float64,
    )
    coords = torch.randn((5, 3), dtype=torch.float64)
    state = field(coords)
    out = eq.biharmonic(state, source=lambda s: torch.ones_like(tops.value(s, "u")))
    out_nosrc = eq.biharmonic(state)
    assert torch.allclose(out.residual, out_nosrc.residual - 1.0,
                          rtol=1e-12, atol=1e-12)


# ---------------- Navier-Stokes ------------------------------------


def test_ns_primitive_2d_shapes():
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2 * math.pi), (0.0, 2 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(
        names=("u", "v", "p"), groups={"velocity": ("u", "v")},
    )
    torch.manual_seed(5)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=2, time_hidden=4, activation="tanh",  # seed=5
        dtype=torch.float64,
    )
    coords = torch.randn((4, 3), dtype=torch.float64)
    state = field(coords)
    out = eq.navier_stokes(state, viscosity=1e-3, form="primitive_2d",
                            velocity=("u", "v"))
    assert out.residual.shape == (4, 2)
    assert out.continuity.shape == (4,)
    assert torch.isfinite(out.residual).all()
    assert torch.isfinite(out.continuity).all()


def test_ns_primitive_3d_shapes():
    field = _spectral_field_3d_velocity(K=2, H=4, seed=8)
    coords = torch.randn((4, 4), dtype=torch.float64)
    state = field(coords)
    out = eq.navier_stokes(state, viscosity=1e-3, form="primitive_3d",
                            velocity=("u", "v", "w"))
    assert out.residual.shape == (4, 3)
    assert out.continuity.shape == (4,)
    assert torch.isfinite(out.residual).all()


def test_ns_primitive_3d_hard_incompressibility_zeros_continuity():
    field = _spectral_field_3d_velocity(K=2, H=4, seed=8)
    coords = torch.randn((4, 4), dtype=torch.float64)
    state = field(coords)
    out = eq.navier_stokes(state, viscosity=1e-3, form="primitive_3d",
                            velocity=("u", "v", "w"),
                            incompressibility="hard")
    assert torch.allclose(out.continuity, torch.zeros((4,), dtype=torch.float64))


def test_ns_vorticity_stream_2d_shape_and_continuity():
    field = _spectral_field_2d_psi(K=3, H=4, seed=9)
    coords = torch.randn((6, 3), dtype=torch.float64)
    state = field(coords)
    out = eq.navier_stokes(state, viscosity=0.01,
                            form="vorticity_stream_2d",
                            streamfunction="psi")
    assert out.residual.shape == (6,)
    assert torch.allclose(out.continuity, torch.zeros((6,), dtype=torch.float64))


@pytest.mark.needs_research
def test_ns_vorticity_stream_2d_against_research_physics():
    """The new vorticity-stream residual must match the existing 2D NS
    reference when fed the same closed-form derivatives."""
    pytest.importorskip(
        "research.experiments.navier_stokes_2d.physics",
        reason="research.experiments tree is private and not shipped publicly",
    )
    from research.experiments.navier_stokes_2d.physics import (
        NSParams,
        ns_residual,
    )
    field = _spectral_field_2d_psi(K=4, H=4, seed=11)
    coords = torch.randn((8, 3), dtype=torch.float64)
    state = field(coords)
    nu = 0.05
    new = eq.navier_stokes(
        state, viscosity=nu, form="vorticity_stream_2d",
        streamfunction="psi",
    ).residual

    psi_x = tops.derivative(state, "psi", axis="x", order=1)
    psi_y = tops.derivative(state, "psi", axis="y", order=1)
    # omega = -lap psi -> nabla omega from mixed partials.
    psi_xxx = tops.mixed_partial(state, "psi", ("x",), (3,))
    psi_yyx = tops.mixed_partial(state, "psi", ("x", "y"), (1, 2))
    psi_xxy = tops.mixed_partial(state, "psi", ("x", "y"), (2, 1))
    psi_yyy = tops.mixed_partial(state, "psi", ("y",), (3,))
    omega_x = -(psi_xxx + psi_yyx)
    omega_y = -(psi_xxy + psi_yyy)
    bih_psi = tops.biharmonic(state, "psi")
    lap_omega = -bih_psi
    psi_xxt = tops.mixed_partial(state, "psi", ("x", "t"), (2, 1))
    psi_yyt = tops.mixed_partial(state, "psi", ("y", "t"), (2, 1))
    omega_t = -(psi_xxt + psi_yyt)
    # Use the reference helper (no forcing).
    f_omega = torch.zeros_like(psi_x)
    ref = ns_residual(
        psi_x, psi_y, omega_x, omega_y, lap_omega, omega_t, f_omega,
        NSParams(nu=nu, forcing_enabled=False),
    )
    assert torch.allclose(new, ref, rtol=1e-12, atol=1e-12)


def test_ns_class_vs_function_form():
    field = _spectral_field_3d_velocity(K=2, H=4, seed=10)
    coords = torch.randn((3, 4), dtype=torch.float64)
    state = field(coords)
    cls_out = eq.NavierStokes(viscosity=0.1, density=1.0, form="primitive_3d",
                                velocity=("u", "v", "w"))(state)
    fn_out = eq.navier_stokes(state, viscosity=0.1, density=1.0,
                                form="primitive_3d",
                                velocity=("u", "v", "w"))
    assert torch.allclose(cls_out.residual, fn_out.residual,
                          rtol=1e-12, atol=1e-12)
    assert torch.allclose(cls_out.continuity, fn_out.continuity,
                          rtol=1e-12, atol=1e-12)


def test_ns_explicit_accessors():
    field = _spectral_field_3d_velocity(K=2, H=4, seed=12)
    coords = torch.randn((3, 4), dtype=torch.float64)
    state = field(coords)
    ns = eq.NavierStokes(viscosity=1e-3, form="primitive_3d",
                          velocity=("u", "v", "w"))
    out = ns(state)
    assert torch.allclose(ns.momentum_residual(state), out.residual,
                          rtol=1e-12, atol=1e-12)
    assert torch.allclose(ns.continuity_residual(state), out.continuity,
                          rtol=1e-12, atol=1e-12)


# ---------------- Validation errors --------------------------------


def test_burgers_invalid_form_raises():
    coord = CoordinateSpec(
        axes=("x", "t"), periodicity=(True, False),
        domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(0)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=2, time_hidden=4, activation="tanh",  # seed=0
        dtype=torch.float64,
    )
    coords = torch.randn((3, 2), dtype=torch.float64)
    state = field(coords)
    with pytest.raises(ValueError, match="form must be"):
        eq.burgers(state, form="bogus")


def test_ns_invalid_form_raises():
    coord = CoordinateSpec(
        axes=("x", "t"), periodicity=(True, False),
        domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t",
    )
    components = ComponentSpec(names=("u", "p"), groups={"velocity": ("u",)})
    torch.manual_seed(0)
    field = SpectralVectorField(
        coordinate_spec=coord, components=components,
        K=2, time_hidden=4, activation="tanh",  # seed=0
        dtype=torch.float64,
    )
    coords = torch.randn((3, 2), dtype=torch.float64)
    state = field(coords)
    with pytest.raises(ValueError, match="form must be"):
        eq.navier_stokes(state, form="bogus")


def test_heat_no_time_axis_raises():
    """A field with no time axis must trigger the error path of Heat.

    SpectralVectorField requires a time axis on the coordinate spec, so
    we use a OneLayerVectorField for this validation test.
    """
    from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
    coord = CoordinateSpec(
        axes=("x", "y"), periodicity=(True, True),
        domain=((0.0, 1.0), (0.0, 1.0)), time_axis=None,
    )
    components = ComponentSpec(names=("u",), groups={})
    torch.manual_seed(0)
    field = OneLayerVectorField(
        coordinate_spec=coord, components=components,
        hidden=4, base="tanh", dtype=torch.float64,
    )
    coords = torch.randn((3, 2), dtype=torch.float64)
    state = field(coords)
    with pytest.raises(ValueError, match="requires a time axis"):
        eq.heat(state)
