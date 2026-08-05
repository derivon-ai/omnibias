# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the jax equation registry (mirrors torch tests)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.jax import equations as jeq  # noqa: E402
from omnibias.pinn.jax import ops as jops  # noqa: E402
from omnibias.pinn.jax.fields.spectral import (  # noqa: E402
    make_spectral_vector_field,
)


def _spectral_2d_psi(K: int, H: int, *, seed: int):
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("psi",), groups={})
    return make_spectral_vector_field(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=H, activation="tanh", seed=seed,
        dtype=jnp.float64,
    )


def _spectral_3d_velocity(K: int, H: int, *, seed: int):
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
    return make_spectral_vector_field(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=H, activation="tanh", seed=seed,
        dtype=jnp.float64,
    )


def _spectral_1d_u(K: int, H: int, *, seed: int):
    coord = CoordinateSpec(
        axes=("x", "t"),
        periodicity=(True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("u",), groups={})
    return make_spectral_vector_field(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=H, activation="tanh", seed=seed,
        dtype=jnp.float64,
    )


def _spectral_2d_uv(K: int, H: int, *, seed: int):
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(
        names=("u", "v"), groups={"velocity": ("u", "v")},
    )
    return make_spectral_vector_field(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=H, activation="tanh", seed=seed,
        dtype=jnp.float64,
    )


def _spectral_2d_uvp(K: int, H: int, *, seed: int):
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(
        names=("u", "v", "p"), groups={"velocity": ("u", "v")},
    )
    return make_spectral_vector_field(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=H, activation="tanh", seed=seed,
        dtype=jnp.float64,
    )


def _spectral_2d_c(K: int, H: int, *, seed: int):
    coord = CoordinateSpec(
        axes=("x", "y", "t"),
        periodicity=(True, True, False),
        domain=((0.0, 2.0 * math.pi), (0.0, 2.0 * math.pi), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("c",), groups={})
    return make_spectral_vector_field(
        coordinate_spec=coord, components=components,
        K=K, time_hidden=H, activation="tanh", seed=seed,
        dtype=jnp.float64,
    )


# ---------------- Heat ---------------------------------------------


def test_heat_residual_finite():
    field = _spectral_1d_u(K=4, H=4, seed=0)
    rng = np.random.default_rng(0)
    coords = jnp.asarray(rng.standard_normal((6, 2)))
    state = field(coords)
    out = jeq.heat(state, alpha=0.01)
    assert out.residual.shape == (6,)
    assert jnp.all(jnp.isfinite(out.residual))


def test_heat_class_and_function_match():
    field = _spectral_1d_u(K=3, H=4, seed=1)
    rng = np.random.default_rng(1)
    coords = jnp.asarray(rng.standard_normal((6, 2)))
    state = field(coords)
    cls_out = jeq.Heat(alpha=0.5)(state)
    fn_out = jeq.heat(state, alpha=0.5)
    assert jnp.allclose(cls_out.residual, fn_out.residual, rtol=1e-12, atol=1e-12)


# ---------------- Burgers ------------------------------------------


def test_burgers_scalar_finite():
    field = _spectral_1d_u(K=4, H=4, seed=2)
    rng = np.random.default_rng(2)
    coords = jnp.asarray(rng.standard_normal((6, 2)))
    state = field(coords)
    out = jeq.burgers(state, nu=0.05, form="scalar")
    assert out.residual.shape == (6,)
    assert jnp.all(jnp.isfinite(out.residual))


def test_burgers_vector_2d_shape():
    field = _spectral_2d_uv(K=3, H=4, seed=3)
    rng = np.random.default_rng(3)
    coords = jnp.asarray(rng.standard_normal((5, 3)))
    state = field(coords)
    out = jeq.burgers(state, nu=0.01, form="vector",
                      velocity=("u", "v"))
    assert out.residual.shape == (5, 2)
    assert jnp.all(jnp.isfinite(out.residual))


# ---------------- KS -----------------------------------------------


def test_ks_1d_residual_shape():
    field = _spectral_1d_u(K=6, H=4, seed=42)
    rng = np.random.default_rng(42)
    coords = jnp.asarray(rng.standard_normal((10, 2)))
    state = field(coords)
    out = jeq.kuramoto_sivashinsky(state, form="1d")
    assert out.residual.shape == (10,)
    assert jnp.all(jnp.isfinite(out.residual))


# ---------------- Cahn-Hilliard ------------------------------------


def test_cahn_hilliard_residual_shape_and_finite():
    field = _spectral_2d_c(K=3, H=4, seed=7)
    rng = np.random.default_rng(7)
    coords = jnp.asarray(rng.standard_normal((9, 3)))
    state = field(coords)
    out = jeq.cahn_hilliard(state, M=1.0, kappa=1e-3)
    assert out.residual.shape == (9,)
    assert jnp.all(jnp.isfinite(out.residual))


# ---------------- Biharmonic ---------------------------------------


def test_biharmonic_steady_residual():
    field = _spectral_2d_c(K=3, H=4, seed=4)
    rng = np.random.default_rng(4)
    coords = jnp.asarray(rng.standard_normal((5, 3)))
    state = field(coords)
    out = jeq.biharmonic(state, component="c", include_time=False)
    assert out.residual.shape == (5,)
    bih = jops.biharmonic(state, "c")
    assert jnp.allclose(out.residual, bih, rtol=1e-12, atol=1e-12)


# ---------------- Navier-Stokes ------------------------------------


def test_ns_primitive_2d_shapes():
    field = _spectral_2d_uvp(K=2, H=4, seed=5)
    rng = np.random.default_rng(5)
    coords = jnp.asarray(rng.standard_normal((4, 3)))
    state = field(coords)
    out = jeq.navier_stokes(state, viscosity=1e-3, form="primitive_2d",
                              velocity=("u", "v"))
    assert out.residual.shape == (4, 2)
    assert out.continuity.shape == (4,)
    assert jnp.all(jnp.isfinite(out.residual))
    assert jnp.all(jnp.isfinite(out.continuity))


def test_ns_primitive_3d_shapes():
    field = _spectral_3d_velocity(K=2, H=4, seed=8)
    rng = np.random.default_rng(8)
    coords = jnp.asarray(rng.standard_normal((4, 4)))
    state = field(coords)
    out = jeq.navier_stokes(state, viscosity=1e-3, form="primitive_3d",
                              velocity=("u", "v", "w"))
    assert out.residual.shape == (4, 3)
    assert out.continuity.shape == (4,)


def test_ns_primitive_3d_hard_incompressibility_zeros_continuity():
    field = _spectral_3d_velocity(K=2, H=4, seed=8)
    rng = np.random.default_rng(8)
    coords = jnp.asarray(rng.standard_normal((4, 4)))
    state = field(coords)
    out = jeq.navier_stokes(state, viscosity=1e-3, form="primitive_3d",
                              velocity=("u", "v", "w"),
                              incompressibility="hard")
    assert jnp.allclose(out.continuity, jnp.zeros((4,), dtype=jnp.float64))


def test_ns_vorticity_stream_2d_shape_and_continuity():
    field = _spectral_2d_psi(K=3, H=4, seed=9)
    rng = np.random.default_rng(9)
    coords = jnp.asarray(rng.standard_normal((6, 3)))
    state = field(coords)
    out = jeq.navier_stokes(state, viscosity=0.01,
                              form="vorticity_stream_2d",
                              streamfunction="psi")
    assert out.residual.shape == (6,)
    assert jnp.allclose(out.continuity, jnp.zeros((6,), dtype=jnp.float64))


def test_ns_class_vs_function_form():
    field = _spectral_3d_velocity(K=2, H=4, seed=10)
    rng = np.random.default_rng(10)
    coords = jnp.asarray(rng.standard_normal((3, 4)))
    state = field(coords)
    cls_out = jeq.NavierStokes(viscosity=0.1, density=1.0,
                                form="primitive_3d",
                                velocity=("u", "v", "w"))(state)
    fn_out = jeq.navier_stokes(state, viscosity=0.1, density=1.0,
                                form="primitive_3d",
                                velocity=("u", "v", "w"))
    assert jnp.allclose(cls_out.residual, fn_out.residual,
                        rtol=1e-12, atol=1e-12)
    assert jnp.allclose(cls_out.continuity, fn_out.continuity,
                        rtol=1e-12, atol=1e-12)


# ---------------- Validation errors --------------------------------


def test_burgers_invalid_form_raises():
    field = _spectral_1d_u(K=2, H=4, seed=0)
    rng = np.random.default_rng(0)
    coords = jnp.asarray(rng.standard_normal((3, 2)))
    state = field(coords)
    with pytest.raises(ValueError, match="form must be"):
        jeq.burgers(state, form="bogus")


def test_ns_invalid_form_raises():
    field = _spectral_1d_u(K=2, H=4, seed=0)
    rng = np.random.default_rng(0)
    coords = jnp.asarray(rng.standard_normal((3, 2)))
    state = field(coords)
    with pytest.raises(ValueError, match="form must be"):
        jeq.navier_stokes(state, form="bogus")
