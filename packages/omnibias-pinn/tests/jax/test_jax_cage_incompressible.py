# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the JAX incompressible cage layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.cage import (
    coulomb_gauge_loss,
    helmholtz_gauge_loss,
    make_helmholtz_projection_field,
    make_streamfunction_field,
    make_vector_potential_field,
)
from omnibias.pinn.jax.fields.spectral import make_spectral_vector_field


def _randomise_spectral(field, rng):
    leaves, td = jax.tree_util.tree_flatten(field)
    new_leaves = list(leaves)
    new_leaves[0] = jnp.asarray(rng.normal(size=field.W_t.shape))
    new_leaves[1] = jnp.asarray(rng.normal(scale=0.1, size=field.beta_t.shape))
    new_leaves[-2] = jnp.asarray(rng.normal(scale=0.1, size=field.V.shape))
    new_leaves[-1] = jnp.asarray(rng.normal(scale=0.1, size=field.b_t.shape))
    return jax.tree_util.tree_unflatten(td, new_leaves)


def _make_2d_cage():
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("psi", "p"))
    base = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=6, time_depth=1, activation="tanh", seed=0,
    )
    base = _randomise_spectral(base, np.random.default_rng(123))
    return make_streamfunction_field(
        base=base, psi="psi",
        velocity_names=("u", "v"),
        passthrough_names=("p",),
        spatial_axes=("x", "y"),
    )


def _coords_2d(B: int = 8):
    coords = np.zeros((B, 3), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0, 1.0, B)
    coords[:, 1] = np.linspace(0.1, 1.5, B)
    coords[:, 2] = np.linspace(0.2, 1.7, B)
    return jnp.asarray(coords)


def test_streamfunction_div_zero_2d():
    cage = _make_2d_cage()
    state = cage(_coords_2d())
    div = state.velocity.div
    assert float(jnp.max(jnp.abs(div))) < 1e-13


def test_streamfunction_passthrough_p_2d():
    cage = _make_2d_cage()
    state = cage(_coords_2d())
    inner = state.extra["_cage_inner_state"]
    assert np.allclose(np.asarray(state.p.value), np.asarray(jops.value(inner, "p")))


def test_streamfunction_velocity_via_psi_2d():
    cage = _make_2d_cage()
    state = cage(_coords_2d())
    inner = state.extra["_cage_inner_state"]
    psi_dx = jops.derivative(inner, "psi", axis="x")
    psi_dy = jops.derivative(inner, "psi", axis="y")
    assert np.allclose(np.asarray(state.u.value), np.asarray(psi_dy))
    assert np.allclose(np.asarray(state.v.value), -np.asarray(psi_dx))


def test_streamfunction_laplacian_identity_2d():
    cage = _make_2d_cage()
    state = cage(_coords_2d())
    lap_u = state.u.lap
    sum_d2 = state.u.d("x", 2) + state.u.d("y", 2)
    assert np.allclose(
        np.asarray(lap_u), np.asarray(sum_d2),
        rtol=1e-12, atol=1e-12,
    )


def test_streamfunction_velocity_derivatives_match_via_psi_2d():
    cage = _make_2d_cage()
    state = cage(_coords_2d())
    inner = state.extra["_cage_inner_state"]
    du_dt = state.u.dt
    auto = jops.mixed_partial(inner, "psi", ("t", "y"), (1, 1))
    assert np.allclose(np.asarray(du_dt), np.asarray(auto))
    du_dxx = state.u.d("x", 2)
    auto = jops.mixed_partial(inner, "psi", ("x", "y"), (2, 1))
    assert np.allclose(np.asarray(du_dxx), np.asarray(auto))


# --------------- 3D vector potential -------------------------------


def _make_3d_cage():
    cspec = CoordinateSpec(
        axes=("t", "x", "y", "z"),
        periodicity=(False, True, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("A1", "A2", "A3", "p"))
    base = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=3, time_hidden=4, time_depth=1, activation="tanh", seed=1,
    )
    base = _randomise_spectral(base, np.random.default_rng(456))
    return make_vector_potential_field(
        base=base, A_components=("A1", "A2", "A3"),
        velocity_names=("u", "v", "w"),
        passthrough_names=("p",),
        spatial_axes=("x", "y", "z"),
    )


def _coords_3d(B: int = 6):
    coords = np.zeros((B, 4), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0, 1.0, B)
    coords[:, 1] = np.linspace(0.1, 1.5, B)
    coords[:, 2] = np.linspace(0.2, 1.7, B)
    coords[:, 3] = np.linspace(-0.5, 1.0, B)
    return jnp.asarray(coords)


def test_vector_potential_div_zero_3d():
    cage = _make_3d_cage()
    state = cage(_coords_3d())
    div = state.velocity.div
    assert float(jnp.max(jnp.abs(div))) < 1e-12


def test_vector_potential_curl_explicit_3d():
    cage = _make_3d_cage()
    state = cage(_coords_3d())
    inner = state.extra["_cage_inner_state"]
    expected_u = (
        jops.derivative(inner, "A3", axis="y")
        - jops.derivative(inner, "A2", axis="z")
    )
    expected_v = (
        jops.derivative(inner, "A1", axis="z")
        - jops.derivative(inner, "A3", axis="x")
    )
    expected_w = (
        jops.derivative(inner, "A2", axis="x")
        - jops.derivative(inner, "A1", axis="y")
    )
    assert np.allclose(np.asarray(state.u.value), np.asarray(expected_u))
    assert np.allclose(np.asarray(state.v.value), np.asarray(expected_v))
    assert np.allclose(np.asarray(state.w.value), np.asarray(expected_w))


def test_coulomb_gauge_loss_positive_3d():
    cage = _make_3d_cage()
    coords = _coords_3d()
    loss = coulomb_gauge_loss(cage, coords)
    assert float(loss) > 0
    assert jnp.isfinite(loss).all()


# --------------- Helmholtz ----------------------------------------


def _make_helmholtz_cage():
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("u_pred", "v_pred", "phi", "p"))
    base = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=6, time_depth=1, activation="tanh", seed=2,
    )
    base = _randomise_spectral(base, np.random.default_rng(789))
    return make_helmholtz_projection_field(
        base=base, u_pred_components=("u_pred", "v_pred"),
        phi="phi", velocity_names=("u", "v"),
        passthrough_names=("p",),
    )


def test_helmholtz_velocity_value_2d():
    cage = _make_helmholtz_cage()
    state = cage(_coords_2d())
    inner = state.extra["_cage_inner_state"]
    expected_u = (
        jops.value(inner, "u_pred") - jops.derivative(inner, "phi", axis="x")
    )
    expected_v = (
        jops.value(inner, "v_pred") - jops.derivative(inner, "phi", axis="y")
    )
    assert np.allclose(np.asarray(state.u.value), np.asarray(expected_u))
    assert np.allclose(np.asarray(state.v.value), np.asarray(expected_v))


def test_helmholtz_gauge_loss_positive():
    cage = _make_helmholtz_cage()
    loss = helmholtz_gauge_loss(cage, _coords_2d())
    assert float(loss) > 0


# --------------- DSL routing -----------------------------------------


def test_streamfunction_state_view_dsl():
    cage = _make_2d_cage()
    state = cage(_coords_2d())
    assert jnp.array_equal(state.u.value, jops.value(state, "u"))
    assert jnp.array_equal(state.u.dt, jops.derivative(state, "u", axis="t"))
    assert jnp.array_equal(state.u.lap, jops.laplacian(state, "u"))
    assert jnp.array_equal(
        state.velocity.div, jops.divergence(state, ("u", "v")),
    )
