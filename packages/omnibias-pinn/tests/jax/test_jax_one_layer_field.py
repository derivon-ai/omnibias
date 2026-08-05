# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the JAX :class:`OneLayerVectorField`.

Mirrors :mod:`tests.torch.test_one_layer_field` so cross-backend bit
parity has a like-for-like test surface.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops
from omnibias.pinn.jax.fields.one_layer import make_one_layer_vector_field


@pytest.fixture
def small_field():
    coord_spec = CoordinateSpec(("x", "y", "t"))
    comp_spec = ComponentSpec(
        ("u", "v", "p"),
        groups={"velocity": ("u", "v")},
    )
    return make_one_layer_vector_field(
        coordinate_spec=coord_spec,
        components=comp_spec,
        hidden=8,
        base="tanh",
        weight_init_scale=0.5,
        seed=0,
        dtype=jnp.float64,
    )


@pytest.fixture
def small_coords():
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.normal(size=(7, 3)).astype(np.float64))


def test_field_evaluate_returns_state(small_field, small_coords):
    state = small_field(small_coords)
    assert state.field is small_field
    assert state.components.names == ("u", "v", "p")
    assert state.coordinate_spec.axes == ("x", "y", "t")


def test_field_value_via_dsl(small_field, small_coords):
    state = small_field(small_coords)
    u_val = state.u.value
    assert u_val.shape == (7,)
    assert jnp.isfinite(u_val).all()


def test_first_partial_against_autograd(small_field, small_coords):
    state = small_field(small_coords)
    du_dx = state.u.dx
    du_dy = state.u.dy
    du_dt = state.u.dt

    def f_u(coords):
        s = small_field(coords)
        return s.u.value.sum()

    g = jax.grad(f_u)(small_coords)
    assert jnp.allclose(du_dx, g[:, 0], rtol=1e-10, atol=1e-12)
    assert jnp.allclose(du_dy, g[:, 1], rtol=1e-10, atol=1e-12)
    assert jnp.allclose(du_dt, g[:, 2], rtol=1e-10, atol=1e-12)


def test_laplacian_against_autograd(small_field, small_coords):
    state = small_field(small_coords)
    L = state.u.lap

    def f_u(coords):
        return small_field(coords).u.value.sum()

    # Build the spatial Laplacian via two autograds.
    L_ref = jnp.zeros(small_coords.shape[0], dtype=small_coords.dtype)
    for axis in (0, 1):
        def fa(coords, a=axis):
            s = small_field(coords)
            return s.u.d(a, order=1).sum()

        gi = jax.grad(fa)(small_coords)
        L_ref = L_ref + gi[:, axis]
    assert jnp.allclose(L, L_ref, rtol=1e-9, atol=1e-12)


def test_hessian_against_autograd(small_field, small_coords):
    state = small_field(small_coords)
    H = state.u.hess
    assert H.shape == (7, 3, 3)

    def f_u(coords):
        return small_field(coords).u.value.sum()

    rows = []
    for j in range(3):
        def gj_fn(coords, j=j):
            return jax.grad(f_u)(coords)[:, j].sum()
        rows.append(jax.grad(gj_fn)(small_coords))
    H_ref = jnp.stack(rows, axis=-1)
    assert jnp.allclose(H, H_ref, rtol=1e-9, atol=1e-12)
    assert jnp.allclose(H, jnp.swapaxes(H, -1, -2), rtol=1e-12, atol=1e-12)


def test_biharmonic_equals_polylaplacian_k_2(small_field, small_coords):
    state = small_field(small_coords)
    B = state.u.biharm
    P = ops.polylaplacian(state, "u", k=2)
    assert jnp.allclose(B, P, rtol=1e-15, atol=1e-15)


def test_polylaplacian_k_1_equals_laplacian(small_field, small_coords):
    state = small_field(small_coords)
    L = state.u.lap
    P = ops.polylaplacian(state, "u", k=1)
    assert jnp.allclose(L, P, rtol=1e-15, atol=1e-15)


def test_divergence_2d(small_field, small_coords):
    state = small_field(small_coords)
    div = ops.divergence(state, ("u", "v"))
    expected = state.u.dx + state.v.dy
    assert jnp.allclose(div, expected, rtol=1e-15, atol=1e-15)


def test_curl_2d_returns_scalar(small_field, small_coords):
    state = small_field(small_coords)
    c = ops.curl(state, ("u", "v"))
    assert c.shape == (7, 1)
    expected = (state.v.dx - state.u.dy)[..., None]
    assert jnp.allclose(c, expected, rtol=1e-15, atol=1e-15)


def test_strain_rate_symmetric(small_field, small_coords):
    state = small_field(small_coords)
    S = ops.strain_rate(state, ("u", "v"))
    assert jnp.allclose(S, jnp.swapaxes(S, -1, -2), rtol=1e-15, atol=1e-15)


def test_self_advection_2d(small_field, small_coords):
    state = small_field(small_coords)
    adv = state.velocity.advect()
    expected = jnp.stack([
        state.u.value * state.u.dx + state.v.value * state.u.dy,
        state.u.value * state.v.dx + state.v.value * state.v.dy,
    ], axis=-1)
    assert jnp.allclose(adv, expected, rtol=1e-12, atol=1e-12)


def test_material_derivative(small_field, small_coords):
    state = small_field(small_coords)
    Dt = state.velocity.material_derivative()
    expected = state.velocity.dt + state.velocity.advect()
    assert jnp.allclose(Dt, expected, rtol=1e-12, atol=1e-12)


def test_p_laplacian_p_2_equals_laplacian(small_field, small_coords):
    state = small_field(small_coords)
    L = state.u.lap
    P = ops.p_laplacian(state, "u", p=2.0)
    assert jnp.allclose(L, P, rtol=1e-12, atol=1e-12)


def test_did_you_mean_typo(small_field, small_coords):
    state = small_field(small_coords)
    with pytest.raises(AttributeError) as ei:
        _ = state.velocty
    assert "velocity" in str(ei.value)


def test_pytree_round_trip(small_field):
    leaves, treedef = jax.tree_util.tree_flatten(small_field)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, type(small_field))
    assert jnp.allclose(rebuilt.W, small_field.W, rtol=1e-15, atol=1e-15)
    assert rebuilt.components == small_field.components
