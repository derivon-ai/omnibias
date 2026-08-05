# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the JAX :class:`ChebyshevVectorField`."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.fields.chebyshev import (
    ChebyshevVectorField,
    make_chebyshev_vector_field,
)


def _make_field_2d(K: int = 5, time_hidden: int = 6) -> ChebyshevVectorField:
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0), (-2.0, 2.0)),
    )
    mspec = ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")})
    field = make_chebyshev_vector_field(
        coordinate_spec=cspec, components=mspec, K=K,
        time_hidden=time_hidden, time_depth=1,
        activation="tanh", seed=0,
    )
    rng = np.random.default_rng(123)
    leaves, td = jax.tree_util.tree_flatten(field)
    new_leaves = list(leaves)
    new_leaves[0] = jnp.asarray(rng.normal(size=(time_hidden, 1)))
    new_leaves[1] = jnp.asarray(rng.normal(scale=0.1, size=(time_hidden,)))
    out_dim = field.C * field._out_per_component
    new_leaves[-3] = jnp.asarray(
        rng.normal(scale=0.1, size=(out_dim, time_hidden))
    )
    new_leaves[-2] = jnp.asarray(
        rng.normal(scale=0.1, size=(out_dim,))
    )
    return jax.tree_util.tree_unflatten(td, new_leaves)


def _make_field_3d(K: int = 4, time_hidden: int = 4) -> ChebyshevVectorField:
    cspec = CoordinateSpec(
        axes=("t", "x", "y", "z"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    )
    mspec = ComponentSpec(
        ("u", "v", "w", "p"), groups={"velocity": ("u", "v", "w")},
    )
    field = make_chebyshev_vector_field(
        coordinate_spec=cspec, components=mspec, K=K,
        time_hidden=time_hidden, time_depth=1,
        activation="tanh", seed=1,
    )
    rng = np.random.default_rng(456)
    leaves, td = jax.tree_util.tree_flatten(field)
    new_leaves = list(leaves)
    new_leaves[0] = jnp.asarray(rng.normal(size=(time_hidden, 1)))
    new_leaves[1] = jnp.asarray(rng.normal(scale=0.1, size=(time_hidden,)))
    out_dim = field.C * field._out_per_component
    new_leaves[-3] = jnp.asarray(
        rng.normal(scale=0.1, size=(out_dim, time_hidden))
    )
    new_leaves[-2] = jnp.asarray(
        rng.normal(scale=0.1, size=(out_dim,))
    )
    return jax.tree_util.tree_unflatten(td, new_leaves)


def _coords_2d(B: int = 5):
    coords = np.zeros((B, 3), dtype=np.float64)
    coords[:, 0] = np.linspace(0.1, 0.9, B)
    coords[:, 1] = np.linspace(-0.8, 0.7, B)
    coords[:, 2] = np.linspace(-1.5, 1.7, B)
    return jnp.asarray(coords)


def _coords_3d(B: int = 5):
    coords = np.zeros((B, 4), dtype=np.float64)
    coords[:, 0] = np.linspace(0.1, 0.9, B)
    coords[:, 1] = np.linspace(-0.8, 0.7, B)
    coords[:, 2] = np.linspace(-0.6, 0.6, B)
    coords[:, 3] = np.linspace(-0.5, 0.4, B)
    return jnp.asarray(coords)


def _per_value_derivative(field, coords, name: str, axis: int):
    def f_one(c_one):
        return jops.value(field(c_one[None, :]), name)[0]
    return jax.vmap(jax.grad(f_one))(coords)[..., axis]


def _per_value_dn(field, coords, name: str, axes: tuple[int, ...]):
    def f_one(c_one):
        return jops.value(field(c_one[None, :]), name)[0]
    cur = f_one
    for a in axes:
        prev = cur
        cur = (lambda prev_fn=prev, ax=a:
               (lambda c: jax.grad(prev_fn)(c)[ax]))()
    return jax.vmap(cur)(coords)


def test_value_finite_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for n in ("u", "v", "p"):
        v = jops.value(state, n)
        assert v.shape == (coords.shape[0],)
        assert jnp.all(jnp.isfinite(v))


def test_d_dx_dy_dt_match_autograd_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for axis_name, axis_idx in (("t", 0), ("x", 1), ("y", 2)):
        closed = jops.derivative(state, "u", axis=axis_name)
        auto = _per_value_derivative(field, coords, "u", axis_idx)
        assert np.allclose(np.asarray(closed), np.asarray(auto),
                           rtol=1e-10, atol=1e-10), axis_name


def test_higher_order_partials_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for axis_name, axis_idx in (("x", 1), ("y", 2)):
        for order in (2, 3):
            closed = jops.derivative(state, "u", axis=axis_name, order=order)
            auto = _per_value_dn(field, coords, "u", (axis_idx,) * order)
            assert np.allclose(np.asarray(closed), np.asarray(auto),
                               rtol=1e-9, atol=1e-9), (axis_name, order)


def test_mixed_partials_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for axes_str, axes_idx in (
        (("x", "y"), (1, 2)),
        (("t", "x"), (0, 1)),
    ):
        closed = jops.mixed_partial(state, "u", axes_str, (1, 1))
        auto = _per_value_dn(field, coords, "u", axes_idx)
        assert np.allclose(np.asarray(closed), np.asarray(auto),
                           rtol=1e-10, atol=1e-10), axes_str


def test_laplacian_matches_d2x_plus_d2y_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    lap = jops.laplacian(state, "u")
    sum_d2 = (
        jops.derivative(state, "u", axis="x", order=2)
        + jops.derivative(state, "u", axis="y", order=2)
    )
    assert np.allclose(np.asarray(lap), np.asarray(sum_d2),
                       rtol=1e-12, atol=1e-12)


def test_biharmonic_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    bih = jops.biharmonic(state, "u")
    pl = jops.polylaplacian(state, "u", k=2)
    assert np.allclose(np.asarray(bih), np.asarray(pl),
                       rtol=1e-12, atol=1e-12)


def test_partials_3d():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    for n, axis_name, axis_idx in [
        ("u", "x", 1), ("v", "y", 2), ("w", "z", 3),
    ]:
        closed = jops.derivative(state, n, axis=axis_name)
        auto = _per_value_derivative(field, coords, n, axis_idx)
        assert np.allclose(np.asarray(closed), np.asarray(auto),
                           rtol=1e-10, atol=1e-10), (n, axis_name)


def test_laplacian_3d():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    lap = jops.laplacian(state, "u")
    sum_d2 = sum(
        jops.derivative(state, "u", axis=a, order=2)
        for a in ("x", "y", "z")
    )
    assert np.allclose(np.asarray(lap), np.asarray(sum_d2),
                       rtol=1e-12, atol=1e-12)


def test_3d_curl_through_velocity_view():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    curl = state.velocity.curl
    assert curl.shape == (coords.shape[0], 3)


def test_default_domain_minus1_to_1():
    cspec = CoordinateSpec(axes=("t", "x"), time_axis="t")
    mspec = ComponentSpec(("u",))
    field = make_chebyshev_vector_field(
        coordinate_spec=cspec, components=mspec, K=4,
        time_hidden=4, time_depth=1, activation="tanh", seed=0,
    )
    assert field.domain == ((-1.0, 1.0),)


def test_state_view_dsl_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    assert jnp.array_equal(state.u.value, jops.value(state, "u"))
    assert jnp.array_equal(state.u.dx, jops.derivative(state, "u", axis="x"))
    assert jnp.array_equal(state.u.lap, jops.laplacian(state, "u"))


def test_repr():
    field = _make_field_2d()
    s = repr(field)
    assert "ChebyshevVectorField" in s
    assert "K=5" in s
    assert "domain=" in s
