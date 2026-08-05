# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the JAX :class:`SpectralVectorField`.

Mirror of the torch test suite. Uses :func:`jax.grad` (sum-aggregation
trick) to compare per-axis derivatives against the gold reference.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.fields.spectral import (
    SpectralVectorField,
    make_spectral_vector_field,
)


def _make_field_2d(K: int = 4, time_hidden: int = 6) -> SpectralVectorField:
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")})
    field = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec, K=K,
        time_hidden=time_hidden, time_depth=1,
        activation="tanh", seed=0,
    )
    rng = np.random.default_rng(123)
    leaves, td = jax.tree_util.tree_flatten(field)
    new_leaves = list(leaves)
    new_leaves[0] = jnp.asarray(rng.normal(size=(time_hidden, 1)))         # W_t
    new_leaves[1] = jnp.asarray(rng.normal(scale=0.1, size=(time_hidden,)))  # beta_t
    new_leaves[-2] = jnp.asarray(
        rng.normal(scale=0.1, size=(field.C * field._out_per_component, time_hidden))
    )
    new_leaves[-1] = jnp.asarray(
        rng.normal(scale=0.1, size=(field.C * field._out_per_component,))
    )
    return jax.tree_util.tree_unflatten(td, new_leaves)


def _make_field_3d(K: int = 3, time_hidden: int = 4) -> SpectralVectorField:
    cspec = CoordinateSpec(
        axes=("t", "x", "y", "z"),
        periodicity=(False, True, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(
        ("u", "v", "w", "p"), groups={"velocity": ("u", "v", "w")},
    )
    field = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec, K=K,
        time_hidden=time_hidden, time_depth=1,
        activation="tanh", seed=1,
    )
    rng = np.random.default_rng(456)
    leaves, td = jax.tree_util.tree_flatten(field)
    new_leaves = list(leaves)
    new_leaves[0] = jnp.asarray(rng.normal(size=(time_hidden, 1)))
    new_leaves[1] = jnp.asarray(rng.normal(scale=0.1, size=(time_hidden,)))
    new_leaves[-2] = jnp.asarray(
        rng.normal(scale=0.1, size=(field.C * field._out_per_component, time_hidden))
    )
    new_leaves[-1] = jnp.asarray(
        rng.normal(scale=0.1, size=(field.C * field._out_per_component,))
    )
    return jax.tree_util.tree_unflatten(td, new_leaves)


def _coords_2d(B: int = 5):
    coords = np.zeros((B, 3), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0, 1.0, B)
    coords[:, 1] = np.linspace(0.1, 1.5, B)
    coords[:, 2] = np.linspace(0.2, 1.7, B)
    return jnp.asarray(coords)


def _coords_3d(B: int = 5):
    coords = np.zeros((B, 4), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0, 1.0, B)
    coords[:, 1] = np.linspace(0.1, 1.5, B)
    coords[:, 2] = np.linspace(0.2, 1.7, B)
    coords[:, 3] = np.linspace(-0.3, 1.1, B)
    return jnp.asarray(coords)


def _per_value_derivative(field, coords, name: str, axis: int):
    """Per-batch first derivative: differentiate the per-coord scalar value
    with ``jax.grad``, then ``vmap`` over the batch."""
    def f_one(c_one):
        return jops.value(field(c_one[None, :]), name)[0]
    g_one = jax.grad(f_one)
    return jax.vmap(g_one)(coords)[..., axis]


def _per_value_dn(field, coords, name: str, axes: tuple[int, ...]):
    """Per-batch derivative along the ``axes`` chain via repeated grad-then-axis."""
    def f_one(c_one):
        return jops.value(field(c_one[None, :]), name)[0]
    cur = f_one
    for a in axes:
        prev = cur
        cur = (lambda prev_fn=prev, ax=a:
               (lambda c: jax.grad(prev_fn)(c)[ax]))()
    return jax.vmap(cur)(coords)


# -- 2D ----------------------------------------------------------------


def test_value_finite_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    for n in ("u", "v", "p"):
        v = jops.value(state, n)
        assert v.shape == (coords.shape[0],)
        assert jnp.all(jnp.isfinite(v))


def test_d_dt_dx_dy_match_autograd_2d():
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
            assert np.allclose(
                np.asarray(closed), np.asarray(auto),
                rtol=1e-9, atol=1e-9,
            ), (axis_name, order)


def test_mixed_partials_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    closed = jops.mixed_partial(state, "u", ("x", "y"), (1, 1))
    auto = _per_value_dn(field, coords, "u", (1, 2))
    assert np.allclose(np.asarray(closed), np.asarray(auto),
                       rtol=1e-10, atol=1e-10)
    closed = jops.mixed_partial(state, "u", ("t", "x"), (1, 1))
    auto = _per_value_dn(field, coords, "u", (0, 1))
    assert np.allclose(np.asarray(closed), np.asarray(auto),
                       rtol=1e-10, atol=1e-10)


def test_laplacian_matches_d2x_plus_d2y_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    lap = jops.laplacian(state, "u")
    sum_d2 = (
        jops.derivative(state, "u", axis="x", order=2)
        + jops.derivative(state, "u", axis="y", order=2)
    )
    assert np.allclose(np.asarray(lap), np.asarray(sum_d2), rtol=1e-12, atol=1e-12)


def test_biharmonic_and_polylaplacian_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    bih = jops.biharmonic(state, "u")
    pl1 = jops.polylaplacian(state, "u", k=2)
    assert np.allclose(np.asarray(bih), np.asarray(pl1), rtol=1e-12, atol=1e-12)
    state = field(_coords_2d())
    d4x = jops.derivative(state, "u", axis="x", order=4)
    d4y = jops.derivative(state, "u", axis="y", order=4)
    dxxyy = jops.mixed_partial(state, "u", ("x", "y"), (2, 2))
    expected = d4x + 2.0 * dxxyy + d4y
    assert np.allclose(np.asarray(bih), np.asarray(expected), rtol=1e-9, atol=1e-9)


def test_divergence_curl_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    div = jops.divergence(state, ("u", "v"))
    expected_div = (
        jops.derivative(state, "u", axis="x")
        + jops.derivative(state, "v", axis="y")
    )
    assert np.allclose(np.asarray(div), np.asarray(expected_div),
                       rtol=1e-12, atol=1e-12)
    curl_z = jops.curl(state, ("u", "v"))
    assert curl_z.shape == (coords.shape[0], 1)
    expected_curl = (
        jops.derivative(state, "v", axis="x")
        - jops.derivative(state, "u", axis="y")
    )
    assert np.allclose(np.asarray(curl_z[:, 0]), np.asarray(expected_curl),
                       rtol=1e-12, atol=1e-12)


# -- 3D ----------------------------------------------------------------


def test_value_finite_3d():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    for n in ("u", "v", "w", "p"):
        v = jops.value(state, n)
        assert v.shape == (coords.shape[0],)
        assert jnp.all(jnp.isfinite(v))


def test_partials_3d_match_autograd():
    field = _make_field_3d()
    coords = _coords_3d()
    state = field(coords)
    for n, axis_name, axis_idx in [
        ("u", "x", 1), ("v", "y", 2), ("w", "z", 3), ("u", "t", 0),
    ]:
        closed = jops.derivative(state, n, axis=axis_name)
        auto = _per_value_derivative(field, coords, n, axis_idx)
        assert np.allclose(np.asarray(closed), np.asarray(auto),
                           rtol=1e-10, atol=1e-10), (n, axis_name)


def test_laplacian_3d_matches_sum_of_d2():
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
    expected_x = (
        jops.derivative(state, "w", axis="y")
        - jops.derivative(state, "v", axis="z")
    )
    expected_y = (
        jops.derivative(state, "u", axis="z")
        - jops.derivative(state, "w", axis="x")
    )
    expected_z = (
        jops.derivative(state, "v", axis="x")
        - jops.derivative(state, "u", axis="y")
    )
    assert np.allclose(np.asarray(curl[:, 0]), np.asarray(expected_x),
                       rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(curl[:, 1]), np.asarray(expected_y),
                       rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(curl[:, 2]), np.asarray(expected_z),
                       rtol=1e-12, atol=1e-12)


# -- exact spatial periodicity ---------------------------------------


def test_periodicity_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    val_a = jops.value(state, "u")

    coords_b = coords.copy()
    coords_b = coords_b.at[:, 1].add(2.0 * math.pi)
    coords_b = coords_b.at[:, 2].add(-2.0 * math.pi)
    state_b = field(coords_b)
    val_b = jops.value(state_b, "u")
    assert np.allclose(np.asarray(val_a), np.asarray(val_b),
                       rtol=1e-11, atol=1e-11)


def test_state_view_dsl_2d():
    field = _make_field_2d()
    coords = _coords_2d()
    state = field(coords)
    assert jnp.array_equal(state.u.value, jops.value(state, "u"))
    assert jnp.array_equal(state.u.dt, jops.derivative(state, "u", axis="t"))
    assert jnp.array_equal(state.u.dx, jops.derivative(state, "u", axis="x"))
    assert jnp.array_equal(state.u.lap, jops.laplacian(state, "u"))
    assert jnp.array_equal(state.u.biharm, jops.biharmonic(state, "u"))
    assert jnp.array_equal(
        state.velocity.div, jops.divergence(state, ("u", "v")),
    )


def test_repr():
    field = _make_field_2d()
    s = repr(field)
    assert "SpectralVectorField" in s
    assert "K=4" in s


# -- deep temporal MLP (time_depth > 1): closed-form jet time derivatives ----


def _make_field_2d_deep(K: int = 3, time_hidden: int = 6, time_depth: int = 3):
    cspec = CoordinateSpec(
        axes=("t", "x", "y"), periodicity=(False, True, True), time_axis="t",
    )
    mspec = ComponentSpec(("u", "v"))
    field = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec, K=K,
        time_hidden=time_hidden, time_depth=time_depth,
        activation="tanh", seed=0,
    )
    rng = np.random.default_rng(2024)
    leaves, td = jax.tree_util.tree_flatten(field)
    new = list(leaves)
    n_inner = time_depth - 1
    out_dim = field.C * field._out_per_component
    new[0] = jnp.asarray(rng.normal(size=(time_hidden, 1)))                # W_t
    new[1] = jnp.asarray(rng.normal(scale=0.1, size=(time_hidden,)))       # beta_t
    for i in range(n_inner):
        new[2 + i] = jnp.asarray(
            rng.normal(scale=1.0 / math.sqrt(time_hidden),
                       size=(time_hidden, time_hidden))
        )
        new[2 + n_inner + i] = jnp.asarray(rng.normal(scale=0.1, size=(time_hidden,)))
    new[-2] = jnp.asarray(rng.normal(scale=0.1, size=(out_dim, time_hidden)))  # V
    new[-1] = jnp.asarray(rng.normal(scale=0.05, size=(out_dim,)))             # b_t
    return jax.tree_util.tree_unflatten(td, new)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_deep_time_head_time_derivs_match_autograd(order: int) -> None:
    """Deep head (``time_depth=3``): closed-form jet d^n/dt^n vs nested grad."""
    field = _make_field_2d_deep(time_depth=3)
    coords = _coords_2d()
    state = field(coords)
    closed = jops.derivative(state, "u", axis="t", order=order)
    auto = _per_value_dn(field, coords, "u", (0,) * order)
    assert np.allclose(np.asarray(closed), np.asarray(auto), rtol=1e-8, atol=1e-8)


def test_deep_time_head_mixed_matches_autograd() -> None:
    """Mixed ``d^3/dt^2 dx`` through the deep closed-form time head."""
    field = _make_field_2d_deep(time_depth=3)
    coords = _coords_2d()
    state = field(coords)
    closed = jops.mixed_partial(state, "u", ("x", "t"), (1, 2))
    auto = _per_value_dn(field, coords, "u", (1, 0, 0))  # d/dx, d/dt, d/dt
    assert np.allclose(np.asarray(closed), np.asarray(auto), rtol=1e-8, atol=1e-8)


def test_deep_time_head_jit() -> None:
    """The deep closed-form time derivative is ``jax.jit``-compatible."""
    field = _make_field_2d_deep(time_depth=3)
    coords = _coords_2d()
    jitted = jax.jit(
        lambda c: jops.derivative(field(c), "u", axis="t", order=2)
    )
    out = jitted(coords)
    ref = jops.derivative(field(coords), "u", axis="t", order=2)
    assert np.allclose(np.asarray(out), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_deep_time_head_param_grad_flows() -> None:
    """``jax.grad`` of a d/dt residual loss flows to the inner temporal layers."""
    field = _make_field_2d_deep(time_depth=3)
    coords = _coords_2d()

    def loss(f):
        dudt = jops.derivative(f(coords), "u", axis="t", order=1)
        return jnp.mean(dudt**2)

    grads = jax.grad(loss)(field)
    assert any(float(jnp.sum(jnp.abs(w))) > 0 for w in grads.inner_W)
    assert float(jnp.sum(jnp.abs(grads.W_t))) > 0
