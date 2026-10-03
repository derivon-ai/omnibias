# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepONet trunk-jet seam (JAX): exactness, caching, jit/grad."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops
from omnibias.pinn.operator.jax import make_deeponet
from omnibias.pinn.operator.jax.deeponet import TRUNK_JET_CACHE_KEY

jax.config.update("jax_enable_x64", True)

TOL = 1e-12


@pytest.fixture
def specs():
    return CoordinateSpec(("x", "t")), ComponentSpec(("u",))


def _operator(specs, *, trunk_width: int = 8, jet_order: int = 3, seed: int = 0):
    cs, comps = specs
    return make_deeponet(
        coordinate_spec=cs,
        components=comps,
        n_sensors=16,
        trunk_width=trunk_width,
        trunk_hidden=12,
        trunk_depth=2,
        branch_hidden=12,
        branch_depth=2,
        base="tanh",
        jet_order=jet_order,
        seed=seed,
    )


def test_value_matches_manual_contraction(specs) -> None:
    op = _operator(specs)
    key = jax.random.PRNGKey(0)
    sensors = jax.random.normal(key, (3, 16), dtype=jnp.float64)
    field = op.condition(sensors)
    coords = jax.random.normal(jax.random.PRNGKey(1), (3, 2), dtype=jnp.float64)
    got = field.forward_values(coords)
    trunk = op.trunk.value(coords)
    want = jnp.einsum("bp,bcp->bc", trunk, field.coeffs) + field.bias
    assert float(jnp.max(jnp.abs(got - want))) < TOL


def test_closed_form_derivatives_match_jax_grad(specs) -> None:
    op = _operator(specs, jet_order=3)
    sensors = jax.random.normal(jax.random.PRNGKey(2), (1, 16), dtype=jnp.float64)
    field = op.condition(sensors)
    coords = jax.random.normal(jax.random.PRNGKey(3), (5, 2), dtype=jnp.float64)
    state = field(coords)

    def value_at(c: jnp.ndarray) -> jnp.ndarray:
        return field.forward_values(c[None, :])[0, 0]

    for order in (1, 2, 3):
        for axis in (0, 1):
            closed = ops.derivative(state, "u", axis=axis, order=order)
            ad = []
            for i in range(coords.shape[0]):
                g: Any = value_at
                for _ in range(order):
                    g = jax.jacfwd(g)
                nested = g(coords[i])
                cur = nested
                for _ in range(order):
                    cur = cur[axis]
                ad.append(cur)
            ad_arr = jnp.stack(ad)
            assert float(jnp.max(jnp.abs(closed - ad_arr))) < 1e-9, (
                f"order={order} axis={axis}"
            )


def test_residual_costs_exactly_one_trunk_jet(specs) -> None:
    op = _operator(specs, jet_order=2)
    field = op.condition(jax.random.normal(jax.random.PRNGKey(4), (1, 16)))
    coords = jax.random.normal(jax.random.PRNGKey(5), (7, 2))
    state = field(coords)
    ops.value(state, "u")
    assert TRUNK_JET_CACHE_KEY not in state.extra or not state.extra[TRUNK_JET_CACHE_KEY]
    ops.gradient(state, "u")
    cached = state.extra[TRUNK_JET_CACHE_KEY]
    assert sorted(cached) == [2]
    ops.laplacian(state, "u")
    assert sorted(state.extra[TRUNK_JET_CACHE_KEY]) == [2]


def test_laplacian_fast_lane_leaves_trunk_jet_cache_unchanged(specs) -> None:
    from omnibias.pinn.jax.fields.jet_mlp import FAST_LANE_CACHE_KEY

    op = _operator(specs, jet_order=2)
    field = op.condition(jax.random.normal(jax.random.PRNGKey(40), (1, 16)))
    state = field(jax.random.normal(jax.random.PRNGKey(41), (5, 2)))
    ops.laplacian(state, "u")
    assert TRUNK_JET_CACHE_KEY not in state.extra or not state.extra[TRUNK_JET_CACHE_KEY]
    assert FAST_LANE_CACHE_KEY in state.extra


def test_laplacian_matches_manual_branch_contraction(specs) -> None:
    """The deep-field Laplacian fast lane must not silently read the raw trunk.

    ``G(u)(y) = bias + sum_k coeffs_k * t_k(y)`` with ``coeffs``/``bias`` constant
    in ``y``, so ``Delta_y G(u)(y) = sum_k coeffs_k * Delta_y t_k(y)``. The trunk
    Laplacian is taken independently via ``jax.hessian`` on the trunk's own
    ``value``, never touching the field's ``polylaplacian`` -- an oracle the fast
    lane cannot share a bug with. This pins
    :class:`~omnibias.pinn.operator.jax.deeponet.DeepONetField` against the
    generic ``_JetFieldOps.polylaplacian`` fast lane, which reads
    ``net._layer_specs()`` directly and would otherwise compute the Laplacian of
    a trunk basis column instead of the branch-contracted operator output.
    """
    op = _operator(specs, trunk_width=5, jet_order=2)
    sensors = jax.random.normal(jax.random.PRNGKey(20), (4, 16), dtype=jnp.float64)
    field = op.condition(sensors)
    coords = jax.random.normal(jax.random.PRNGKey(21), (4, 2), dtype=jnp.float64)
    state = field(coords)
    got = ops.laplacian(state, "u")

    def trunk_laplacian(y: Array) -> Array:
        # ``t`` is the (non-spatial) time axis in this fixture's CoordinateSpec,
        # so the PINN Laplacian sums only over the spatial axis ``x`` (index 0).
        hess = jax.hessian(lambda yi: op.trunk.value(yi[None, :])[0])(y)
        return hess[:, 0, 0]

    trunk_lap = jnp.stack([trunk_laplacian(coords[i]) for i in range(coords.shape[0])])
    want = jnp.einsum("bp,bcp->bc", trunk_lap, field.coeffs)[:, 0]
    assert float(jnp.max(jnp.abs(got - want))) < 1e-9
    # Regression guard: the buggy fast lane would instead equal a *raw* trunk
    # basis column's Laplacian, which disagrees with the correct contraction
    # whenever the branch coefficients are not a one-hot selector (true here).
    raw_trunk_col0 = jnp.stack(
        [trunk_laplacian(coords[i])[0] for i in range(coords.shape[0])]
    )
    assert float(jnp.max(jnp.abs(got - raw_trunk_col0))) > 1e-6


def test_polylaplacian_matches_manual_branch_contraction(specs) -> None:
    """``Delta^2`` (biharmonic) on the DeepONet field, same oracle as above."""
    op = _operator(specs, trunk_width=5, jet_order=4)
    sensors = jax.random.normal(jax.random.PRNGKey(22), (3, 16), dtype=jnp.float64)
    field = op.condition(sensors)
    coords = jax.random.normal(jax.random.PRNGKey(23), (3, 2), dtype=jnp.float64)
    state = field(coords)
    got = ops.biharmonic(state, "u")

    def trunk_bilaplacian(y: Array) -> Array:
        # ``t`` is the (non-spatial) time axis, so ``Delta^2`` is the pure
        # fourth ``x``-derivative (axis 0) of each trunk basis function.
        def lap(yi: Array) -> Array:
            hess = jax.hessian(lambda z: op.trunk.value(z[None, :])[0])(yi)
            return hess[:, 0, 0]

        out = []
        for k in range(op.trunk.out_dim):
            hess_k = jax.hessian(lambda z: lap(z)[k])(y)
            out.append(hess_k[0, 0])
        return jnp.stack(out)

    trunk_bilap = jnp.stack(
        [trunk_bilaplacian(coords[i]) for i in range(coords.shape[0])]
    )
    want = jnp.einsum("bp,bcp->bc", trunk_bilap, field.coeffs)[:, 0]
    assert float(jnp.max(jnp.abs(got - want))) < 1e-6


def test_shared_grid_and_aligned_paths_agree(specs) -> None:
    op = _operator(specs)
    sensors = jax.random.normal(jax.random.PRNGKey(6), (4, 16))
    field = op.condition(sensors)
    query = jax.random.normal(jax.random.PRNGKey(7), (6, 2))
    state_shared = field.on_grid(query)
    u_shared = ops.value(state_shared, "u").reshape(4, 6)
    for f in range(4):
        one = op.condition(sensors[f : f + 1])
        st = one(query)
        assert float(jnp.max(jnp.abs(ops.value(st, "u") - u_shared[f]))) < TOL


def test_shared_grid_trunk_jet_computed_once(specs) -> None:
    op = _operator(specs, jet_order=2)
    field = op.condition(jax.random.normal(jax.random.PRNGKey(50), (5, 16)))
    query = jax.random.normal(jax.random.PRNGKey(51), (8, 2))
    state = field.on_grid(query)
    ops.gradient(state, "u")
    cached = state.extra[TRUNK_JET_CACHE_KEY]
    assert sorted(cached) == [2]
    assert cached[2].shape[0] == 8
    ops.laplacian(state, "u")
    assert cached[2].shape[0] == 8


def test_fastpath_refusal_on_order_cap(specs) -> None:
    """arctan caps at order 2; a jet_order=3 DeepONet must refuse at construction."""
    cs, comps = specs
    with pytest.raises(ValueError, match="does not support order 3"):
        make_deeponet(
            coordinate_spec=cs,
            components=comps,
            n_sensors=8,
            trunk_width=4,
            base="arctan",
            jet_order=3,
        )


def test_fastpath_refusal_on_missing_kernel(specs) -> None:
    """An activation with fastpath=None is rejected at construction."""
    import dataclasses

    from omnibias.jax.activations import get_activation

    cs, comps = specs
    nofp = dataclasses.replace(get_activation("tanh"), name="tanh_nofp", fastpath=None)
    with pytest.raises(ValueError, match="closed-form derivative"):
        make_deeponet(
            coordinate_spec=cs,
            components=comps,
            n_sensors=8,
            trunk_width=4,
            base=nofp,
            jet_order=2,
        )


def test_jit_and_grad_trace(specs) -> None:
    op = _operator(specs, jet_order=2)

    def loss(op_in: Any, sensors: Array, coords: Array) -> Array:
        field = op_in.condition(sensors)
        state = field(coords)
        u = ops.value(state, "u")
        ux = ops.derivative(state, "u", axis=0, order=1)
        return jnp.mean(u**2 + ux**2)

    sensors = jax.random.normal(jax.random.PRNGKey(8), (2, 16))
    coords = jax.random.normal(jax.random.PRNGKey(9), (2, 2))
    jitted = jax.jit(loss)
    val = jitted(op, sensors, coords)
    assert np.isfinite(float(val))
    grads = jax.grad(loss)(op, sensors, coords)
    # Gradients flow to trunk weights.
    assert grads.trunk.weights[0].shape == op.trunk.weights[0].shape
    assert float(jnp.max(jnp.abs(grads.trunk.weights[0]))) > 0.0
