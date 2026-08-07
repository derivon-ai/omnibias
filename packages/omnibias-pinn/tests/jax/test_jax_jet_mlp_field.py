# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deep / Fourier-feature PINN fields (JAX): exactness, caching, transformations.

Mirrors :mod:`tests.torch.test_torch_jet_mlp_field` so cross-backend parity has a
like-for-like test surface, and additionally pins the JAX-specific contract: the
field is a pytree, so ``jax.grad`` / ``jax.jit`` see through it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops
from omnibias.pinn.jax.fields import (
    make_fourier_feature_vector_field,
    make_jet_mlp_vector_field,
    make_one_layer_vector_field,
    make_siren_vector_field,
)
from omnibias.pinn.jax.fields.jet_mlp import JET_CACHE_KEY, JetMLPVectorField

TOL = 1e-12


@pytest.fixture
def coords():
    rng = np.random.default_rng(7)
    return jnp.asarray(rng.normal(size=(6, 2)).astype(np.float64))


@pytest.fixture
def specs():
    return CoordinateSpec(("x", "y")), ComponentSpec(("u", "v"), groups={"vel": ("u", "v")})


def _scalar(field, name: str):
    ci = field.components.index(name)
    return lambda xi: field.net.value(xi[None])[0, ci]


# -- exactness vs jax autodiff ------------------------------------------------ #


@pytest.mark.parametrize("base", ["tanh", "sigmoid", "softplus", "sin"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_gradient_and_hessian_match_autodiff(coords, specs, base, depth):
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=depth, base=base, seed=1,
    )
    state = field(coords)
    f = _scalar(field, "u")
    grad_ref = jax.vmap(jax.grad(f))(coords)
    hess_ref = jax.vmap(jax.hessian(f))(coords)
    assert np.allclose(ops.gradient(state, "u"), grad_ref, rtol=TOL, atol=TOL)
    assert np.allclose(ops.hessian(state, "u"), hess_ref, rtol=TOL, atol=TOL)
    assert np.allclose(
        ops.laplacian(state, "u"), jnp.trace(hess_ref, axis1=1, axis2=2),
        rtol=TOL, atol=TOL,
    )


def test_third_order_partials_match_autodiff(coords, specs):
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, jet_order=3, seed=2,
    )
    state = field(coords)
    f = _scalar(field, "u")
    third = jax.vmap(jax.jacfwd(jax.hessian(f)))(coords)  # (B, 2, 2, 2)
    assert np.allclose(
        ops.derivative(state, "u", axis="x", order=3), third[:, 0, 0, 0],
        rtol=TOL, atol=TOL,
    )
    assert np.allclose(
        ops.mixed_partial(state, "u", ("x", "y"), (2, 1)), third[:, 0, 0, 1],
        rtol=TOL, atol=TOL,
    )
    assert np.allclose(
        ops.mixed_partial(state, "u", ("x", "y"), (1, 2)), third[:, 0, 1, 1],
        rtol=TOL, atol=TOL,
    )


def test_polylaplacian_and_biharmonic(coords, specs):
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=5, depth=2, jet_order=6, seed=3,
    )
    state = field(coords)
    f = _scalar(field, "u")

    def lap_op(g):
        return lambda xi: jnp.trace(jax.hessian(g)(xi))

    ref = f
    for k in (1, 2, 3):
        ref = lap_op(ref)
        got = ops.polylaplacian(state, "u", k=k)
        assert np.allclose(got, jax.vmap(ref)(coords), rtol=1e-11, atol=1e-11)
    assert np.allclose(
        ops.biharmonic(state, "u"), ops.polylaplacian(state, "u", k=2),
        rtol=TOL, atol=TOL,
    )


def test_depth1_reproduces_one_layer_field(coords, specs):
    """A depth-1 jet field *is* ``OneLayerVectorField``, so every op must agree.

    They agree to float64 round-off, not bit-for-bit: the multi-layer Faa di Bruno
    recursion and the single-layer ``sigma``-tower contraction reach the same value
    by different reductions.
    """
    cs, comps = specs
    one = make_one_layer_vector_field(
        coordinate_spec=cs, components=comps, hidden=9, base="tanh", seed=0,
    )
    deep = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=9, depth=1, base="tanh",
        jet_order=4, seed=0,
    )
    # Re-parameterise the deep field with the one-layer field's weights.
    net = deep.net.__class__(
        weights=(one.W, one.c),
        biases=(one.beta, one.b),
        spec=deep.net.spec,
        in_dim=deep.net.in_dim,
        out_dim=deep.net.out_dim,
    )
    deep = JetMLPVectorField(
        coordinate_spec=cs, components=comps, net=net, jet_order=4,
    )

    s_one, s_deep = one(coords), deep(coords)
    for op in (ops.value, ops.gradient, ops.laplacian, ops.hessian, ops.biharmonic):
        assert np.allclose(op(s_one, "u"), op(s_deep, "u"), rtol=TOL, atol=TOL), (
            f"{op.__name__} disagrees between one-layer and depth-1 jet field"
        )
    for order in (1, 2, 3, 4):
        assert np.allclose(
            ops.derivative(s_one, "u", axis="x", order=order),
            ops.derivative(s_deep, "u", axis="x", order=order),
            rtol=TOL, atol=TOL,
        )
    assert np.allclose(
        ops.jacobian(s_one, ("u", "v")), ops.jacobian(s_deep, ("u", "v")),
        rtol=TOL, atol=TOL,
    )


# -- the jet cache ------------------------------------------------------------ #


def test_order2_residual_triggers_exactly_one_jet(coords, specs, monkeypatch):
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, jet_order=2, seed=2,
    )
    calls: list[int] = []
    original = JetMLPVectorField._compute_hidden_jet

    def counting(self, c, order):
        calls.append(order)
        return original(self, c, order)

    monkeypatch.setattr(JetMLPVectorField, "_compute_hidden_jet", counting, raising=False)

    state = field(coords)
    ops.value(state, "u")
    ops.gradient(state, "u")
    ops.laplacian(state, "u")
    ops.hessian(state, "u")
    ops.divergence(state, ("u", "v"))
    ops.mixed_partial(state, "u", ("x", "y"), (1, 1))
    ops.derivative(state, "v", axis="x", order=2)
    assert calls == [2], f"expected one order-2 jet, got {calls}"


def test_cached_higher_order_jet_serves_lower_orders(coords, specs):
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=5, depth=1, jet_order=3, seed=4,
    )
    state = field(coords)
    ops.laplacian(state, "u")
    assert sorted(state.extra[JET_CACHE_KEY]) == [3]
    ops.gradient(state, "u")
    assert sorted(state.extra[JET_CACHE_KEY]) == [3]


def test_value_only_never_pays_for_a_jet(coords, specs):
    """A boundary-condition loss must not populate the hidden-jet cache."""
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=5, depth=2, jet_order=2, seed=4,
    )
    state = field(coords)
    ops.value(state, "u")
    ops.value(state, "v")
    assert JET_CACHE_KEY not in state.extra or not state.extra[JET_CACHE_KEY]
    ops.gradient(state, "u")
    assert sorted(state.extra[JET_CACHE_KEY]) == [2]


# -- JAX transformations ------------------------------------------------------ #


def test_field_is_a_pytree_and_grad_flows(coords, specs):
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, seed=0,
    )

    def loss(f):
        return jnp.sum(ops.laplacian(f(coords), "u") ** 2)

    grads = jax.grad(loss)(field)
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(field)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and any(float(jnp.abs(g).max()) > 0 for g in leaves)


def test_jit_matches_eager(coords, specs):
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, seed=0,
    )

    def residual(f, x):
        return ops.laplacian(f(x), "u")

    assert np.allclose(
        jax.jit(residual)(field, coords), residual(field, coords), rtol=TOL, atol=TOL,
    )


# -- Fourier-feature / SIREN variants ----------------------------------------- #


def test_fourier_feature_field_bands(coords, specs):
    cs, comps = specs
    field = make_fourier_feature_vector_field(
        coordinate_spec=cs, components=comps, num_features=4, hidden=6, depth=2,
        frequency_scale=(0.5, 2.0, 8.0), seed=0,
    )
    assert field.scales == (0.5, 2.0, 8.0)
    assert field.feature_dim == 2 * 4 * 3
    state = field(coords)
    hess_ref = jax.vmap(jax.hessian(_scalar(field, "u")))(coords)
    expected = jnp.trace(hess_ref, axis1=1, axis2=2)
    scale = float(jnp.abs(expected).max())
    assert np.allclose(
        ops.laplacian(state, "u"), expected, rtol=1e-10, atol=1e-10 * scale,
    )


def test_siren_field_high_order(coords, specs):
    cs, comps = specs
    field = make_siren_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, omega_0=5.0,
        jet_order=3, seed=0,
    )
    state = field(coords)
    third = jax.vmap(jax.jacfwd(jax.hessian(_scalar(field, "u"))))(coords)
    assert np.allclose(
        ops.derivative(state, "u", axis="x", order=3), third[:, 0, 0, 0],
        rtol=1e-11, atol=1e-11,
    )


def test_dispatch_tag_and_validation(specs):
    cs, comps = specs
    field = make_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=4, depth=1,
    )
    assert field._omnibias_dispatch == "jet_mlp"
    assert "JetMLPVectorField" in repr(field)
    with pytest.raises(ValueError, match="jet_order must be >= 1"):
        make_jet_mlp_vector_field(
            coordinate_spec=cs, components=comps, hidden=4, depth=1, jet_order=0,
        )
    # arctan's fast path caps at order 2, so a field promising order-3 residuals
    # must be rejected at construction, not deep in the kernel.
    with pytest.raises(ValueError, match="does not support order 3"):
        make_jet_mlp_vector_field(
            coordinate_spec=cs, components=comps, hidden=4, depth=1,
            base="arctan", jet_order=3,
        )
