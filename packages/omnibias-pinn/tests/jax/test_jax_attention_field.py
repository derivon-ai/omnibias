# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The non-local attention PINN field (JAX).

Mirrors :mod:`tests.torch.test_torch_attention_field` so cross-backend parity has a
like-for-like surface, and additionally pins the JAX-specific contract: the memory
and the inverse temperature are pytree leaves, so ``jax.grad`` reaches them and
``jax.jit`` traces the whole softmax-jet construction.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import equations as jeq
from omnibias.pinn.jax import ops
from omnibias.pinn.jax.fields import (
    make_attention_vector_field,
    make_jet_mlp_vector_field,
)
from omnibias.pinn.jax.fields.jet_mlp import JET_CACHE_KEY


@pytest.fixture
def coords():
    rng = np.random.default_rng(11)
    return jnp.asarray(rng.normal(size=(6, 2)).astype(np.float64))


@pytest.fixture
def specs():
    return (
        CoordinateSpec(("x", "t")),
        ComponentSpec(("u", "v"), groups={"vel": ("u", "v")}),
    )


@pytest.fixture
def field(specs):
    cspec, mspec = specs
    return make_attention_vector_field(
        coordinate_spec=cspec,
        components=mspec,
        hidden=6,
        depth=2,
        memory=5,
        beta=1.4,
        jet_order=3,
        seed=5,
    )


def _scalar(field, name: str):
    ci = field.components.index(name)
    return lambda xi: field.net.value(xi[None])[0, ci]


# ---------------------- exactness of the coordinate story --------------------


@pytest.mark.parametrize("name", ["u", "v"])
def test_derivatives_match_autodiff(field, coords, name) -> None:
    state = field(coords)
    f = _scalar(field, name)
    grad_ref = jax.vmap(jax.grad(f))(coords)
    hess_ref = jax.vmap(jax.hessian(f))(coords)
    third_ref = jax.vmap(jax.jacfwd(jax.hessian(f)))(coords)
    scale = max(float(jnp.abs(third_ref).max()), 1.0)
    assert np.allclose(
        ops.gradient(state, name), grad_ref[:, :1], rtol=1e-11, atol=1e-11 * scale
    )
    assert np.allclose(
        ops.hessian(state, name), hess_ref, rtol=1e-11, atol=1e-11 * scale
    )
    assert np.allclose(
        ops.derivative(state, name, axis=0, order=3),
        third_ref[:, 0, 0, 0],
        rtol=1e-11,
        atol=1e-11 * scale,
    )
    assert np.allclose(
        ops.mixed_partial(state, name, (0, 1), (2, 1)),
        third_ref[:, 0, 0, 1],
        rtol=1e-11,
        atol=1e-11 * scale,
    )


def test_value_path_agrees_with_the_jet(field, coords) -> None:
    """The cheap forward path and row 0 of the jet are the same function."""
    state = field(coords)
    jet, _ = field._jet_at_least(state, 2)
    assert np.allclose(field.net.value(coords), jet[:, 0], atol=1e-14)


def test_residual_costs_exactly_one_jet(field, coords) -> None:
    state = field(coords)
    out = jeq.burgers(state, nu=0.05)
    assert out.residual.shape == (coords.shape[0],)
    assert list(state.extra[JET_CACHE_KEY]) == [3]


# --------------------------- the non-local structure -------------------------


def test_attention_weights_are_a_partition_of_unity(field, coords) -> None:
    w = field.attention_weights(coords)
    assert w.shape == (6, field.memory)
    assert bool((w >= 0).all())
    assert np.allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-14)


def test_larger_beta_sharpens_the_partition(specs, coords) -> None:
    """Temperature collapse (the feasibility sense), not the founding delta -> 0 one."""
    cspec, mspec = specs
    peaks = []
    for beta in (0.5, 4.0, 40.0):
        f = make_attention_vector_field(
            coordinate_spec=cspec,
            components=mspec,
            hidden=6,
            depth=2,
            memory=5,
            beta=beta,
            seed=5,
        )
        peaks.append(float(jnp.mean(jnp.max(f.attention_weights(coords), axis=-1))))
    assert peaks[0] < peaks[1] < peaks[2]


def test_the_field_is_genuinely_non_local(field, coords) -> None:
    """Perturbing one memory slot moves the value at *every* collocation point."""
    before = field.net.value(coords)
    bumped = dataclasses.replace(
        field.net, values=field.net.values.at[0].add(1.0)
    )
    moved = jnp.min(jnp.abs(bumped.value(coords) - before), axis=0)
    assert bool((moved > 1e-9).all())


# --------------------------- pytree / trainability ---------------------------


def test_gradients_reach_the_memory_and_the_temperature(field, coords) -> None:
    def loss(f):
        return jnp.mean(ops.laplacian(f(coords), "u") ** 2)

    grads = jax.grad(loss)(field)
    assert float(jnp.abs(grads.net.keys).max()) > 0.0
    assert float(jnp.abs(grads.net.values).max()) > 0.0
    assert float(jnp.abs(grads.net.beta)) > 0.0


def test_the_field_is_jittable(field, coords) -> None:
    def residual(f):
        return ops.laplacian(f(coords), "u")

    assert np.allclose(jax.jit(residual)(field), residual(field), atol=1e-14)


def test_pytree_round_trip_preserves_the_field(field, coords) -> None:
    leaves, treedef = jax.tree_util.tree_flatten(field)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.array_equal(rebuilt.net.value(coords), field.net.value(coords))
    assert rebuilt.coordinate_spec == field.coordinate_spec
    assert rebuilt.jet_order == field.jet_order


# ------------------------------- construction --------------------------------


def test_uniform_memory_reduces_to_a_plain_deep_field(specs, coords) -> None:
    """Identical value slots make the mixture constant, leaving encoder + readout."""
    cspec, mspec = specs
    field = make_attention_vector_field(
        coordinate_spec=cspec,
        components=mspec,
        hidden=6,
        depth=2,
        memory=5,
        residual=True,
        seed=1,
    )
    net = dataclasses.replace(
        field.net, values=jnp.broadcast_to(field.net.values[:1], field.net.values.shape)
    )
    field = dataclasses.replace(field, net=net)

    plain = make_jet_mlp_vector_field(
        coordinate_spec=cspec, components=mspec, hidden=6, depth=2, jet_order=2
    )
    folded = dataclasses.replace(
        plain.net,
        weights=(*net.weights, net.readout_weight),
        biases=(*net.biases, net.readout_bias + net.readout_weight @ net.values[0]),
    )
    plain = dataclasses.replace(plain, net=folded)

    assert np.allclose(field.net.value(coords), plain.net.value(coords), atol=1e-13)
    fs, ps = field(coords), plain(coords)
    for order in (1, 2):
        assert np.allclose(
            ops.derivative(fs, "u", axis=0, order=order),
            ops.derivative(ps, "u", axis=0, order=order),
            atol=1e-13,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"memory": 0}, "memory must be >= 1"),
        ({"beta": 0.0}, "beta must be > 0"),
        ({"depth": 0}, "depth"),
        ({"value_dim": 3}, "residual needs value_dim == hidden"),
    ],
)
def test_construction_is_validated(specs, kwargs, match) -> None:
    cspec, mspec = specs
    base = dict(
        coordinate_spec=cspec, components=mspec, hidden=6, depth=2, memory=4
    )
    with pytest.raises(ValueError, match=match):
        make_attention_vector_field(**{**base, **kwargs})


def test_asymmetric_value_width_is_allowed_without_the_skip(specs, coords) -> None:
    cspec, mspec = specs
    field = make_attention_vector_field(
        coordinate_spec=cspec,
        components=mspec,
        hidden=6,
        depth=2,
        memory=4,
        value_dim=3,
        residual=False,
    )
    assert field.net.value(coords).shape == (6, 2)
    hess_ref = jax.vmap(jax.hessian(_scalar(field, "u")))(coords)
    assert np.allclose(ops.hessian(field(coords), "u"), hess_ref, atol=1e-11)


def test_an_activation_without_a_tower_is_rejected(specs) -> None:
    """Every registered activation now has a tower, so probe the guard directly."""
    cspec, mspec = specs
    nofp = dataclasses.replace(get_activation("tanh"), name="tanh_nofp", fastpath=None)
    with pytest.raises(ValueError, match="closed-form"):
        make_attention_vector_field(
            coordinate_spec=cspec, components=mspec, hidden=4, depth=1, base=nofp
        )
