# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multi-scale PINN fields (JAX): adaptive slopes and MscaleDNN band mixtures.

Mirrors :mod:`tests.torch.test_torch_multiscale_field` so cross-backend parity has
a like-for-like surface, and additionally pins the JAX-specific contract: the
trainable slopes and band weights are pytree leaves, so ``jax.grad`` reaches them
and ``jax.jit`` traces through the whole construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation
from omnibias.jax.architectures.multiscale import (
    make_adaptive_activation,
    make_mscale_mlp,
)
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.multiscale import suggest_frequency_bands
from omnibias.pinn.jax import ops
from omnibias.pinn.jax.fields import (
    make_adaptive_jet_mlp_vector_field,
    make_jet_mlp_vector_field,
    make_mscale_vector_field,
)
from omnibias.pinn.jax.fields.jet_mlp import JET_CACHE_KEY

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


def _assert_matches_autodiff(field, coords, *, name: str = "u"):
    state = field(coords)
    f = _scalar(field, name)
    grad_ref = jax.vmap(jax.grad(f))(coords)
    hess_ref = jax.vmap(jax.hessian(f))(coords)
    scale = max(float(jnp.abs(hess_ref).max()), 1.0)
    assert np.allclose(ops.gradient(state, name), grad_ref, rtol=1e-11, atol=1e-11 * scale)
    assert np.allclose(ops.hessian(state, name), hess_ref, rtol=1e-11, atol=1e-11 * scale)
    assert np.allclose(
        ops.laplacian(state, name), jnp.trace(hess_ref, axis1=1, axis2=2),
        rtol=1e-11, atol=1e-11 * scale,
    )


# -- the adaptive activation is a genuine ActivationSpec ---------------------- #


@pytest.mark.parametrize("base", ["tanh", "sigmoid", "softplus", "sin"])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_adaptive_spec_obeys_the_chain_rule_exactly(base, order):
    """``sigma_a^(k)(z)`` must be exactly ``(n a)^k sigma^(k)(n a z)``."""
    act = make_adaptive_activation(base, slope_scale=3.0, scale_init=1.7)
    z = jnp.linspace(-2.0, 2.0, 17, dtype=jnp.float64)
    s = act.scale
    expected = s**order * get_activation(base).fastpath(s * z, order)
    assert np.allclose(act.spec.fastpath(z, order), expected, rtol=TOL, atol=TOL)


def test_unit_slope_is_the_base_activation():
    act = make_adaptive_activation("tanh", scale_init=1.0)
    assert float(act.scale) == 1.0
    z = jnp.linspace(-3.0, 3.0, 21, dtype=jnp.float64)
    base = get_activation("tanh")
    for k in (0, 1, 2, 3):
        assert np.array_equal(act.spec.fastpath(z, k), base.fastpath(z, k))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"slope_scale": 0.0}, "slope_scale must be > 0"),
        ({"scale_init": -1.0}, "scale_init must be > 0"),
        ({"width": 0}, "width must be >= 1"),
    ],
)
def test_adaptive_activation_rejects_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_adaptive_activation("tanh", **kwargs)


# -- the adaptive field ------------------------------------------------------- #


@pytest.mark.parametrize("granularity", ["layer", "neuron"])
@pytest.mark.parametrize("depth", [1, 3])
def test_adaptive_field_derivatives_match_autodiff(coords, specs, granularity, depth):
    cs, comps = specs
    field = make_adaptive_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=depth,
        granularity=granularity, slope_scale=2.0, scale_init=1.5, seed=1,
    )
    _assert_matches_autodiff(field, coords)


def test_unit_slope_adaptive_field_equals_a_plain_jet_field(coords, specs):
    """At effective slope 1 the adaptive field *is* the plain deep field.

    Both builders draw their weights from the same seeded ``make_jet_mlp``, so the
    two fields are the same network and every operator must agree exactly.
    """
    cs, comps = specs
    kw = dict(coordinate_spec=cs, components=comps, hidden=7, depth=2, seed=0)
    adaptive = make_adaptive_jet_mlp_vector_field(**kw)
    plain = make_jet_mlp_vector_field(**kw)
    s_a, s_p = adaptive(coords), plain(coords)
    for op in (ops.value, ops.gradient, ops.laplacian, ops.hessian):
        assert np.allclose(op(s_a, "u"), op(s_p, "u"), rtol=TOL, atol=TOL), op.__name__


def test_slope_receives_gradient_and_slope_scale_amplifies_it(coords, specs):
    """``n`` exists to scale the slope gradient; that factor must be exactly ``n``."""
    cs, comps = specs

    def loss(field, xs):
        return jnp.mean(ops.laplacian(field(xs), "u") ** 2)

    def grads(n: float):
        field = make_adaptive_jet_mlp_vector_field(
            coordinate_spec=cs, components=comps, hidden=6, depth=2,
            slope_scale=n, scale_init=1.0, seed=0,
        )
        return float(loss(field, coords)), jax.grad(loss)(field, coords)

    l1, g1 = grads(1.0)
    l10, g10 = grads(10.0)
    assert l1 == pytest.approx(l10), "same effective slope, same function"
    a1 = float(g1.net.activations[0].a)
    a10 = float(g10.net.activations[0].a)
    assert abs(a1) > 0.0
    assert a10 == pytest.approx(10.0 * a1, rel=1e-9)


def test_neuron_granularity_gives_one_slope_per_unit(specs):
    cs, comps = specs
    field = make_adaptive_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=9, depth=2, granularity="neuron",
    )
    assert all(s.shape == (9,) for s in field.slopes())
    layerwise = make_adaptive_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=9, depth=2,
    )
    assert all(s.shape == () for s in layerwise.slopes())


def test_adaptive_field_rejects_bad_granularity(specs):
    cs, comps = specs
    with pytest.raises(ValueError, match="granularity must be"):
        make_adaptive_jet_mlp_vector_field(
            coordinate_spec=cs, components=comps, hidden=4, depth=1, granularity="global",
        )


# -- the Mscale band mixture -------------------------------------------------- #


@pytest.mark.parametrize("scales", [(1.0,), (1.0, 4.0), (0.5, 2.0, 8.0)])
def test_mscale_field_derivatives_match_autodiff(coords, specs, scales):
    cs, comps = specs
    field = make_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=8, depth=2, scales=scales, seed=1,
    )
    _assert_matches_autodiff(field, coords)


def test_single_unit_band_reduces_to_a_plain_jet_field(coords, specs):
    """A one-band mixture at ``alpha = 1`` is exactly a plain deep field."""
    cs, comps = specs
    kw = dict(coordinate_spec=cs, components=comps, hidden=6, depth=2, seed=0)
    mix = make_mscale_vector_field(scales=(1.0,), **kw)
    plain = make_jet_mlp_vector_field(**kw)
    s_m, s_p = mix(coords), plain(coords)
    for op in (ops.value, ops.gradient, ops.laplacian, ops.biharmonic):
        assert np.allclose(op(s_m, "u"), op(s_p, "u"), rtol=TOL, atol=TOL), op.__name__


def test_band_scaling_obeys_the_derivative_scaling_law(coords, specs):
    r"""One band at ``alpha`` must satisfy ``d/dx f(alpha x) = alpha f'(alpha x)``."""
    cs, comps = specs
    alpha = 3.0
    kw = dict(coordinate_spec=cs, components=comps, hidden=6, depth=2, seed=4)
    scaled = make_mscale_vector_field(scales=(alpha,), **kw)
    plain = make_jet_mlp_vector_field(**kw)

    s_scaled = scaled(coords)
    s_plain = plain(alpha * coords)
    assert np.allclose(
        ops.value(s_scaled, "u"), ops.value(s_plain, "u"), rtol=TOL, atol=TOL
    )
    assert np.allclose(
        ops.gradient(s_scaled, "u"), alpha * ops.gradient(s_plain, "u"),
        rtol=1e-11, atol=1e-11,
    )
    assert np.allclose(
        ops.laplacian(s_scaled, "u"), alpha**2 * ops.laplacian(s_plain, "u"),
        rtol=1e-11, atol=1e-11,
    )


def test_mixture_is_the_sum_of_its_bands(coords, specs):
    cs, comps = specs
    field = make_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=8, depth=2, scales=(1.0, 4.0), seed=2,
    )
    total = sum(
        sub.value(scale * coords)
        for scale, sub in zip(field.net.scales, field.net.subnets, strict=True)
    )
    assert np.allclose(field.net.value(coords), total, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scales": ()}, "at least one band"),
        ({"scales": (1.0, -2.0)}, "must be > 0"),
    ],
)
def test_mscale_rejects_bad_scales(specs, kwargs, match):
    cs, comps = specs
    with pytest.raises(ValueError, match=match):
        make_mscale_vector_field(
            coordinate_spec=cs, components=comps, hidden=4, depth=1, **kwargs
        )


def test_band_widths_are_split_from_the_total(specs):
    net = make_mscale_mlp(2, 64, out_dim=1, depth=2, scales=(1.0, 2.0, 4.0, 8.0))
    assert len(net.subnets) == 4
    assert all(int(sub.weights[0].shape[0]) == 16 for sub in net.subnets)


# -- shared jet_mlp plumbing -------------------------------------------------- #


@pytest.fixture(params=["adaptive", "mscale"])
def multiscale_field(request, specs):
    cs, comps = specs
    if request.param == "adaptive":
        return make_adaptive_jet_mlp_vector_field(
            coordinate_spec=cs, components=comps, hidden=6, depth=2, jet_order=2,
        )
    return make_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=8, depth=2,
        scales=(1.0, 4.0), jet_order=2,
    )


def test_multiscale_fields_carry_the_jet_mlp_tag_and_cache(coords, multiscale_field):
    field = multiscale_field
    assert field._omnibias_dispatch == "jet_mlp"
    state = field(coords)
    ops.value(state, "u")
    ops.gradient(state, "u")
    ops.laplacian(state, "u")
    ops.hessian(state, "u")
    ops.divergence(state, ("u", "v"))
    assert sorted(state.extra[JET_CACHE_KEY]) == [2], "one jet serves the whole residual"


def test_field_is_a_pytree_and_grad_reaches_every_leaf(coords, multiscale_field):
    field = multiscale_field
    leaves = jax.tree_util.tree_leaves(field)
    assert leaves and all(hasattr(x, "shape") for x in leaves)

    def loss(f, xs):
        return jnp.mean(ops.laplacian(f(xs), "u") ** 2)

    grads = jax.grad(loss)(field, coords)
    g_leaves = jax.tree_util.tree_leaves(grads)
    assert len(g_leaves) == len(leaves)
    assert any(float(jnp.abs(g).max()) > 0 for g in g_leaves)


def test_jit_traces_through_the_field(coords, multiscale_field):
    field = multiscale_field

    @jax.jit
    def residual(f, xs):
        return ops.laplacian(f(xs), "u")

    assert np.allclose(
        residual(field, coords), ops.laplacian(field(coords), "u"),
        rtol=TOL, atol=TOL,
    )


def test_roundtrip_through_flatten_unflatten(coords, multiscale_field):
    field = multiscale_field
    leaves, treedef = jax.tree_util.tree_flatten(field)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(field)
    assert np.allclose(
        ops.laplacian(rebuilt(coords), "u"), ops.laplacian(field(coords), "u"),
        rtol=TOL, atol=TOL,
    )


# -- the feedback loop: measured spectrum -> band scales -> field ------------- #


def test_suggested_bands_configure_a_multiscale_field(specs):
    cs, comps = specs
    x = np.linspace(0.0, 1.0, 256, endpoint=False)
    u = np.sin(2 * np.pi * 2 * x) + 0.5 * np.sin(2 * np.pi * 15 * x)
    bands = suggest_frequency_bands(u[None, :], L=1.0, n_bands=2)
    assert bands == pytest.approx((2.0, 15.0))
    field = make_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=8, depth=1, scales=bands,
    )
    assert field.scales == bands
