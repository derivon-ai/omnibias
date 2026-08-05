# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multi-scale PINN fields (torch): adaptive slopes and MscaleDNN band mixtures.

The claim under test is that putting the frequency knob *inside* the network
costs nothing in exactness. A trainable activation slope and a band mixture are
both places where closed-form differentiation is normally abandoned, so each is
checked three ways: against ``torch.autograd.grad`` to float64 round-off, against
the analytic scaling law it is supposed to obey, and for the gradient that makes
the knob trainable in the first place.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.multiscale import suggest_frequency_bands
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import (
    AdaptiveJetMLPVectorField,
    FourierFeatureVectorField,
    JetMLPVectorField,
    MscaleVectorField,
    build_adaptive_jet_mlp_vector_field,
    build_mscale_vector_field,
)
from omnibias.pinn.torch.fields.jet_mlp import JET_CACHE_KEY
from omnibias.torch.activations import get_activation
from omnibias.torch.architectures.multiscale import AdaptiveActivation, MscaleMLP

TOL = 1e-12


@pytest.fixture
def coords() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randn(6, 2, generator=g, dtype=torch.float64)


@pytest.fixture
def specs():
    return CoordinateSpec(("x", "y")), ComponentSpec(("u", "v"), groups={"vel": ("u", "v")})


def _autograd_partials(field, coords: torch.Tensor, name: str, max_order: int):
    """``{alpha: D^alpha u}`` from nested ``torch.autograd.grad``, as the reference."""
    ci = field.components.index(name)
    x = coords.clone().requires_grad_(True)
    D = coords.shape[-1]
    out: dict[tuple[int, ...], torch.Tensor] = {}
    frontier = {(0,) * D: field.forward_values(x)[:, ci]}
    out.update(frontier)
    for _ in range(max_order):
        nxt: dict[tuple[int, ...], torch.Tensor] = {}
        for alpha, val in frontier.items():
            g = torch.autograd.grad(val.sum(), x, create_graph=True)[0]
            for i in range(D):
                beta = tuple(a + (1 if j == i else 0) for j, a in enumerate(alpha))
                if beta not in nxt:
                    nxt[beta] = g[:, i]
        out.update(nxt)
        frontier = nxt
    return out


def _assert_matches_autograd(field, coords, max_order: int, *, name: str = "u"):
    state = field(coords)
    ref = _autograd_partials(field, coords, name, max_order)
    for alpha, expected in ref.items():
        if sum(alpha) == 0:
            got = tops.value(state, name)
        else:
            axes = tuple(i for i, a in enumerate(alpha) if a)
            orders = tuple(alpha[i] for i in axes)
            got = tops.mixed_partial(state, name, axes, orders)
        scale = max(float(expected.detach().abs().max()), 1.0)
        assert torch.allclose(got, expected, rtol=1e-11, atol=1e-11 * scale), (
            f"D^{alpha} mismatch"
        )


# -- the adaptive activation is a genuine ActivationSpec ---------------------- #


@pytest.mark.parametrize("base", ["tanh", "sigmoid", "softplus", "sin"])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_adaptive_spec_obeys_the_chain_rule_exactly(base, order):
    """``sigma_a^(k)(z)`` must be exactly ``(n a)^k sigma^(k)(n a z)``."""
    act = AdaptiveActivation(base, slope_scale=3.0, scale_init=1.7, dtype=torch.float64)
    z = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64)
    s = act.scale
    expected = s**order * get_activation(base).fastpath(s * z, order)
    assert torch.allclose(act.fastpath(z, order), expected, rtol=TOL, atol=TOL)


def test_unit_slope_is_the_base_activation():
    """The default start (effective slope 1, ``n = 1``) is the base activation exactly."""
    act = AdaptiveActivation("tanh", scale_init=1.0, dtype=torch.float64)
    assert float(act.scale.detach()) == 1.0
    z = torch.linspace(-3.0, 3.0, 21, dtype=torch.float64)
    base = get_activation("tanh")
    for k in (0, 1, 2, 3):
        assert torch.equal(act.fastpath(z, k), base.fastpath(z, k))


def test_amplified_slope_still_starts_at_the_base_activation():
    """With ``n != 1`` the stored ``scale_init / n`` round trip costs at most an ulp."""
    act = AdaptiveActivation("tanh", slope_scale=7.0, scale_init=1.0, dtype=torch.float64)
    z = torch.linspace(-3.0, 3.0, 21, dtype=torch.float64)
    base = get_activation("tanh")
    for k in (0, 1, 2, 3):
        assert torch.allclose(act.fastpath(z, k), base.fastpath(z, k), rtol=TOL, atol=TOL)


def test_adaptive_spec_tracks_the_parameter_after_an_update():
    """The spec is rebuilt from the live parameter, not snapshotted at construction."""
    act = AdaptiveActivation("tanh", dtype=torch.float64)
    z = torch.tensor([0.5], dtype=torch.float64)
    before = act.forward(z).clone()
    with torch.no_grad():
        act.a.fill_(2.0)
    after = act.forward(z)
    assert not torch.allclose(before, after)
    assert torch.allclose(after, torch.tanh(2.0 * z), rtol=TOL, atol=TOL)


def test_adaptive_activation_rejects_a_base_without_a_tower():
    with pytest.raises(ValueError, match="closed-form"):
        AdaptiveActivation(
            get_activation("tanh").__class__(
                name="no_tower",
                forward=torch.tanh,
                derivative=torch.tanh,
                fastpath=None,
            )
        )


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
        AdaptiveActivation("tanh", **kwargs)


# -- the adaptive field ------------------------------------------------------- #


@pytest.mark.parametrize("granularity", ["layer", "neuron"])
@pytest.mark.parametrize("depth", [1, 3])
def test_adaptive_field_derivatives_match_autograd(coords, specs, granularity, depth):
    cs, comps = specs
    field = build_adaptive_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=depth,
        granularity=granularity, slope_scale=2.0, scale_init=1.5,
        jet_order=3, seed=1,
    )
    _assert_matches_autograd(field, coords, 3)


def test_unit_slope_adaptive_field_equals_a_plain_jet_field(coords, specs):
    """At effective slope 1 the adaptive field *is* the plain deep field."""
    cs, comps = specs
    adaptive = build_adaptive_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=7, depth=2, jet_order=3, seed=0,
    )
    plain = JetMLPVectorField(
        coordinate_spec=cs, components=comps, hidden=7, depth=2, jet_order=3,
    )
    with torch.no_grad():
        for dst, src in zip(plain.net.linears, adaptive.net.linears, strict=True):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)

    s_a, s_p = adaptive(coords), plain(coords)
    for op in (tops.value, tops.gradient, tops.laplacian, tops.hessian):
        assert torch.allclose(op(s_a, "u"), op(s_p, "u"), rtol=TOL, atol=TOL), op.__name__


def test_slope_receives_gradient_and_slope_scale_amplifies_it(coords, specs):
    """``n`` exists to scale the slope gradient; that factor must be exactly ``n``.

    Two fields with identical weights and the *same* effective slope ``n a = 1``
    compute the same function, so any difference in ``dL/da`` is due to ``n``
    alone -- and the chain rule says it is exactly ``n``.
    """
    cs, comps = specs

    def make(n: float):
        torch.manual_seed(0)
        return AdaptiveJetMLPVectorField(
            coordinate_spec=cs, components=comps, hidden=6, depth=2,
            slope_scale=n, scale_init=1.0,
        )

    f1, f10 = make(1.0), make(10.0)
    losses = []
    for f in (f1, f10):
        loss = tops.laplacian(f(coords), "u").pow(2).mean()
        loss.backward()
        losses.append(float(loss.detach()))
    assert losses[0] == pytest.approx(losses[1]), "same effective slope, same function"

    g1 = f1.net.activations[0].a.grad
    g10 = f10.net.activations[0].a.grad
    assert g1 is not None and g10 is not None
    assert float(g1.abs()) > 0.0
    assert float(g10) == pytest.approx(10.0 * float(g1), rel=1e-9)


def test_neuron_granularity_gives_one_slope_per_unit(specs):
    cs, comps = specs
    field = build_adaptive_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=9, depth=2, granularity="neuron",
    )
    for s in field.slopes():
        assert s.shape == (9,)
    layerwise = build_adaptive_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=9, depth=2,
    )
    for s in layerwise.slopes():
        assert s.shape == ()


def test_adaptive_field_rejects_bad_granularity(specs):
    cs, comps = specs
    with pytest.raises(ValueError, match="granularity must be"):
        AdaptiveJetMLPVectorField(
            coordinate_spec=cs, components=comps, hidden=4, depth=1, granularity="global",
        )


# -- the Mscale band mixture -------------------------------------------------- #


@pytest.mark.parametrize("scales", [(1.0,), (1.0, 4.0), (0.5, 2.0, 8.0)])
def test_mscale_field_derivatives_match_autograd(coords, specs, scales):
    cs, comps = specs
    field = build_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=8, depth=2, scales=scales,
        jet_order=3, seed=1,
    )
    _assert_matches_autograd(field, coords, 3)


def test_single_unit_band_reduces_to_a_plain_jet_field(coords, specs):
    """A one-band mixture at ``alpha = 1`` is exactly a plain deep field."""
    cs, comps = specs
    mix = build_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, scales=(1.0,),
        jet_order=3, seed=0,
    )
    plain = JetMLPVectorField(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, jet_order=3,
    )
    with torch.no_grad():
        for dst, src in zip(plain.net.linears, mix.net.subnets[0].linears, strict=True):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
    s_m, s_p = mix(coords), plain(coords)
    for op in (tops.value, tops.gradient, tops.laplacian, tops.biharmonic):
        assert torch.allclose(op(s_m, "u"), op(s_p, "u"), rtol=TOL, atol=TOL), op.__name__


def test_band_scaling_obeys_the_derivative_scaling_law(coords, specs):
    r"""One band at ``alpha`` must satisfy ``d/dx f(alpha x) = alpha f'(alpha x)``.

    This is the identity the whole construction rests on -- scaling the input is
    the same as scaling the first weight matrix -- so it is checked directly
    against the *unscaled* band evaluated at ``alpha x``.
    """
    cs, comps = specs
    alpha = 3.0
    scaled = build_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, scales=(alpha,),
        jet_order=2, seed=4,
    )
    plain = JetMLPVectorField(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, jet_order=2,
    )
    with torch.no_grad():
        for dst, src in zip(plain.net.linears, scaled.net.subnets[0].linears, strict=True):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)

    s_scaled = scaled(coords)
    s_plain = plain(alpha * coords)
    assert torch.allclose(
        tops.value(s_scaled, "u"), tops.value(s_plain, "u"), rtol=TOL, atol=TOL
    )
    assert torch.allclose(
        tops.gradient(s_scaled, "u"), alpha * tops.gradient(s_plain, "u"),
        rtol=1e-11, atol=1e-11,
    )
    assert torch.allclose(
        tops.laplacian(s_scaled, "u"), alpha**2 * tops.laplacian(s_plain, "u"),
        rtol=1e-11, atol=1e-11,
    )


def test_mixture_is_the_sum_of_its_bands(coords, specs):
    """``u = sum_j f_j(alpha_j x)``: the field value must equal the band sum."""
    cs, comps = specs
    field = build_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=8, depth=2, scales=(1.0, 4.0), seed=2,
    )
    total = sum(
        sub.value(scale * coords)
        for scale, sub in zip(field.net.scales, field.net.subnets, strict=True)
    )
    assert torch.allclose(field.forward_values(coords), total, rtol=1e-11, atol=1e-11)


def test_total_width_is_split_across_bands(specs):
    cs, comps = specs
    field = build_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=64, depth=2,
        scales=(1.0, 2.0, 4.0, 8.0),
    )
    assert field.band_hidden == 16
    assert field.scales == (1.0, 2.0, 4.0, 8.0)
    plain = JetMLPVectorField(
        coordinate_spec=cs, components=comps, hidden=64, depth=2,
    )
    n_mix = sum(p.numel() for p in field.parameters())
    n_plain = sum(p.numel() for p in plain.parameters())
    assert n_mix < n_plain, "splitting the width must not cost more than one wide MLP"


def test_adaptive_bands_compose(coords, specs):
    """``adaptive=True`` puts a trainable slope inside every band, still exactly."""
    cs, comps = specs
    field = build_mscale_vector_field(
        coordinate_spec=cs, components=comps, hidden=8, depth=2, scales=(1.0, 4.0),
        adaptive=True, jet_order=2, seed=3,
    )
    _assert_matches_autograd(field, coords, 2)
    tops.laplacian(field(coords), "u").pow(2).mean().backward()
    slopes = [sub.activations[0].a for sub in field.net.subnets]
    assert all(s.grad is not None and float(s.grad.abs()) > 0 for s in slopes)


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
        MscaleVectorField(
            coordinate_spec=cs, components=comps, hidden=4, depth=1, **kwargs
        )


def test_mscale_is_not_a_single_layer_chain(specs):
    """A parallel graph must refuse to pretend it is one chain rather than lie."""
    net = MscaleMLP(2, 8, out_dim=1, depth=1, scales=(1.0, 2.0))
    with pytest.raises(NotImplementedError, match="sum of band subnetworks"):
        net._layer_specs()


# -- shared jet_mlp plumbing -------------------------------------------------- #


@pytest.mark.parametrize("builder", ["adaptive", "mscale"])
def test_multiscale_fields_carry_the_jet_mlp_tag_and_cache(coords, specs, builder):
    cs, comps = specs
    field = (
        build_adaptive_jet_mlp_vector_field(
            coordinate_spec=cs, components=comps, hidden=6, depth=2, jet_order=2,
        )
        if builder == "adaptive"
        else build_mscale_vector_field(
            coordinate_spec=cs, components=comps, hidden=8, depth=2,
            scales=(1.0, 4.0), jet_order=2,
        )
    )
    assert field._omnibias_dispatch == "jet_mlp"
    assert all(p.dtype is torch.float64 for p in field.parameters())

    state = field(coords)
    tops.value(state, "u")
    tops.gradient(state, "u")
    tops.laplacian(state, "u")
    tops.hessian(state, "u")
    tops.divergence(state, ("u", "v"))
    assert sorted(state.extra[JET_CACHE_KEY]) == [2], "one jet serves the whole residual"


@pytest.mark.parametrize("builder", ["adaptive", "mscale"])
def test_gradients_flow_to_every_parameter(coords, specs, builder):
    cs, comps = specs
    field = (
        build_adaptive_jet_mlp_vector_field(
            coordinate_spec=cs, components=comps, hidden=6, depth=2,
        )
        if builder == "adaptive"
        else build_mscale_vector_field(
            coordinate_spec=cs, components=comps, hidden=8, depth=2, scales=(1.0, 4.0),
        )
    )
    tops.laplacian(field(coords), "u").pow(2).mean().backward()
    grads = [p.grad for p in field.parameters()]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().max()) > 0 for g in grads if g is not None)


# -- the feedback loop: measured spectrum -> band scales -> field ------------- #


def test_suggested_bands_configure_a_multiscale_field(specs):
    """The loop the diagnostics exist for: measure, then build with what you measured."""
    cs, comps = specs
    x = np.linspace(0.0, 1.0, 256, endpoint=False)
    u = np.sin(2 * np.pi * 2 * x) + 0.5 * np.sin(2 * np.pi * 15 * x)
    bands = suggest_frequency_bands(u[None, :], L=1.0, n_bands=2)
    assert bands == pytest.approx((2.0, 15.0))

    mix = MscaleVectorField(
        coordinate_spec=cs, components=comps, hidden=8, depth=1, scales=bands,
    )
    assert mix.scales == bands
    fourier = FourierFeatureVectorField(
        coordinate_spec=cs, components=comps, num_features=4, hidden=6, depth=1,
        frequency_scale=bands,
    )
    assert fourier.scales == bands
