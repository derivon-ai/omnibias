# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The non-local attention PINN field (torch).

Every other field on the substrate is a chain of elementwise activations over
affine maps, so its jet is the plain Faa di Bruno recursion. This one routes the
coordinates through a softmax over a trainable memory, which couples all slots
through a shared denominator -- exactly the construction where closed-form
differentiation is normally abandoned.

The tests therefore pin, in order: that the coordinate derivatives still match
``torch.autograd.grad`` to float64 round-off at orders 1-3; that the whole
operator surface (gradient / Hessian / Laplacian / prebuilt residuals) reaches
the field on one memoised *hidden* jet (live readout applied per call); that
the softmax weights are a genuine partition of unity that sharpens with
``beta``; and that gradients flow to the memory and the temperature so the
block is trainable.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch import equations as teq
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import (
    AttentionVectorField,
    JetMLPVectorField,
    build_attention_vector_field,
)
from omnibias.pinn.torch.fields.jet_mlp import JET_CACHE_KEY
from omnibias.torch.activations import get_activation
from omnibias.torch.architectures.attention import AttentionJetMLP


@pytest.fixture
def coords() -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    return torch.randn(6, 2, generator=g, dtype=torch.float64)


@pytest.fixture
def specs():
    return (
        CoordinateSpec(("x", "t")),
        ComponentSpec(("u", "v"), groups={"vel": ("u", "v")}),
    )


@pytest.fixture
def field(specs) -> AttentionVectorField:
    cspec, mspec = specs
    return build_attention_vector_field(
        coordinate_spec=cspec,
        components=mspec,
        hidden=6,
        depth=2,
        memory=5,
        beta=1.4,
        jet_order=3,
        seed=5,
    )


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


# ---------------------- exactness of the coordinate story --------------------


@pytest.mark.parametrize("name", ["u", "v"])
def test_partials_match_autograd(field, coords, name) -> None:
    state = field(coords)
    for alpha, expected in _autograd_partials(field, coords, name, 3).items():
        if sum(alpha) == 0:
            got = tops.value(state, name)
        else:
            axes = tuple(i for i, a in enumerate(alpha) if a)
            orders = tuple(alpha[i] for i in axes)
            got = tops.mixed_partial(state, name, axes, orders)
        scale = max(float(expected.detach().abs().max()), 1.0)
        assert torch.allclose(got, expected, rtol=1e-11, atol=1e-11 * scale), alpha


def test_value_path_agrees_with_the_jet(field, coords) -> None:
    """The cheap forward path and row 0 of the jet are the same function."""
    state = field(coords)
    jet, _ = field._jet_at_least(state, 2)
    assert torch.allclose(field.forward_values(coords), jet[:, 0], atol=1e-14)


def test_residual_costs_exactly_one_jet(field, coords) -> None:
    state = field(coords)
    out = teq.burgers(state, nu=0.05)
    assert out.residual.shape == (coords.shape[0],)
    assert list(state.extra[JET_CACHE_KEY]) == [3]


def test_operator_surface_reaches_the_field(field, coords) -> None:
    state = field(coords)
    assert tops.gradient(state, "u").shape == (6, 1)  # one spatial axis
    assert tops.hessian(state, "u").shape == (6, 2, 2)  # all axes, including t
    assert tops.laplacian(state, "u").shape == (6,)
    assert tops.divergence(state, ("u",)).shape == (6,)  # one spatial axis


# --------------------------- the non-local structure -------------------------


def test_attention_weights_are_a_partition_of_unity(field, coords) -> None:
    w = field.attention_weights(coords)
    assert w.shape == (6, field.memory)
    assert bool((w >= 0).all())
    assert torch.allclose(w.sum(-1), torch.ones(6, dtype=torch.float64), atol=1e-14)


def test_larger_beta_sharpens_the_partition(specs, coords) -> None:
    """Temperature collapse (the feasibility sense), not the founding delta -> 0 one."""
    cspec, mspec = specs
    peaks = []
    for beta in (0.5, 4.0, 40.0):
        f = build_attention_vector_field(
            coordinate_spec=cspec,
            components=mspec,
            hidden=6,
            depth=2,
            memory=5,
            beta=beta,
            seed=5,
        )
        w = f.attention_weights(coords).max(dim=-1).values
        peaks.append(float(w.mean().detach()))
    assert peaks[0] < peaks[1] < peaks[2]


def test_the_field_is_genuinely_non_local(field, coords) -> None:
    """Perturbing one memory slot moves the value at *every* collocation point.

    A local field cannot do this: its parameters act through a pointwise chain, so
    a change confined to one slot would leave far-away points untouched only if the
    softmax decoupled them. It does not -- the shared denominator couples all of
    them, which is the property this field exists to provide.
    """
    before = field.forward_values(coords).clone()
    with torch.no_grad():
        field.net.values[0] += 1.0
    after = field.forward_values(coords)
    moved = (after - before).abs().min(dim=0).values
    assert bool((moved > 1e-9).all())


# ------------------------------ trainability ---------------------------------


def test_gradients_reach_the_memory_and_the_temperature(specs, coords) -> None:
    cspec, mspec = specs
    field = AttentionVectorField(
        coordinate_spec=cspec,
        components=mspec,
        hidden=6,
        depth=2,
        memory=5,
        beta=1.2,
        learnable_temperature=True,
        jet_order=2,
    )
    state = field(coords)
    loss = tops.laplacian(state, "u").pow(2).mean()
    loss.backward()
    net = field.net
    assert isinstance(net, AttentionJetMLP)
    for name, p in (("keys", net.keys), ("values", net.values), ("beta", net.beta_raw)):
        assert p.grad is not None, name
        assert float(p.grad.abs().max()) > 0.0, name


def test_temperature_is_a_buffer_unless_asked_for(field) -> None:
    net = field.net
    assert isinstance(net, AttentionJetMLP)
    assert not isinstance(net.beta_raw, torch.nn.Parameter)
    assert "beta_raw" in dict(net.named_buffers())


# ------------------------------- construction --------------------------------


def test_uniform_memory_reduces_to_a_plain_deep_field(specs, coords) -> None:
    """Identical value slots make the mixture constant, leaving encoder + readout.

    The sanity check that the block is wired correctly: with ``V`` constant across
    slots the softmax cannot matter, and the field must agree with the
    corresponding plain jet field built from the same encoder weights.
    """
    cspec, mspec = specs
    field = build_attention_vector_field(
        coordinate_spec=cspec,
        components=mspec,
        hidden=6,
        depth=2,
        memory=5,
        residual=True,
        seed=1,
    )
    net = field.net
    assert isinstance(net, AttentionJetMLP)
    with torch.no_grad():
        net.values.copy_(net.values[:1].expand_as(net.values))

    plain = JetMLPVectorField(
        coordinate_spec=cspec, components=mspec, hidden=6, depth=2, jet_order=2
    )
    with torch.no_grad():
        for lin, src in zip(plain.net.linears[:-1], net.encoder, strict=True):
            lin.weight.copy_(src.weight)
            lin.bias.copy_(src.bias)
        # Fold the constant attention output into the readout bias.
        plain.net.linears[-1].weight.copy_(net.readout.weight)
        plain.net.linears[-1].bias.copy_(
            net.readout.bias + net.readout.weight @ net.values[0]
        )
    # The plain field applies the readout to sigma(W_2 h); the attention field
    # applies it to sigma(W_2 h) + const, which the folded bias reproduces.
    assert torch.allclose(
        field.forward_values(coords), plain.forward_values(coords), atol=1e-13
    )
    fs, ps = field(coords), plain(coords)
    for order in (1, 2):
        assert torch.allclose(
            tops.derivative(fs, "u", axis=0, order=order),
            tops.derivative(ps, "u", axis=0, order=order),
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
        AttentionVectorField(**{**base, **kwargs})


def test_asymmetric_value_width_is_allowed_without_the_skip(specs, coords) -> None:
    cspec, mspec = specs
    field = AttentionVectorField(
        coordinate_spec=cspec,
        components=mspec,
        hidden=6,
        depth=2,
        memory=4,
        value_dim=3,
        residual=False,
    )
    assert field.forward_values(coords).shape == (6, 2)
    for alpha, expected in _autograd_partials(field, coords, "u", 2).items():
        if sum(alpha) == 0:
            continue
        axes = tuple(i for i, a in enumerate(alpha) if a)
        orders = tuple(alpha[i] for i in axes)
        got = tops.mixed_partial(field(coords), "u", axes, orders)
        assert torch.allclose(got, expected, rtol=1e-11, atol=1e-11)


def test_an_activation_without_a_tower_is_rejected(specs) -> None:
    """Every registered activation now has a tower, so probe the guard directly."""
    cspec, mspec = specs
    nofp = dataclasses.replace(get_activation("tanh"), name="tanh_nofp", fastpath=None)
    with pytest.raises(ValueError, match="closed-form"):
        AttentionVectorField(
            coordinate_spec=cspec, components=mspec, hidden=4, depth=1, base=nofp
        )
