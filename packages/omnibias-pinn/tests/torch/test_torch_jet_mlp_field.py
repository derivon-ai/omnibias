# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deep / Fourier-feature PINN fields (torch): exactness, caching, integration.

The contract under test is that ``jet_mlp`` fields deliver every derivative from
the exact multivariate jet -- matching ``torch.autograd.grad`` to float64
round-off at arbitrary depth and order -- while costing a *single* jet per
residual and reproducing ``OneLayerVectorField`` when the depth is 1.
"""

from __future__ import annotations

import pytest
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import (
    FourierFeatureVectorField,
    JetMLPVectorField,
    OneLayerVectorField,
    build_fourier_feature_vector_field,
    build_jet_mlp_vector_field,
    make_siren_vector_field,
)
from omnibias.pinn.torch.fields.jet_mlp import JET_CACHE_KEY

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


# -- exactness vs autograd ---------------------------------------------------- #


@pytest.mark.parametrize("base", ["tanh", "sigmoid", "softplus", "sin"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_all_partials_match_autograd(coords, specs, base, depth):
    cs, comps = specs
    field = build_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=depth,
        base=base, jet_order=3, seed=1,
    )
    state = field(coords)
    ref = _autograd_partials(field, coords, "u", 3)
    for alpha, expected in ref.items():
        total = sum(alpha)
        if total == 0:
            got = tops.value(state, "u")
        else:
            axes = tuple(i for i, a in enumerate(alpha) if a)
            orders = tuple(alpha[i] for i in axes)
            got = tops.mixed_partial(state, "u", axes, orders)
        assert torch.allclose(got, expected, rtol=TOL, atol=TOL), (
            f"D^{alpha} mismatch for base={base!r}, depth={depth}"
        )


def test_gradient_hessian_laplacian_match_autograd(coords, specs):
    cs, comps = specs
    field = build_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=8, depth=2, seed=3,
    )
    state = field(coords)
    ref = _autograd_partials(field, coords, "u", 2)
    D = coords.shape[-1]

    grad = tops.gradient(state, "u", axes=("x", "y"))
    for i in range(D):
        e = tuple(1 if j == i else 0 for j in range(D))
        assert torch.allclose(grad[:, i], ref[e], rtol=TOL, atol=TOL)

    hess = tops.hessian(state, "u")
    for i in range(D):
        for j in range(D):
            alpha = tuple(
                (1 if k == i else 0) + (1 if k == j else 0) for k in range(D)
            )
            assert torch.allclose(hess[:, i, j], ref[alpha], rtol=TOL, atol=TOL)

    lap = tops.laplacian(state, "u")
    expected = sum(
        ref[tuple(2 if j == i else 0 for j in range(D))] for i in range(D)
    )
    assert torch.allclose(lap, expected, rtol=TOL, atol=TOL)


def test_polylaplacian_matches_iterated_laplacian(coords, specs):
    """``Delta^k`` from the multinomial jet read must equal nested autograd Laplacians."""
    cs, comps = specs
    field = build_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, jet_order=6, seed=5,
    )
    state = field(coords)
    x = coords.clone().requires_grad_(True)
    val = field.forward_values(x)[:, 0]

    def lap(u):
        g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        return sum(
            torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0][:, i]
            for i in range(2)
        )

    for k in (1, 2, 3):
        val = lap(val)
        got = tops.polylaplacian(state, "u", k=k)
        assert torch.allclose(got, val, rtol=1e-11, atol=1e-11), f"Delta^{k}"
    assert torch.allclose(
        tops.biharmonic(state, "u"), tops.polylaplacian(state, "u", k=2),
        rtol=TOL, atol=TOL,
    )


# -- agreement with the single-layer closed-form tower ------------------------ #


def test_depth1_reproduces_one_layer_field(coords, specs):
    """A depth-1 jet field *is* ``OneLayerVectorField``, so every op must agree.

    Both compute the same mathematical object by different reductions -- the
    multi-layer Faa di Bruno recursion versus the single-layer ``sigma``-tower
    contraction -- so they agree to float64 round-off rather than bit-for-bit.
    """
    cs, comps = specs
    one = OneLayerVectorField(
        coordinate_spec=cs, components=comps, hidden=9, base="tanh",
        dtype=torch.float64,
    )
    jet = JetMLPVectorField(
        coordinate_spec=cs, components=comps, hidden=9, depth=1, base="tanh",
        jet_order=4,
    )
    with torch.no_grad():
        jet.net.linears[0].weight.copy_(one.W.weight)
        jet.net.linears[0].bias.copy_(one.W.bias)
        jet.net.linears[1].weight.copy_(one.c.weight)
        jet.net.linears[1].bias.copy_(one.c.bias)

    s_one, s_jet = one(coords), jet(coords)
    for op, kwargs in [
        (tops.value, {}),
        (tops.gradient, {}),
        (tops.laplacian, {}),
        (tops.hessian, {}),
        (tops.biharmonic, {}),
    ]:
        assert torch.allclose(
            op(s_one, "u", **kwargs), op(s_jet, "u", **kwargs), rtol=TOL, atol=TOL,
        ), f"{op.__name__} disagrees between one-layer and depth-1 jet field"
    for order in (1, 2, 3, 4):
        assert torch.allclose(
            tops.derivative(s_one, "u", axis="x", order=order),
            tops.derivative(s_jet, "u", axis="x", order=order),
            rtol=TOL, atol=TOL,
        )
    assert torch.allclose(
        tops.mixed_partial(s_one, "u", ("x", "y"), (2, 1)),
        tops.mixed_partial(s_jet, "u", ("x", "y"), (2, 1)),
        rtol=TOL, atol=TOL,
    )
    assert torch.allclose(
        tops.jacobian(s_one, ("u", "v")), tops.jacobian(s_jet, ("u", "v")),
        rtol=TOL, atol=TOL,
    )


# -- the jet cache is the whole point ----------------------------------------- #


def test_order2_residual_triggers_exactly_one_jet(coords, specs, monkeypatch):
    cs, comps = specs
    field = build_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, jet_order=2, seed=2,
    )
    calls: list[int] = []
    original = type(field)._compute_jet

    def counting(self, c, order):
        calls.append(order)
        return original(self, c, order)

    monkeypatch.setattr(type(field), "_compute_jet", counting, raising=False)

    state = field(coords)
    # A full second-order residual surface: value, gradient, Laplacian, Hessian,
    # divergence, a mixed partial and a first derivative.
    tops.value(state, "u")
    tops.gradient(state, "u")
    tops.laplacian(state, "u")
    tops.hessian(state, "u")
    tops.divergence(state, ("u", "v"))
    tops.mixed_partial(state, "u", ("x", "y"), (1, 1))
    tops.derivative(state, "v", axis="x", order=2)
    assert calls == [2], f"expected one order-2 jet, got {calls}"


def test_cache_is_per_state_and_reused_across_orders(coords, specs):
    cs, comps = specs
    field = build_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=5, depth=1, jet_order=3, seed=4,
    )
    s1 = field(coords)
    tops.laplacian(s1, "u")
    assert sorted(s1.extra[JET_CACHE_KEY]) == [3]
    tops.gradient(s1, "u")
    assert sorted(s1.extra[JET_CACHE_KEY]) == [3], "a cached order-3 jet must serve order 1"

    s2 = field(coords)
    assert JET_CACHE_KEY not in s2.extra or not s2.extra[JET_CACHE_KEY]


# -- Fourier-feature / SIREN variants ----------------------------------------- #


def test_fourier_feature_field_bands_and_exactness(coords, specs):
    cs, comps = specs
    field = build_fourier_feature_vector_field(
        coordinate_spec=cs, components=comps, num_features=4, hidden=6, depth=2,
        frequency_scale=(0.5, 2.0, 8.0), seed=0,
    )
    assert field.scales == (0.5, 2.0, 8.0)
    assert field.feature_dim == 2 * 4 * 3
    state = field(coords)
    ref = _autograd_partials(field, coords, "u", 2)
    lap = tops.laplacian(state, "u")
    expected = ref[(2, 0)] + ref[(0, 2)]
    # A high band amplifies second derivatives by ~(2 pi * 8)^2, so the comparison
    # has to be relative to that magnitude rather than absolute.
    scale = float(expected.detach().abs().max())
    assert torch.allclose(lap, expected, rtol=1e-10, atol=1e-10 * scale)


def test_trainable_features_are_parameters(specs):
    cs, comps = specs
    fixed = FourierFeatureVectorField(
        coordinate_spec=cs, components=comps, num_features=3, hidden=4, depth=1,
    )
    train = FourierFeatureVectorField(
        coordinate_spec=cs, components=comps, num_features=3, hidden=4, depth=1,
        trainable_features=True,
    )
    assert not any(p is fixed.net.W_ff for p in fixed.parameters())
    assert any(p is train.net.W_ff for p in train.parameters())


def test_siren_field_high_order(coords, specs):
    cs, comps = specs
    field = make_siren_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, omega_0=5.0,
        jet_order=3, seed=0,
    )
    state = field(coords)
    ref = _autograd_partials(field, coords, "u", 3)
    got = tops.derivative(state, "u", axis="x", order=3)
    assert torch.allclose(got, ref[(3, 0)], rtol=1e-11, atol=1e-11)


# -- training / plumbing ------------------------------------------------------ #


def test_gradients_flow_to_parameters(coords, specs):
    cs, comps = specs
    field = build_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=6, depth=2, seed=0,
    )
    loss = tops.laplacian(field(coords), "u").pow(2).mean()
    loss.backward()
    grads = [p.grad for p in field.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().max() > 0 for g in grads if g is not None)


def test_dtype_and_dispatch_tag(specs):
    cs, comps = specs
    field = build_jet_mlp_vector_field(
        coordinate_spec=cs, components=comps, hidden=4, depth=1,
    )
    assert field._omnibias_dispatch == "jet_mlp"
    assert all(p.dtype is torch.float64 for p in field.parameters())
    assert "JetMLPVectorField" in repr(field)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"jet_order": 0}, "jet_order must be >= 1"),
        # arctan's fast path caps at order 2, so a field promising order-3
        # residuals must be rejected at construction, not deep in the kernel.
        ({"base": "arctan", "jet_order": 3}, "does not support order 3"),
    ],
)
def test_construction_rejects_bad_config(specs, kwargs, match):
    cs, comps = specs
    with pytest.raises(ValueError, match=match):
        JetMLPVectorField(
            coordinate_spec=cs, components=comps, hidden=4, depth=1, **kwargs
        )


def test_shape_mismatch_is_rejected(specs):
    from omnibias.torch.architectures.pinn import JetMLP

    cs, comps = specs
    with pytest.raises(ValueError, match="net.out_dim"):
        JetMLPVectorField(
            coordinate_spec=cs,
            components=comps,
            net=JetMLP(in_dim=2, hidden=4, out_dim=1, depth=1),
        )
