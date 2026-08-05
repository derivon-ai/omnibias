# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation suite for the torch Faà di Bruno jet kernel.

Oracles (float64): single-layer reduction to the closed-form fast path, nested
``torch.func.jacfwd``, the Bell-polynomial decomposition vs the shifted-power
kernel, and the order-cap error path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.core.bell import faa_di_bruno_terms  # noqa: E402
from omnibias.torch.activations.registry import get_activation  # noqa: E402
from omnibias.torch.jet import (  # noqa: E402
    _path_jet,
    affine_jet,
    antiderivative_jet,
    compose_jet,
    derivative_jet,
    jet_to_tower,
    layer_jet,
    mlp_jet,
)
from torch.func import jacfwd  # noqa: E402


@pytest.fixture(autouse=True)
def _default_float64():
    """Run these float64 oracle tests without leaking the global default dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _build_mlp(seed: int = 0, dims=(3, 5, 4, 2), act: str = "tanh"):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        W = torch.as_tensor(rng.normal(scale=0.7, size=(dout, din)))
        b = torch.as_tensor(rng.normal(scale=0.5, size=(dout,)))
        spec = None if i == len(dims) - 2 else get_activation(act)
        layers.append((W, b, spec))
    x0 = torch.as_tensor(rng.normal(size=(dims[0],)))
    v = torch.as_tensor(rng.normal(size=(dims[0],)))
    return layers, x0, v


def _forward(layers):
    def f(x):
        z = x
        for W, b, spec in layers:
            z = W @ z + b
            if spec is not None:
                z = spec.forward(z)
        return z

    return f


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "gaussian", "cosh"])
def test_single_layer_reduces_to_fastpath(act: str) -> None:
    rng = np.random.default_rng(3)
    D, H = 4, 6
    W = torch.as_tensor(rng.normal(size=(H, D)))
    b = torch.as_tensor(rng.normal(size=(H,)))
    x0 = torch.as_tensor(rng.normal(size=(D,)))
    v = torch.as_tensor(rng.normal(size=(D,)))
    spec = get_activation(act)
    order = 6
    tower = jet_to_tower(layer_jet(_path_jet(x0, v, order), W, b, spec, order))

    z0 = W @ x0 + b
    wv = W @ v
    for k in range(order + 1):
        expected = spec.forward(z0) if k == 0 else spec.fastpath(z0, k) * wv**k
        assert torch.allclose(tower[k], expected, rtol=1e-12, atol=1e-12)


def test_deep_mlp_matches_nested_jacfwd() -> None:
    layers, x0, v = _build_mlp(seed=2)
    f = _forward(layers)
    order = 6

    def g(t):
        return f(x0 + t * v)

    def kth(k):
        fn = g
        for _ in range(k):
            fn = jacfwd(fn)
        return fn(torch.tensor(0.0))

    tower = jet_to_tower(mlp_jet(x0, v, layers, order))
    for k in range(order + 1):
        assert torch.allclose(tower[k], kth(k), rtol=1e-9, atol=1e-9)


def test_compose_jet_matches_bell_oracle() -> None:
    rng = np.random.default_rng(11)
    order = 6
    u_jet = torch.as_tensor(rng.normal(size=(order + 1, 3)))
    sigma_tower = torch.as_tensor(rng.normal(size=(order + 1, 3)))
    out = jet_to_tower(compose_jet(u_jet, sigma_tower)).numpy()

    u_jet_np = u_jet.numpy()
    sigma_np = sigma_tower.numpy()
    u_deriv = np.array([math.factorial(i) * u_jet_np[i] for i in range(order + 1)])
    expected = np.empty_like(sigma_np)
    expected[0] = sigma_np[0]
    for n in range(1, order + 1):
        acc = np.zeros(u_jet_np.shape[1])
        for k, exps, coeff in faa_di_bruno_terms(n):
            prod = np.ones(u_jet_np.shape[1])
            for i, e in enumerate(exps, start=1):
                if e:
                    prod = prod * u_deriv[i] ** e
            acc = acc + coeff * sigma_np[k] * prod
        expected[n] = acc
    assert np.allclose(out, expected, rtol=1e-10, atol=1e-10)


def test_order_cap_raises_value_error() -> None:
    rng = np.random.default_rng(5)
    D, H = 3, 4
    W = torch.as_tensor(rng.normal(size=(H, D)))
    b = torch.as_tensor(rng.normal(size=(H,)))
    x0 = torch.as_tensor(rng.normal(size=(D,)))
    v = torch.as_tensor(rng.normal(size=(D,)))
    # arctan caps at order 2, so an order-3 jet must raise.
    with pytest.raises(ValueError, match="does not support order"):
        layer_jet(_path_jet(x0, v, 3), W, b, "arctan", 3)


def test_affine_jet_is_linear_per_order() -> None:
    rng = np.random.default_rng(7)
    order = 4
    z_jet = torch.as_tensor(rng.normal(size=(order + 1, 5)))
    W = torch.as_tensor(rng.normal(size=(3, 5)))
    b = torch.as_tensor(rng.normal(size=(3,)))
    out = affine_jet(z_jet, W, b)
    assert torch.allclose(out[0], W @ z_jet[0] + b)
    for k in range(1, order + 1):
        assert torch.allclose(out[k], W @ z_jet[k])


# ----- two-sided (integral) tower: antiderivative_jet / derivative_jet -----


def test_antiderivative_jet_ftc_part1_roundtrip() -> None:
    """FTC part 1: d/dt of the antiderivative jet recovers the original vector jet."""
    layers, x0, v = _build_mlp(seed=4)
    order = 6
    jet = mlp_jet(x0, v, layers, order)  # (order+1, C)
    anti = antiderivative_jet(jet, constant=0.3)
    assert anti.shape[0] == jet.shape[0] + 1
    assert torch.allclose(anti[0], torch.full_like(jet[0], 0.3))
    back = derivative_jet(anti)
    assert back.shape == jet.shape
    assert torch.allclose(back, jet, rtol=1e-12, atol=1e-12)


def test_antiderivative_jet_of_mlp_jet_matches_mpmath() -> None:
    mpmath = pytest.importorskip("mpmath")
    a, c = 1.1, -0.3
    order = 6
    jet = mlp_jet(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        [(torch.tensor([[a]]), torch.tensor([c]), "tanh")],
        order,
    )  # net(t) = tanh(a t + c)
    anti = antiderivative_jet(jet)  # F(t) = int_0^t net, F(0) = 0

    def big_f(t: object) -> object:
        return (mpmath.log(mpmath.cosh(a * t + c)) - mpmath.log(mpmath.cosh(c))) / a

    with mpmath.workdps(50):
        taylor_f = mpmath.taylor(big_f, 0.0, order + 1)
    got = anti[:, 0].tolist()
    for k in range(order + 2):
        assert abs(got[k] - float(taylor_f[k])) <= 1e-10


def test_derivative_jet_scalar_matches_manual() -> None:
    jet = torch.tensor([2.0, 3.0, 5.0, 7.0])  # a_0 .. a_3
    d = derivative_jet(jet)  # (k+1) a_{k+1}
    assert torch.allclose(d, torch.tensor([3.0, 10.0, 21.0]))
