# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation suite for the torch *multivariate* Faà di Bruno jet kernel.

Oracles (float64): full mixed-partial tensor of a deep MLP vs nested
``torch.func.jacfwd``, gradient / Hessian extraction vs ``torch.func`` AD,
directional restriction vs the directional :func:`mlp_jet`, affine linearity per
coefficient, and the order-cap / order-requirement error paths.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.core.multi_index import multi_indices  # noqa: E402
from omnibias.torch.activations.registry import get_activation  # noqa: E402
from omnibias.torch.jet import jet_to_tower, mlp_jet  # noqa: E402
from omnibias.torch.jet_mv import (  # noqa: E402
    affine_jet_mv,
    identity_jet,
    jet_gradient,
    jet_hessian,
    jet_partials,
    mlp_jet_mv,
)
from torch.func import grad as func_grad  # noqa: E402
from torch.func import hessian as func_hessian  # noqa: E402
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


def _build_mlp(seed: int = 0, dims=(3, 5, 4, 1), act: str = "tanh"):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        W = torch.as_tensor(rng.normal(scale=0.6, size=(dout, din)))
        b = torch.as_tensor(rng.normal(scale=0.4, size=(dout,)))
        spec = None if i == len(dims) - 2 else get_activation(act)
        layers.append((W, b, spec))
    x0 = torch.as_tensor(rng.normal(size=(dims[0],)))
    return layers, x0


def _forward_scalar(layers):
    def f(x):
        z = x
        for W, b, spec in layers:
            z = W @ z + b
            if spec is not None:
                z = spec.forward(z)
        return z[0]

    return f


def _full_deriv_tensor(f, x0, n: int):
    fn = f
    for _ in range(n):
        fn = jacfwd(fn)
    return fn(x0)


def _partial_from_tensor(tensor, alpha: tuple[int, ...]) -> float:
    axes: list[int] = []
    for i, a in enumerate(alpha):
        axes.extend([i] * a)
    val = tensor
    for ax in axes:
        val = val[ax]
    return float(val)


# ----- full multi-index tensor vs nested jacfwd -----


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "gaussian", "cosh"])
def test_full_multi_index_matches_nested_jacfwd(act: str) -> None:
    layers, x0 = _build_mlp(seed=1, act=act)
    f = _forward_scalar(layers)
    dim, order = int(x0.shape[0]), 4

    jet = mlp_jet_mv(x0, layers, order)
    parts = jet_partials(jet, dim, order)

    tensors = {n: _full_deriv_tensor(f, x0, n) for n in range(order + 1)}
    for alpha in multi_indices(dim, order):
        ref = _partial_from_tensor(tensors[sum(alpha)], alpha)
        got = float(parts[alpha][0])
        assert got == pytest.approx(ref, rel=1e-9, abs=1e-10)


# ----- capstone: full multivariate Faà di Bruno, deep + high order + multi-output -----


def test_full_multivariate_capstone() -> None:
    layers, x0 = _build_mlp(seed=42, dims=(3, 6, 5, 4, 2), act="tanh")
    dim, order = int(x0.shape[0]), 5

    def f_vec(x):
        z = x
        for W, b, spec in layers:
            z = W @ z + b
            if spec is not None:
                z = spec.forward(z)
        return z  # (2,)

    jet = mlp_jet_mv(x0, layers, order)
    parts = jet_partials(jet, dim, order)
    tensors = {n: _full_deriv_tensor(f_vec, x0, n) for n in range(order + 1)}
    for alpha in multi_indices(dim, order):
        t = tensors[sum(alpha)]
        for c in range(2):
            ref = _partial_from_tensor(t[c], alpha)
            got = float(parts[alpha][c])
            assert got == pytest.approx(ref, rel=1e-8, abs=1e-9)


# ----- gradient / Hessian extraction -----


def test_gradient_and_hessian_match_autodiff() -> None:
    layers, x0 = _build_mlp(seed=2)
    f = _forward_scalar(layers)
    dim, order = int(x0.shape[0]), 3

    jet = mlp_jet_mv(x0, layers, order)
    g = jet_gradient(jet, dim, order)[:, 0]
    h = jet_hessian(jet, dim, order)[:, :, 0]

    assert torch.allclose(g, func_grad(f)(x0), rtol=1e-10, atol=1e-10)
    assert torch.allclose(h, func_hessian(f)(x0), rtol=1e-10, atol=1e-10)
    assert torch.allclose(h, h.T, rtol=0, atol=1e-12)


# ----- directional restriction ties mv to the directional kernel -----


def test_directional_restriction_matches_mlp_jet() -> None:
    layers, x0 = _build_mlp(seed=3)
    dim, order = int(x0.shape[0]), 5
    rng = np.random.default_rng(99)
    v = torch.as_tensor(rng.normal(size=(dim,)))

    jet = mlp_jet_mv(x0, layers, order)
    parts = jet_partials(jet, dim, order)

    idx = multi_indices(dim, order)
    v_np = np.asarray(v)
    by_degree: dict[int, torch.Tensor] = {}
    for alpha in idx:
        k = sum(alpha)
        afact = 1
        valpha = 1.0
        for i, a in enumerate(alpha):
            afact *= math.factorial(a)
            valpha *= v_np[i] ** a
        coeff = math.factorial(k) / afact * valpha
        contrib = parts[alpha] * coeff
        by_degree[k] = by_degree.get(k, torch.zeros_like(contrib)) + contrib

    directional = jet_to_tower(mlp_jet(x0, v, layers, order))
    for k in range(order + 1):
        assert torch.allclose(by_degree[k], directional[k], rtol=1e-9, atol=1e-10)


# ----- affine linearity per coefficient -----


def test_affine_jet_mv_linear_per_coefficient() -> None:
    rng = np.random.default_rng(7)
    dim, order = 2, 3
    m = len(multi_indices(dim, order))
    z_jet = torch.as_tensor(rng.normal(size=(m, 5)))
    W = torch.as_tensor(rng.normal(size=(3, 5)))
    b = torch.as_tensor(rng.normal(size=(3,)))
    out = affine_jet_mv(z_jet, W, b)
    assert torch.allclose(out[0], W @ z_jet[0] + b)
    for k in range(1, m):
        assert torch.allclose(out[k], W @ z_jet[k])


# ----- identity seed -----


def test_identity_jet_seed() -> None:
    x0 = torch.as_tensor([0.3, -1.1, 2.0])
    order = 2
    jet = identity_jet(x0, order)
    idx = multi_indices(3, order)
    assert torch.allclose(jet[0], x0)
    for i, alpha in enumerate(idx):
        if sum(alpha) == 0:
            assert torch.allclose(jet[i], x0)
        elif sum(alpha) == 1:
            j = alpha.index(1)
            expected = torch.as_tensor([1.0 if k == j else 0.0 for k in range(3)])
            assert torch.allclose(jet[i], expected)
        else:
            assert torch.allclose(jet[i], torch.zeros(3))


# ----- error paths -----


def test_order_cap_raises_value_error() -> None:
    layers, x0 = _build_mlp(seed=5)
    # arctan caps at order 2, so an order-3 jet must raise.
    bad = [(layers[0][0], layers[0][1], get_activation("arctan"))]
    with pytest.raises(ValueError, match="does not support order"):
        mlp_jet_mv(x0, bad, 3)


def test_gradient_requires_order_one() -> None:
    layers, x0 = _build_mlp(seed=6)
    jet = mlp_jet_mv(x0, layers, 0)
    with pytest.raises(ValueError, match="gradient needs order >= 1"):
        jet_gradient(jet, int(x0.shape[0]), 0)


def test_hessian_requires_order_two() -> None:
    layers, x0 = _build_mlp(seed=6)
    jet = mlp_jet_mv(x0, layers, 1)
    with pytest.raises(ValueError, match="hessian needs order >= 2"):
        jet_hessian(jet, int(x0.shape[0]), 1)
