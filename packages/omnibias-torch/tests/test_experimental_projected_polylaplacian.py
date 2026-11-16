# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Projected bi- and tri-Laplacian programs (PyTorch) against nested dense autodiff, values and parameter gradients."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.torch.experimental import (  # noqa: E402
    projected_deep_bilaplacian,
    projected_deep_trilaplacian,
)
from torch.func import hessian, jacfwd, vmap  # noqa: E402

DT = torch.float64


def _layers(seed, dims, acts):
    rng = np.random.default_rng(seed)
    return [
        (
            torch.as_tensor(rng.standard_normal((b, a)) / np.sqrt(a), dtype=DT),
            torch.as_tensor(0.2 * rng.standard_normal(b), dtype=DT),
            act,
        )
        for (a, b), act in zip(zip(dims[:-1], dims[1:], strict=True), acts, strict=True)
    ]


def _forward(layers, z):
    h = z
    for W, b, act in layers:
        h = W @ h + b
        if act == "tanh":
            h = torch.tanh(h)
        elif act == "sin":
            h = torch.sin(h)
    return h


def _lap(f, coords):
    def lap(z):
        hess = hessian(f)(z)
        idx = list(range(hess.shape[-1])) if coords is None else list(coords)
        return sum(hess[..., i, i] for i in idx)

    return lap


def _close(a, b, rtol=1e-8, atol=1e-10):
    np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dims", "acts", "coords"),
    [
        ([4, 6, 6, 2], ["tanh", "tanh", None], None),
        ([5, 4, 4, 1], ["sin", "tanh", None], None),  # D > H: the factor has k = H columns
        ([4, 5, 1], ["tanh", None], [0, 2, 3]),  # time-dependent field: the operator skips input 1
    ],
)
def test_bilaplacian_matches_nested_autodiff(dims, acts, coords) -> None:
    layers = _layers(0, dims, acts)
    x = torch.as_tensor(np.random.default_rng(1).standard_normal((3, dims[0])) * 0.7, dtype=DT)
    f = lambda z: _forward(layers, z)  # noqa: E731
    u, gs, lap, bi, dv = projected_deep_bilaplacian(x, layers, coordinates=coords)
    assert dv is None
    lap_f = _lap(f, coords)
    _close(u, vmap(f)(x))
    _close(lap, vmap(lap_f)(x), 1e-9, 1e-11)
    _close(bi, vmap(_lap(lap_f, coords))(x), 1e-7, 1e-9)
    jac = vmap(jacfwd(f))(x)
    cols = list(range(dims[0])) if coords is None else coords
    _close(gs, (jac[..., cols] ** 2).sum(-1), 1e-9, 1e-11)


def test_bilaplacian_drift_and_unbatched_points() -> None:
    layers = _layers(2, [4, 5, 5, 1], ["tanh", "tanh", None])
    x = torch.as_tensor(np.random.default_rng(3).standard_normal((3, 4)) * 0.7, dtype=DT)
    V = torch.as_tensor(np.random.default_rng(4).standard_normal((4, 2)), dtype=DT)
    f = lambda z: _forward(layers, z)  # noqa: E731
    _, _, _, _, dv = projected_deep_bilaplacian(x, layers, directions=V)
    _close(dv, torch.einsum("bcd,dm->bmc", vmap(jacfwd(f))(x), V), 1e-10, 1e-12)
    one = projected_deep_bilaplacian(x[0], layers)
    many = projected_deep_bilaplacian(x, layers)
    for a, b in zip(one[:4], many[:4], strict=True):
        _close(a, b[0], 1e-12, 1e-14)


def test_bilaplacian_parameter_gradients_match_nested_autodiff() -> None:
    base = _layers(5, [4, 5, 5, 1], ["tanh", "tanh", None])
    x = torch.as_tensor(np.random.default_rng(6).standard_normal((3, 4)) * 0.7, dtype=DT)

    def grads(program):
        params = [(W.clone().requires_grad_(), b.clone().requires_grad_()) for W, b, _ in base]
        layers = [(W, b, a) for (W, b), (_, _, a) in zip(params, base, strict=True)]
        loss = program(layers).pow(2).sum()
        # the last-layer bias drops out of every derivative of order >= 1: its gradient is exactly zero
        return torch.autograd.grad(
            loss, [t for pair in params for t in pair], allow_unused=True, materialize_grads=True
        )

    def structured(layers):
        return projected_deep_bilaplacian(x, layers)[3]

    def dense(layers):
        f = lambda z: _forward(layers, z)  # noqa: E731
        return vmap(_lap(_lap(f, None), None))(x)

    for a, b in zip(grads(structured), grads(dense), strict=True):
        _close(a, b, 1e-6, 1e-8)


@pytest.mark.parametrize("outer", ["projected", "coordinates"])
def test_trilaplacian_matches_nested_autodiff(outer) -> None:
    layers = _layers(7, [3, 5, 5, 1], ["tanh", "tanh", None])
    x = torch.as_tensor(np.random.default_rng(8).standard_normal((2, 3)) * 0.6, dtype=DT)
    f = lambda z: _forward(layers, z)  # noqa: E731
    u, _, lap, bi, tri, _ = projected_deep_trilaplacian(x, layers, outer=outer, chunk=2)
    lap_f = _lap(f, None)
    bi_f = _lap(lap_f, None)
    _close(lap, vmap(lap_f)(x), 1e-9, 1e-11)
    _close(bi, vmap(bi_f)(x), 1e-7, 1e-9)
    _close(tri, vmap(_lap(bi_f, None))(x), 1e-6, 1e-8)
    with pytest.raises(ValueError, match="outer must be"):
        projected_deep_trilaplacian(x, layers, outer="nonsense")


def test_trilaplacian_parameter_gradients_and_refusals() -> None:
    base = _layers(9, [3, 4, 4, 1], ["tanh", "tanh", None])
    x = torch.as_tensor(np.random.default_rng(10).standard_normal((2, 3)) * 0.6, dtype=DT)

    def grads(program):
        params = [(W.clone().requires_grad_(), b.clone().requires_grad_()) for W, b, _ in base]
        layers = [(W, b, a) for (W, b), (_, _, a) in zip(params, base, strict=True)]
        loss = (program(layers) / 100.0).pow(2).sum()
        # the last-layer bias drops out of every derivative of order >= 1: its gradient is exactly zero
        return torch.autograd.grad(
            loss, [t for pair in params for t in pair], allow_unused=True, materialize_grads=True
        )

    def dense(layers):
        f = lambda z: _forward(layers, z)  # noqa: E731
        return vmap(_lap(_lap(_lap(f, None), None), None))(x)

    for a, b in zip(
        grads(lambda layers: projected_deep_trilaplacian(x, layers)[4]), grads(dense), strict=True
    ):
        _close(a, b, 1e-5, 1e-8)
    with pytest.raises(ValueError, match="first layer weight"):
        projected_deep_bilaplacian(x[:, :2], base)
