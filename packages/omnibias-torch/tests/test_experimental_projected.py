# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch ports of the projected Laplacian, divergence, log-determinant and Woodbury solves.

Every program is checked against dense autograd on small chains (value, gradients into the parameters and into the
inputs), including the wide-input regime (``D > H``) that the first-layer projection is for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.experimental.projected import (
    input_factor,
    projected_deep_divergence,
    projected_deep_laplacian,
    projected_deep_logdet,
)
from omnibias.torch.experimental.woodbury import (
    residual_inverse_newton,
    residual_solve,
    residual_solve_transpose,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def make_layers(dims, acts, *, seed=0, scale=0.4, bias=True):
    g = torch.Generator().manual_seed(seed)
    layers = []
    for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        W = (
            torch.randn(n_out, n_in, generator=g, dtype=torch.float64) * scale / np.sqrt(n_in)
        ).requires_grad_()
        b = (
            (torch.randn(n_out, generator=g, dtype=torch.float64) * 0.1).requires_grad_()
            if bias
            else None
        )
        layers.append((W, b, acts[i]))
    return layers


def forward(x, layers):
    a = x
    for W, b, spec in layers:
        a = a @ W.T
        if b is not None:
            a = a + b
        if spec is not None:
            a = get_activation(spec).forward(a)
    return a


def points(n, d, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


@pytest.mark.parametrize("dim,width", [(3, 6), (12, 5)])
@pytest.mark.parametrize("acts", [("tanh", "tanh", None), ("silu", "sin", None)])
def test_laplacian_matches_dense_hessian(dim, width, acts):
    layers = make_layers([dim, width, width, 2], acts)
    x = points(4, dim)
    u, gs, lap, dv = projected_deep_laplacian(x, layers)
    assert dv is None
    for i in range(x.shape[0]):
        H = torch.func.jacrev(torch.func.jacrev(lambda z: forward(z, layers)))(x[i])
        J = torch.func.jacrev(lambda z: forward(z, layers))(x[i])
        assert torch.allclose(lap[i], torch.einsum("ckk->c", H), atol=1e-11)
        assert torch.allclose(gs[i], (J * J).sum(-1), atol=1e-11)
        assert torch.allclose(u[i], forward(x[i], layers), atol=1e-12)


def test_laplacian_diffusion_coordinates_directions_and_point_metric():
    dim, width = 9, 5
    layers = make_layers([dim, width, width, 1], ("tanh", "tanh", None))
    x = points(3, dim)
    coords = np.array([1, 3, 4, 7, 8, 0])
    S = torch.randn(len(coords), 3, dtype=torch.float64) * 0.5
    V = torch.randn(dim, 2, dtype=torch.float64)
    _, gs, lap, dv = projected_deep_laplacian(
        x, layers, diffusion=S, coordinates=coords, directions=V
    )
    embed = torch.zeros(dim, 3, dtype=torch.float64)
    embed[coords] = S
    for i in range(3):
        H = torch.func.hessian(lambda z: forward(z, layers)[0])(x[i])
        g = torch.func.grad(lambda z: forward(z, layers)[0])(x[i])
        assert torch.allclose(lap[i, 0], torch.trace(embed.T @ H @ embed), atol=1e-11)
        assert torch.allclose(gs[i, 0], ((embed.T @ g) ** 2).sum(), atol=1e-11)
        assert torch.allclose(dv[i, :, 0], V.T @ g, atol=1e-11)
    s = torch.rand(3, len(coords), dtype=torch.float64) + 0.5
    _, gs2, lap2, _ = projected_deep_laplacian(x, layers, point_diffusion=s, coordinates=coords)
    for i in range(3):
        H = torch.func.hessian(lambda z: forward(z, layers)[0])(x[i])
        Hs = H[np.ix_(coords, coords)]
        assert torch.allclose(lap2[i, 0], (s[i] ** 2 * torch.diagonal(Hs)).sum(), atol=1e-11)


def test_laplacian_gradients_reach_parameters_and_inputs():
    dim, width = 10, 4
    layers = make_layers([dim, width, width, 1], ("tanh", "tanh", None))
    x = points(3, dim).requires_grad_()
    _, _, lap, _ = projected_deep_laplacian(x, layers)
    params = [p for W, b, _ in layers for p in (W, b)]
    got = torch.autograd.grad(lap.sum(), [*params, x], allow_unused=True)

    def dense(z, layers):
        out = 0.0
        for i in range(z.shape[0]):
            H = torch.func.hessian(lambda q: forward(q, layers)[0])(z[i])
            out = out + torch.trace(H)
        return out

    want = torch.autograd.grad(dense(x, layers), [*params, x], allow_unused=True)
    for a, b in zip(got, want, strict=True):
        if a is None or b is None:  # the readout bias does not enter the Laplacian
            assert a is None and b is None or float(torch.abs(a if b is None else b).max()) == 0.0
            continue
        assert torch.allclose(a, b, atol=1e-9, rtol=1e-9)


def test_input_factor_rank_deficient_falls_back_to_a_valid_factor():
    g = torch.Generator().manual_seed(3)
    base = torch.randn(2, 8, generator=g, dtype=torch.float64)
    W1 = torch.cat([base, base[:1] * 2.0, base[1:] * -1.0], dim=0)  # rank 2, H = 4, D = 8
    F = input_factor(W1)
    assert F.shape == (4, 4)
    assert torch.allclose(F @ F.T, W1 @ W1.T, atol=1e-10)


@pytest.mark.parametrize("dim,width,depth", [(4, 7, 1), (4, 7, 2), (4, 7, 4), (9, 3, 3)])
def test_divergence_matches_jacobian_trace(dim, width, depth):
    acts = ["tanh"] * depth + [None]
    layers = make_layers([dim, *[width] * depth, dim], acts)
    x = points(5, dim)
    u, div = projected_deep_divergence(x, layers)
    for i in range(x.shape[0]):
        J = torch.func.jacrev(lambda z: forward(z, layers))(x[i])
        assert torch.allclose(div[i], torch.trace(J), atol=1e-11)
        assert torch.allclose(u[i], forward(x[i], layers), atol=1e-12)


def test_divergence_with_coordinate_pairing_and_no_hidden_layer():
    layers = make_layers([6, 4, 3], ("tanh", None))
    x = points(3, 6)
    coords = np.array([5, 1, 2])
    _, div = projected_deep_divergence(x, layers, coordinates=coords)
    for i in range(3):
        J = torch.func.jacrev(lambda z: forward(z, layers))(x[i])
        assert torch.allclose(div[i], sum(J[j, coords[j]] for j in range(3)), atol=1e-11)
    linear = make_layers([4, 4], (None,))
    _, div = projected_deep_divergence(points(2, 4), linear)
    assert torch.allclose(div, torch.trace(linear[0][0]).expand(2))


@pytest.mark.parametrize("dim,width,depth", [(4, 7, 2), (12, 5, 3), (3, 8, 1)])
def test_logdet_matches_slogdet_of_the_jacobian(dim, width, depth):
    acts = ["tanh"] * depth + [None]
    layers = make_layers([dim, *[width] * depth, dim], acts)
    x = points(4, dim)
    u, sign, logabs = projected_deep_logdet(x, layers)
    for i in range(x.shape[0]):
        J = torch.func.jacrev(lambda z: forward(z, layers))(x[i])
        s, la = torch.linalg.slogdet(torch.eye(dim, dtype=torch.float64) + J)
        assert torch.allclose(logabs[i], la, atol=1e-10)
        assert torch.equal(sign[i], s)


@pytest.mark.parametrize("dim,width", [(4, 9), (10, 4)])
@pytest.mark.parametrize("transpose", [False, True])
def test_woodbury_solve_matches_dense_solve(dim, width, transpose):
    layers = make_layers([dim, width, width, dim], ("tanh", "tanh", None))
    x = points(3, dim)
    rhs = points(3, dim, seed=7)
    u, y = (residual_solve_transpose if transpose else residual_solve)(x, layers, rhs)
    for i in range(3):
        J = torch.func.jacrev(lambda z: forward(z, layers))(x[i])
        M = torch.eye(dim, dtype=torch.float64) + (J.T if transpose else J)
        assert torch.allclose(y[i], torch.linalg.solve(M, rhs[i]), atol=1e-10)


def test_woodbury_solve_is_differentiable_and_newton_inverts_the_block():
    dim, width = 5, 8
    layers = make_layers([dim, width, dim], ("tanh", None), scale=0.3)
    x = points(2, dim).requires_grad_()
    rhs = points(2, dim, seed=9)
    _, y = residual_solve(x, layers, rhs)
    grads = torch.autograd.grad(y.sum(), [layers[0][0], x])
    assert all(torch.isfinite(g).all() for g in grads)
    target = x.detach() + forward(x.detach(), layers)
    solution, steps, residual = residual_inverse_newton(target, layers)
    assert torch.allclose(solution, x.detach(), atol=1e-9)
    assert float(residual.max()) < 1e-10 and int(steps) >= 1


def test_wrong_shapes_and_nonaffine_readout_are_refused():
    layers = make_layers([3, 4, 3], ("tanh", "tanh"))
    with pytest.raises(ValueError, match="affine"):
        projected_deep_divergence(points(2, 3), layers)
    with pytest.raises(ValueError, match="must have shape"):
        projected_deep_laplacian(points(2, 5), make_layers([3, 4, 1], ("tanh", None)))
    with pytest.raises(ValueError, match="pass either"):
        projected_deep_laplacian(
            points(2, 3),
            make_layers([3, 4, 1], ("tanh", None)),
            diffusion=torch.ones(3, dtype=torch.float64),
            point_diffusion=torch.ones(2, 3, dtype=torch.float64),
        )
