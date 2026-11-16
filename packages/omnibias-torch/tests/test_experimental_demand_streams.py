# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The Torch stream engine serves the extended demand vocabulary; every quantity is checked against nested autodiff."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.core.demand import Demand  # noqa: E402
from omnibias.torch.experimental import (  # noqa: E402
    QuantitySpec,
    Residual,
    deep_residual_gram,
    streams_quantities,
    streams_residual,
)
from torch.func import hessian, jacfwd, jvp, vmap  # noqa: E402

D, C = 3, 2
DT = torch.float64


def _layers(seed=0, hidden=7, depth=2, out=C):
    rng = np.random.default_rng(seed)
    dims = [D] + [hidden] * depth + [out]
    return [
        (
            torch.as_tensor(rng.standard_normal((b, a)) / np.sqrt(a), dtype=DT),
            torch.as_tensor(0.2 * rng.standard_normal(b), dtype=DT),
            "tanh" if i < depth else None,
        )
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True))
    ]


def _forward(layers, x):
    h = x
    for W, b, spec in layers:
        h = W @ h + b
        if spec is not None:
            h = torch.tanh(h)
    return h


def _partial(f, alpha):
    for axis, n in enumerate(alpha):
        e = torch.eye(D, dtype=DT)[axis]
        for _ in range(n):
            f = (lambda g, e: lambda x: jvp(g, (x,), (e,))[1])(f, e)
    return f


def _points(B=5, seed=3):
    return torch.as_tensor(np.random.default_rng(seed).standard_normal((B, D)) * 0.6, dtype=DT)


def _close(a, b, rtol=1e-8, atol=1e-10):
    np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=rtol, atol=atol)


def test_hess_form_for_an_indefinite_symmetric_matrix() -> None:
    layers = _layers(2)
    x = _points()
    A = torch.as_tensor([[1.0, 0.5, -0.3], [0.5, -2.0, 0.7], [-0.3, 0.7, 0.4]], dtype=DT)
    res = Residual(lambda q: q.hess_form, ["hess_form"])
    q = streams_quantities(x, layers, res, QuantitySpec(hess_form=A.numpy()))
    ref = vmap(lambda p: torch.einsum("ij,cij->c", A, hessian(lambda z: _forward(layers, z))(p)))(x)
    _close(q.hess_form, ref, 1e-10, 1e-12)


@pytest.mark.parametrize(
    ("name", "alphas"),
    [
        ("mixed", ((1, 1, 0), (0, 1, 1), (2, 0, 0))),
        ("third", ((3, 0, 0), (1, 2, 0), (1, 1, 1))),
        ("fourth", ((4, 0, 0), (2, 2, 0), (2, 1, 1))),
    ],
)
def test_partials_match_nested_autodiff(name, alphas) -> None:
    layers = _layers(4)
    x = _points()
    res = Residual(lambda q: getattr(q, name), [name])
    got = getattr(streams_quantities(x, layers, res, QuantitySpec(**{name: alphas})), name)
    assert tuple(got.shape) == (x.shape[0], len(alphas), C)
    for r, alpha in enumerate(alphas):
        ref = vmap(_partial(lambda z: _forward(layers, z), alpha))(x)
        _close(got[:, r], ref)


def test_laplacian_gradient_and_bilaplacian() -> None:
    layers = _layers(5)
    x = _points(4)
    res = Residual(lambda q: q.lap_grad.sum(1) + q.bilap, ["lap_grad", "bilap", "third"])
    q = streams_quantities(x, layers, res, QuantitySpec(third=((3, 0, 0),)))

    def lap(z):
        return torch.diagonal(hessian(lambda w: _forward(layers, w))(z), dim1=-2, dim2=-1).sum(-1)

    _close(q.lap_grad, vmap(jacfwd(lap))(x).transpose(1, 2))
    bilap = vmap(lambda z: torch.diagonal(hessian(lap)(z), dim1=-2, dim2=-1).sum(-1))(x)
    _close(q.bilap, bilap)
    assert tuple(q.third.shape) == (4, 1, C) and q.lap is None and q.grad is None


def test_grad_metric_and_per_point_drift_directions() -> None:
    layers = _layers(6)
    x = _points(4)
    rng = np.random.default_rng(7)
    V = torch.as_tensor(rng.standard_normal((4, D, 2)), dtype=DT)
    res = Residual(lambda q: q.grad_sq + q.lap, ["grad", "grad_sq", "lap", "directional"])
    q = streams_quantities(x, layers, res, directions=V)
    f = lambda z: _forward(layers, z)  # noqa: E731
    jac = vmap(jacfwd(f))(x)  # (B, C, D)
    _close(q.grad, jac.transpose(1, 2), 1e-10)
    _close(q.grad_sq, (jac**2).sum(-1), 1e-10)
    lap = vmap(lambda z: torch.diagonal(hessian(f)(z), dim1=-2, dim2=-1).sum(-1))(x)
    _close(q.lap, lap, 1e-9, 1e-11)
    _close(q.directional, torch.einsum("bcd,bdm->bmc", jac, V), 1e-10)


def test_new_names_are_refused_by_the_gram_engine_and_vocabulary_is_validated() -> None:
    layers = _layers(1, out=1)
    x = _points(4)
    with pytest.raises(ValueError, match="cannot provide"):
        deep_residual_gram([(x, Residual(lambda q: q.third[:, 0, 0], ["third"]))], layers)
    with pytest.raises(ValueError, match="total order 3"):
        QuantitySpec(third=((2, 0, 0),))
    assert Demand.of("lap", "fourth").order == 4


def test_kdv_residual_gradients_match_the_dense_oracle() -> None:
    base = _layers(9, out=1)
    x = _points(6)
    res = Residual(
        lambda q: q.directional[:, 1, 0]
        + q.u[:, 0] * q.directional[:, 0, 0]
        + 0.0025 * q.third[:, 0, 0],
        ["directional", "third"],
    )
    spec = QuantitySpec(third=((3, 0, 0),))
    V = torch.eye(D, dtype=DT)[:, :2]

    def make(params):
        return [(W, b, s) for (W, b), (_, _, s) in zip(params, base, strict=True)]

    def streams_loss(params):
        r = streams_residual(x, make(params), res, spec, directions=V)
        return 0.5 * (r * r).sum()

    def dense_loss(params):
        layers = make(params)
        f = lambda z: _forward(layers, z)[0]  # noqa: E731
        u_xxx = _partial(lambda z: _forward(layers, z), (3, 0, 0))

        def row(z):
            g = torch.func.grad(f)(z)
            return g[1] + f(z) * g[0] + 0.0025 * u_xxx(z)[0]

        r = vmap(row)(x)
        return 0.5 * (r * r).sum()

    def grads(fn):
        params = [(W.clone().requires_grad_(), b.clone().requires_grad_()) for W, b, _ in base]
        value = fn(params)
        flat = [t for pair in params for t in pair]
        return value, torch.autograd.grad(value, flat)

    a, ga = grads(streams_loss)
    b, gb = grads(dense_loss)
    assert abs(float(a.detach()) - float(b.detach())) <= 1e-9 * abs(float(b.detach()))
    for u, v in zip(ga, gb, strict=True):
        _close(u, v, 1e-7, 1e-10)


# --------------------------------------------------------------------------- skip blocks and Fourier features
def _structured_layers(seed=11, m=4, out=C):
    from omnibias.torch.experimental import Skip, fourier_feature_layer

    rng = np.random.default_rng(seed)
    width = 2 * m
    t = lambda a: torch.as_tensor(a, dtype=DT)  # noqa: E731
    first = fourier_feature_layer(t(rng.standard_normal((m, D))), t(rng.standard_normal(m)))
    blocks = [
        (
            t(rng.standard_normal((width, width)) / np.sqrt(width)),
            t(0.1 * rng.standard_normal(width)),
            Skip("tanh"),
        )
        for _ in range(2)
    ]
    head = (t(rng.standard_normal((out, width)) / np.sqrt(width)), torch.zeros(out, dtype=DT), None)
    return [first, *blocks, head]


def _forward_structured(layers, x):
    from omnibias.torch.experimental import Skip

    h = x
    for W, b, spec in layers:
        z = W @ h + b
        if isinstance(spec, Skip):
            h = h + torch.tanh(z)
        elif spec == "sin":
            h = torch.sin(z)
        else:
            h = z
    return h


def test_fourier_feature_layer_is_the_sin_cos_embedding() -> None:
    from omnibias.torch.experimental import fourier_feature_layer

    rng = np.random.default_rng(0)
    freq = torch.as_tensor(rng.standard_normal((5, D)), dtype=DT)
    phase = torch.as_tensor(rng.standard_normal(5), dtype=DT)
    W, b, spec = fourier_feature_layer(freq, phase)
    x = _points(4)
    z = x @ freq.T + phase
    _close(torch.sin(x @ W.T + b), torch.cat([torch.sin(z), torch.cos(z)], dim=1), 1e-12, 1e-14)
    assert spec == "sin" and tuple(W.shape) == (10, D)


def test_skip_blocks_and_fourier_features_through_every_stream_program() -> None:
    from omnibias.torch.experimental import Skip

    layers = _structured_layers()
    x = _points(4)
    f = lambda z: _forward_structured(layers, z)  # noqa: E731
    res = Residual(lambda q: q.lap, ["grad", "lap", "third", "hess_form"])
    A = np.array([[1.0, 0.0, 0.4], [0.0, -1.0, 0.0], [0.4, 0.0, 0.5]])
    q = streams_quantities(x, layers, res, QuantitySpec(third=((3, 0, 0), (1, 1, 1)), hess_form=A))
    _close(q.grad, vmap(jacfwd(f))(x).transpose(1, 2), 1e-10, 1e-12)
    hess = vmap(hessian(f))(x)
    _close(q.lap, torch.diagonal(hess, dim1=-2, dim2=-1).sum(-1), 1e-9, 1e-11)
    _close(
        q.hess_form, torch.einsum("ij,bcij->bc", torch.as_tensor(A, dtype=DT), hess), 1e-9, 1e-11
    )
    for r, alpha in enumerate(((3, 0, 0), (1, 1, 1))):
        _close(q.third[:, r], vmap(_partial(f, alpha))(x))
    with pytest.raises(ValueError, match="first layer cannot be a Skip"):
        streams_quantities(
            x, [(layers[0][0], layers[0][1], Skip("sin"))], Residual(lambda q: q.lap, ["lap"])
        )
