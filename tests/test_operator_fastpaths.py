# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Affine-input jets and pruned readouts against independent nested autodiff."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@pytest.fixture(autouse=True)
def _enable_jit_for_transform_checks():
    """Do not inherit another test module's global JIT setting."""
    with jax.disable_jit(False):
        yield


def arrays(seed=17):
    rng = np.random.default_rng(seed)
    dims = (3, 4, 3, 2)
    return [
        (rng.normal(size=(b, a)) * 0.25, rng.normal(size=b) * 0.1)
        for a, b in zip(dims[:-1], dims[1:], strict=True)
    ]


@pytest.mark.parametrize("order", [0, 1, 4, 6])
@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "sin"])
def test_torch_affine_input_path_full_parameters_against_nested_jacfwd(order, activation):
    from omnibias.torch.jet import mlp_jet

    params = tuple(
        torch.tensor(a, dtype=torch.float64, requires_grad=True) for pair in arrays() for a in pair
    )
    x = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([0.3, 0.2, -0.4], dtype=torch.float64, requires_grad=True)
    act = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "sin": torch.sin}[activation]
    layers = [(params[2 * i], params[2 * i + 1], activation if i < 2 else None) for i in range(3)]

    def path(t):
        h = x + t * v
        for i in range(3):
            h = h @ params[2 * i].T + params[2 * i + 1]
            if i < 2:
                h = act(h)
        return h

    fn = path
    for _ in range(order):
        fn = torch.func.jacfwd(fn)
    reference = fn(torch.zeros((), dtype=x.dtype))
    actual = math.factorial(order) * mlp_jet(x, v, layers, order)[-1]
    torch.testing.assert_close(actual, reference, rtol=2e-10, atol=2e-11)
    left = torch.autograd.grad(
        (actual - 0.3).square().sum(), (*params, x, v), allow_unused=True, materialize_grads=True
    )
    right = torch.autograd.grad(
        (reference - 0.3).square().sum(), (*params, x, v), allow_unused=True, materialize_grads=True
    )
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("order", [0, 1, 4, 6])
def test_jax_affine_input_path_jit_vmap_and_parameters(order):
    from omnibias.jax.jet import mlp_jet

    params = tuple(jnp.asarray(a) for pair in arrays() for a in pair)
    x = jnp.asarray([0.2, -0.1, 0.3])
    v = jnp.asarray([0.3, 0.2, -0.4])

    def evaluate(x, v, params):
        return (
            math.factorial(order)
            * mlp_jet(
                x,
                v,
                [(params[2 * i], params[2 * i + 1], "tanh" if i < 2 else None) for i in range(3)],
                order,
            )[-1]
        )

    def reference(x, v, params):
        def path(t):
            h = x + t * v
            for i in range(3):
                h = h @ params[2 * i].T + params[2 * i + 1]
                if i < 2:
                    h = jnp.tanh(h)
            return h

        fn = path
        for _ in range(order):
            fn = jax.jacfwd(fn)
        return fn(jnp.asarray(0.0))

    np.testing.assert_allclose(
        jax.jit(evaluate)(x, v, params), reference(x, v, params), rtol=2e-10, atol=2e-11
    )
    left = jax.jit(jax.grad(lambda p: jnp.sum((evaluate(x, v, p) - 0.3) ** 2)))(params)
    right = jax.grad(lambda p: jnp.sum((reference(x, v, p) - 0.3) ** 2))(params)
    for a, b in zip(left, right, strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-10)
    xx = jnp.stack([x, x + 0.1])
    vv = jnp.stack([v, v * 0])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate, in_axes=(0, 0, None)))(xx, vv, params),
        jax.vmap(reference, in_axes=(0, 0, None))(xx, vv, params),
        rtol=2e-10,
        atol=2e-11,
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("affine_only", [False, True])
def test_laplacian_affine_readout_values_and_all_parameters(backend, affine_only):
    if backend == "torch":
        from omnibias.torch.laplacian import deep_field_laplacian

        params = tuple(
            torch.tensor(a, dtype=torch.float64, requires_grad=True)
            for pair in arrays()
            for a in pair
        )
        x = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
        layers = [
            (params[2 * i], params[2 * i + 1], None if affine_only or i == 2 else "tanh")
            for i in range(3)
        ]

        def field(y):
            for i, (w, b, _) in enumerate(layers):
                y = y @ w.T + b
                if not affine_only and i < 2:
                    y = torch.tanh(y)
            return y

        expected = torch.func.hessian(field)(x).diagonal(dim1=-2, dim2=-1).sum(-1)
        actual = deep_field_laplacian(x, layers)
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
        ga = torch.autograd.grad(
            (actual - 0.4).square().sum(), params, allow_unused=True, materialize_grads=True
        )
        if affine_only:
            assert all(torch.count_nonzero(a) == 0 for a in ga)
        else:
            gb = torch.autograd.grad(
                (expected - 0.4).square().sum(), params, allow_unused=True, materialize_grads=True
            )
            for a, b in zip(ga, gb, strict=True):
                torch.testing.assert_close(a, b, rtol=2e-9, atol=2e-10)
    else:
        from omnibias.jax.laplacian import deep_field_laplacian

        params = tuple(jnp.asarray(a) for pair in arrays() for a in pair)
        x = jnp.asarray([0.2, -0.1, 0.3])

        def actual(p):
            return deep_field_laplacian(
                x,
                [
                    (p[2 * i], p[2 * i + 1], None if affine_only or i == 2 else "tanh")
                    for i in range(3)
                ],
            )

        def expected(p):
            def field(y):
                for i in range(3):
                    y = y @ p[2 * i].T + p[2 * i + 1]
                    if not affine_only and i < 2:
                        y = jnp.tanh(y)
                return y

            return jnp.trace(jax.hessian(field)(x), axis1=-2, axis2=-1)

        np.testing.assert_allclose(
            jax.jit(actual)(params), expected(params), rtol=2e-10, atol=2e-11
        )
        ga = jax.jit(jax.grad(lambda p: jnp.sum((actual(p) - 0.4) ** 2)))(params)
        gb = jax.grad(lambda p: jnp.sum((expected(p) - 0.4) ** 2))(params)
        for a, b in zip(ga, gb, strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-10)
