# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation suite for the jax Faà di Bruno jet kernel.

Oracles (float64):

* single-layer reduction to the existing closed-form fast path,
* :mod:`jax.experimental.jet` Taylor-mode autodiff,
* nested ``jax.jacfwd``,
* the explicit Bell-polynomial decomposition
  (:func:`omnibias.core.bell.faa_di_bruno_terms`) vs the shifted-power kernel,
* the order-cap error path for a bounded-order activation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.experimental import jet as jax_jet  # noqa: E402
from omnibias.core.bell import faa_di_bruno_terms  # noqa: E402
from omnibias.jax.activations import get_activation  # noqa: E402
from omnibias.jax.jet import (  # noqa: E402
    _path_jet,
    affine_jet,
    antiderivative_jet,
    compose_jet,
    derivative_jet,
    jet_to_tower,
    layer_jet,
    mlp_jet,
)


def _build_mlp(seed: int = 0, dims=(3, 5, 4, 2), act: str = "tanh"):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        W = jnp.asarray(rng.normal(scale=0.7, size=(dout, din)))
        b = jnp.asarray(rng.normal(scale=0.5, size=(dout,)))
        spec = None if i == len(dims) - 2 else get_activation(act)
        layers.append((W, b, spec))
    x0 = jnp.asarray(rng.normal(size=(dims[0],)))
    v = jnp.asarray(rng.normal(size=(dims[0],)))
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


# ----- single-layer analytic reduction -----


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "gaussian", "sin"])
def test_single_layer_reduces_to_fastpath(act: str) -> None:
    rng = np.random.default_rng(3)
    D, H = 4, 6
    W = jnp.asarray(rng.normal(size=(H, D)))
    b = jnp.asarray(rng.normal(size=(H,)))
    x0 = jnp.asarray(rng.normal(size=(D,)))
    v = jnp.asarray(rng.normal(size=(D,)))
    spec = get_activation(act)
    order = 6
    x_jet = _path_jet(x0, v, order)
    tower = jet_to_tower(layer_jet(x_jet, W, b, spec, order))  # (order+1, H)

    z0 = W @ x0 + b
    wv = W @ v
    for k in range(order + 1):
        expected = spec.forward(z0) if k == 0 else spec.fastpath(z0, k) * wv**k
        assert jnp.allclose(tower[k], expected, rtol=1e-12, atol=1e-12)


# ----- deep MLP vs jax.experimental.jet -----


def test_deep_mlp_matches_experimental_jet() -> None:
    layers, x0, v = _build_mlp(seed=1)
    f = _forward(layers)
    order = 6
    series = [v] + [jnp.zeros_like(v) for _ in range(order - 1)]
    y0, terms = jax_jet.jet(f, (x0,), (series,))
    expected = [y0, *terms]  # derivative tower d^k/dt^k f
    tower = jet_to_tower(mlp_jet(x0, v, layers, order))
    for k in range(order + 1):
        assert jnp.allclose(tower[k], expected[k], rtol=1e-10, atol=1e-10)


# ----- deep MLP vs nested jacfwd -----


def test_deep_mlp_matches_nested_jacfwd() -> None:
    layers, x0, v = _build_mlp(seed=2)
    f = _forward(layers)
    order = 6

    def g(t):
        return f(x0 + t * v)

    def kth(k):
        fn = g
        for _ in range(k):
            fn = jax.jacfwd(fn)
        return fn(0.0)

    tower = jet_to_tower(mlp_jet(x0, v, layers, order))
    for k in range(order + 1):
        assert jnp.allclose(tower[k], kth(k), rtol=1e-9, atol=1e-9)


# ----- Bell-polynomial decomposition vs shifted-power kernel -----


def test_compose_jet_matches_bell_oracle() -> None:
    rng = np.random.default_rng(11)
    order = 6
    # Arbitrary pre-activation jet and an arbitrary "activation" derivative tower
    # (decoupled from any real sigma, so this tests the composition algebra).
    u_jet = jnp.asarray(rng.normal(size=(order + 1, 3)))
    sigma_tower = jnp.asarray(rng.normal(size=(order + 1, 3)))
    out = jet_to_tower(compose_jet(u_jet, sigma_tower))  # derivative tower of b

    # Independent Bell-polynomial re-summation.
    u_jet_np = np.asarray(u_jet)
    sigma_np = np.asarray(sigma_tower)
    u_deriv = np.array(
        [math.factorial(i) * u_jet_np[i] for i in range(order + 1)]
    )  # u^(i)
    expected = np.empty_like(sigma_np)
    expected[0] = sigma_np[0]  # b = sigma(u0)
    for n in range(1, order + 1):
        acc = np.zeros(u_jet_np.shape[1])
        for k, exps, coeff in faa_di_bruno_terms(n):
            prod = np.ones(u_jet_np.shape[1])
            for i, e in enumerate(exps, start=1):
                if e:
                    prod = prod * u_deriv[i] ** e
            acc = acc + coeff * sigma_np[k] * prod
        expected[n] = acc
    assert np.allclose(np.asarray(out), expected, rtol=1e-10, atol=1e-10)


# ----- order-cap error path -----


def test_order_cap_raises_value_error() -> None:
    rng = np.random.default_rng(5)
    D, H = 3, 4
    W = jnp.asarray(rng.normal(size=(H, D)))
    b = jnp.asarray(rng.normal(size=(H,)))
    x0 = jnp.asarray(rng.normal(size=(D,)))
    v = jnp.asarray(rng.normal(size=(D,)))
    # arctan caps at order 2; an order-3 jet must fail loudly.
    with pytest.raises(ValueError, match="does not support order"):
        layer_jet(_path_jet(x0, v, 3), W, b, "arctan", 3)


def test_affine_jet_is_linear_per_order() -> None:
    rng = np.random.default_rng(7)
    order = 4
    z_jet = jnp.asarray(rng.normal(size=(order + 1, 5)))
    W = jnp.asarray(rng.normal(size=(3, 5)))
    b = jnp.asarray(rng.normal(size=(3,)))
    out = affine_jet(z_jet, W, b)
    assert jnp.allclose(out[0], W @ z_jet[0] + b)
    for k in range(1, order + 1):
        assert jnp.allclose(out[k], W @ z_jet[k])


# ----- two-sided (integral) tower: antiderivative_jet / derivative_jet -----


def test_antiderivative_jet_ftc_part1_roundtrip() -> None:
    layers, x0, v = _build_mlp(seed=4)
    order = 6
    jet = mlp_jet(x0, v, layers, order)  # (order+1, C)
    anti = antiderivative_jet(jet, constant=0.3)
    assert anti.shape[0] == jet.shape[0] + 1
    assert jnp.allclose(anti[0], jnp.full_like(jet[0], 0.3))
    back = derivative_jet(anti)
    assert back.shape == jet.shape
    assert jnp.allclose(back, jet, rtol=1e-12, atol=1e-12)


def test_antiderivative_derivative_jit_vmap_safe() -> None:
    rng = np.random.default_rng(9)
    jet = jnp.asarray(rng.normal(size=(6, 3)))
    a1 = jax.jit(antiderivative_jet)(jet)
    a2 = antiderivative_jet(jet)
    assert jnp.array_equal(a1, a2)
    d1 = jax.jit(derivative_jet)(jet)
    assert jnp.array_equal(d1, derivative_jet(jet))
    batch = jnp.stack([jet, jet * 2.0])
    av = jax.vmap(antiderivative_jet)(batch)
    assert av.shape == (2, 7, 3)


def test_antiderivative_jet_torch_parity() -> None:
    """Bit-identical antiderivative / derivative jets across the jax and torch twins."""
    torch = pytest.importorskip("torch")
    from omnibias.torch.jet import antiderivative_jet as t_anti
    from omnibias.torch.jet import derivative_jet as t_deriv

    rng = np.random.default_rng(13)
    arr = rng.normal(size=(6, 4))
    jx = jnp.asarray(arr)
    tx = torch.as_tensor(arr, dtype=torch.float64)

    aj = np.asarray(antiderivative_jet(jx, constant=0.7))
    at = t_anti(tx, constant=0.7).numpy()
    assert np.array_equal(aj, at)

    dj = np.asarray(derivative_jet(jx))
    dt = t_deriv(tx).numpy()
    assert np.array_equal(dj, dt)
