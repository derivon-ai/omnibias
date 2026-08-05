# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Asymptotic / removable boundary-condition losses (jax).

These build on the exact directional jet (``mlp_jet``) and the differentiable
L'Hopital limit (``lhopital_ratio``); we check the limit values against the raw
Taylor coefficients and confirm the losses backpropagate into the weights.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.jet import jet_to_tower, mlp_jet  # noqa: E402
from omnibias.pinn.jax.losses import (  # noqa: E402
    asymptotic_bc_loss,
    asymptotic_ratio,
    far_field_decay_loss,
    network_ray_jet,
)


def _build_mlp(seed: int = 0, dims=(2, 5, 4, 1), act: str = "tanh"):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        W = jnp.asarray(rng.normal(scale=0.6, size=(dout, din)))
        b = jnp.asarray(rng.normal(scale=0.4, size=(dout,)))
        spec = None if i == len(dims) - 2 else act
        layers.append((W, b, spec))
    x0 = jnp.asarray(rng.normal(size=(dims[0],)))
    v = jnp.asarray(rng.normal(size=(dims[0],)))
    return layers, x0, v


def test_network_ray_jet_matches_mlp_jet() -> None:
    layers, x0, v = _build_mlp(seed=1)
    got = network_ray_jet(layers, x0, v, order=3, out_index=0)
    full = mlp_jet(x0, v, layers, 3)
    assert got.shape == (4,)
    assert jnp.allclose(got, full[:, 0], rtol=1e-13, atol=1e-13)


def test_asymptotic_ratio_equals_leading_coefficients() -> None:
    # lhopital with den = t**rate reads exactly the rate-th Taylor coefficient.
    layers, x0, v = _build_mlp(seed=2)
    net_jet = network_ray_jet(layers, x0, v, order=2)
    for rate in (0, 1, 2):
        val = asymptotic_ratio(layers, x0, v, rate=rate, order=2)
        assert float(val) == pytest.approx(float(net_jet[rate]), abs=1e-13)


def test_rate0_is_value_and_rate1_is_directional_derivative() -> None:
    layers, x0, v = _build_mlp(seed=3)
    tower = jet_to_tower(mlp_jet(x0, v, layers, 1))  # tower[k] = d^k/dt^k N(x0+tv)
    value = asymptotic_ratio(layers, x0, v, rate=0)
    slope = asymptotic_ratio(layers, x0, v, rate=1)
    assert float(value) == pytest.approx(float(tower[0, 0]), abs=1e-12)
    assert float(slope) == pytest.approx(float(tower[1, 0]), abs=1e-12)


def test_asymptotic_bc_loss_zero_at_target() -> None:
    layers, x0, v = _build_mlp(seed=4)
    target = asymptotic_ratio(layers, x0, v, rate=1)
    loss = asymptotic_bc_loss(layers, x0, v, target=target, rate=1)
    assert float(loss) == pytest.approx(0.0, abs=1e-20)


def test_asymptotic_bc_loss_is_differentiable() -> None:
    layers, x0, v = _build_mlp(seed=5)
    W2, b2, _ = layers[-1]

    def loss_of_scale(a: jax.Array) -> jax.Array:
        scaled = layers[:-1] + [(a * W2, a * b2, None)]
        return asymptotic_bc_loss(scaled, x0, v, target=0.3, rate=1)

    g = float(jax.grad(loss_of_scale)(1.25))
    # finite-difference cross-check
    eps = 1e-5
    fd = (float(loss_of_scale(1.25 + eps)) - float(loss_of_scale(1.25 - eps))) / (2 * eps)
    assert g == pytest.approx(fd, rel=1e-5, abs=1e-7)
    assert abs(g) > 1e-8


def test_far_field_decay_loss_value_and_grad() -> None:
    layers, x0, v = _build_mlp(seed=6)
    order = 3
    net_jet = network_ray_jet(layers, x0, v, order=order)
    loss = far_field_decay_loss(layers, x0, v, order=order)
    assert float(loss) == pytest.approx(float(jnp.mean(net_jet * net_jet)), abs=1e-13)
    assert float(loss) > 0.0

    W2, b2, _ = layers[-1]

    def loss_of_scale(a: jax.Array) -> jax.Array:
        scaled = layers[:-1] + [(a * W2, a * b2, None)]
        return far_field_decay_loss(scaled, x0, v, order=order)

    # scaling the readout toward 0 must reduce the decay penalty
    assert float(loss_of_scale(0.1)) < float(loss_of_scale(1.0))
    assert abs(float(jax.grad(loss_of_scale)(1.0))) > 1e-8


def test_invalid_arguments_raise() -> None:
    layers, x0, v = _build_mlp(seed=7)
    with pytest.raises(ValueError, match="rate must be >= 0"):
        asymptotic_ratio(layers, x0, v, rate=-1)
    with pytest.raises(ValueError, match="must be >= rate"):
        asymptotic_ratio(layers, x0, v, rate=2, order=1)
    with pytest.raises(ValueError, match="order must be >= 0"):
        far_field_decay_loss(layers, x0, v, order=-1)
