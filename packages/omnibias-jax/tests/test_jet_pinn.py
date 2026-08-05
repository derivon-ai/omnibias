# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deep, arbitrary-order omnibias PINN (`JetMLP`) -- JAX side.

Checks the closed-form multivariate-jet derivative path against ``jax`` autograd
(``jacfwd`` / ``hessian``) to machine precision, confirms the network is a proper
pytree (``jax.grad`` flows to the weight leaves), and that everything is
jit-compatible. Cross-backend bit-parity against torch lives in
``tests/test_jet_pinn_parity.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation  # noqa: E402
from omnibias.jax.architectures import JetMLP, make_jet_mlp  # noqa: E402


def _x(seed: int, b: int = 6, d: int = 2):
    return jax.random.normal(jax.random.PRNGKey(seed), (b, d), dtype=jnp.float64)


def _val(net: JetMLP):
    def val(xi):
        return net.value(xi).squeeze(-1)

    return val


def test_value_equals_jet_row0() -> None:
    net = make_jet_mlp(2, 16, 1, depth=3, seed=1)
    x = _x(2)
    jet = net.jet(x, 2)
    assert jnp.allclose(jet[:, 0, :], net.value(x), atol=1e-14)


def test_gradient_matches_autograd() -> None:
    net = make_jet_mlp(2, 16, 1, depth=3, seed=1)
    x = _x(2)
    g = net.gradient(x).squeeze(-1)
    g_ad = jax.vmap(jax.jacfwd(_val(net)))(x)
    assert jnp.allclose(g, g_ad, atol=1e-11)


def test_hessian_matches_autograd_and_symmetric() -> None:
    net = make_jet_mlp(2, 16, 1, depth=3, seed=1)
    x = _x(2)
    h = net.hessian(x).squeeze(-1)
    h_ad = jax.vmap(jax.hessian(_val(net)))(x)
    assert jnp.allclose(h, h_ad, atol=1e-10)
    assert jnp.allclose(h, jnp.transpose(h, (0, 2, 1)), atol=1e-12)


def test_third_order_matches_autograd() -> None:
    net = make_jet_mlp(2, 12, 1, depth=2, seed=3)
    x = _x(4, b=5)
    parts = net.partials(x, 3)
    val = _val(net)
    jf = jax.jacfwd
    t3 = jax.vmap(jf(jf(jf(val))))(x)
    assert jnp.allclose(parts[(3, 0)].squeeze(-1), t3[:, 0, 0, 0], atol=1e-9)
    assert jnp.allclose(parts[(2, 1)].squeeze(-1), t3[:, 0, 0, 1], atol=1e-9)
    assert jnp.allclose(parts[(1, 2)].squeeze(-1), t3[:, 0, 1, 1], atol=1e-9)
    assert jnp.allclose(parts[(0, 3)].squeeze(-1), t3[:, 1, 1, 1], atol=1e-9)


def test_value_grad_hessian_consistent() -> None:
    net = make_jet_mlp(2, 10, 1, depth=3, seed=5)
    x = _x(6)
    v, g, h = net.value_grad_hessian(x)
    assert jnp.allclose(v, net.value(x), atol=1e-14)
    assert jnp.allclose(g, net.gradient(x), atol=1e-13)
    assert jnp.allclose(h, net.hessian(x), atol=1e-13)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_depth_sweep_matches_autograd(depth: int) -> None:
    net = make_jet_mlp(2, 12, 1, depth=depth, seed=depth)
    x = _x(7, b=5)
    g = net.gradient(x).squeeze(-1)
    g_ad = jax.vmap(jax.jacfwd(_val(net)))(x)
    assert jnp.allclose(g, g_ad, atol=1e-10)


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "softplus"])
def test_activation_sweep_matches_autograd(activation: str) -> None:
    net = make_jet_mlp(2, 12, 1, depth=2, activation=activation, seed=2)
    x = _x(8, b=5)
    h = net.hessian(x).squeeze(-1)
    h_ad = jax.vmap(jax.hessian(_val(net)))(x)
    assert jnp.allclose(h, h_ad, atol=1e-9)


def test_multioutput_gradient_shape_and_parity() -> None:
    net = make_jet_mlp(3, 8, out_dim=2, depth=2, seed=4)
    x = _x(9, b=4, d=3)
    g = net.gradient(x)  # (B, in=3, out=2)
    assert g.shape == (4, 3, 2)
    j_ad = jax.vmap(jax.jacfwd(net.value))(x)  # (B, out=2, in=3)
    assert jnp.allclose(g, jnp.transpose(j_ad, (0, 2, 1)), atol=1e-11)


def test_partials_order2_equal_hessian() -> None:
    net = make_jet_mlp(2, 10, 1, depth=2, seed=11)
    x = _x(12, b=5)
    parts = net.partials(x, 2)
    h = net.hessian(x).squeeze(-1)
    assert jnp.allclose(parts[(2, 0)].squeeze(-1), h[:, 0, 0], atol=1e-12)
    assert jnp.allclose(parts[(1, 1)].squeeze(-1), h[:, 0, 1], atol=1e-12)
    assert jnp.allclose(parts[(0, 2)].squeeze(-1), h[:, 1, 1], atol=1e-12)


def test_pytree_grad_flows_to_weight_leaves() -> None:
    """The network is a pytree: ``jax.grad`` of a residual loss hits every weight."""
    net = make_jet_mlp(2, 12, 1, depth=3, seed=0, weight_init_scale=0.8)
    x = _x(1)

    def loss(n: JetMLP) -> jax.Array:
        v, g, h = n.value_grad_hessian(x)
        residual = g[:, 1, 0] - 0.1 * h[:, 0, 0, 0]  # heat residual u_t - a u_xx
        return jnp.mean(residual**2)

    grads = jax.grad(loss)(net)
    for w in grads.weights:
        assert jnp.all(jnp.isfinite(w))
    assert any(float(jnp.linalg.norm(w)) > 0 for w in grads.weights)


def test_jit_compatible() -> None:
    net = make_jet_mlp(2, 10, 1, depth=2, seed=2)
    x = _x(3, b=4)
    g_jit = jax.jit(lambda n, xx: n.gradient(xx))(net, x)
    assert jnp.allclose(g_jit, net.gradient(x), atol=1e-12)


def test_fastpath_none_raises() -> None:
    nofp = dataclasses.replace(get_activation("tanh"), name="tanh_nofp", fastpath=None)
    net = make_jet_mlp(2, 8, 1, depth=2, activation=nofp)
    with pytest.raises(ValueError):
        net.gradient(_x(0, b=3))


def test_make_jet_mlp_validation() -> None:
    with pytest.raises(ValueError):
        make_jet_mlp(0, 8, 1, depth=2)
    with pytest.raises(ValueError):
        make_jet_mlp(2, 8, 1, depth=0)


def test_depth_property() -> None:
    net = make_jet_mlp(2, 8, 1, depth=3, seed=0)
    assert net.depth == 3
    assert len(net.weights) == 4  # 3 hidden + 1 readout


def test_value_matches_numpy_reference() -> None:
    net = make_jet_mlp(2, 8, 1, depth=2, activation="tanh", seed=7)
    x = _x(8, b=4)
    h = np.asarray(x)
    ws = [np.asarray(w) for w in net.weights]
    bs = [np.asarray(b) for b in net.biases]
    for i in range(len(ws)):
        h = h @ ws[i].T + bs[i]
        if i < len(ws) - 1:
            h = np.tanh(h)
    assert np.allclose(np.asarray(net.value(x)), h, atol=1e-12)
