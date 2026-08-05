# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Spectral-bias-mitigating omnibias PINNs (JAX): ``FourierFeatureMLP`` / ``make_siren``.

Checks the closed-form multivariate-jet derivatives of the sin-encoded Fourier-feature
net and of the SIREN against ``jax`` autograd (``jacfwd`` / ``hessian``), confirms both
are jit-compatible proper pytrees, and demonstrates the representational gain that
underlies spectral-bias mitigation. Cross-backend bit-parity vs torch lives in
``tests/test_spectral_pinn_parity.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation  # noqa: E402
from omnibias.jax.architectures import (  # noqa: E402
    FourierFeatureMLP,
    JetMLP,
    make_fourier_feature_mlp,
    make_siren,
)


def _x(seed: int, b: int = 6, d: int = 2) -> jnp.ndarray:
    return jax.random.normal(jax.random.PRNGKey(seed), (b, d), dtype=jnp.float64)


def _val(net: FourierFeatureMLP | JetMLP):
    def val(xi: jnp.ndarray) -> jnp.ndarray:
        return net.value(xi).reshape(())

    return val


# --- FourierFeatureMLP: exact derivatives ---------------------------------


def test_fourier_value_equals_jet_row0() -> None:
    net = make_fourier_feature_mlp(2, num_features=8, hidden=16, depth=2, seed=1)
    x = _x(2)
    jet = net.jet(x, 2)
    assert jnp.allclose(jet[:, 0, :], net.value(x), atol=1e-13)


def test_fourier_encoding_is_cos_sin() -> None:
    net = make_fourier_feature_mlp(3, num_features=5, hidden=8, depth=1, seed=2)
    x = _x(3, b=4, d=3)
    f_total = net.num_features * len(net.scales)
    b_mat = net.w_ff[:f_total]
    z = x @ b_mat.T
    feats = jnp.sin(x @ net.w_ff.T + net.b_ff)
    assert jnp.allclose(feats[:, :f_total], jnp.cos(z), atol=1e-13)
    assert jnp.allclose(feats[:, f_total:], jnp.sin(z), atol=1e-13)


def test_fourier_gradient_matches_autograd() -> None:
    net = make_fourier_feature_mlp(2, num_features=8, hidden=16, depth=2, seed=3)
    x = _x(4, b=7)
    g = net.gradient(x).squeeze(-1)
    g_ad = jax.vmap(jax.jacfwd(_val(net)))(x)
    assert jnp.allclose(g, g_ad, atol=1e-10)


def test_fourier_hessian_matches_autograd() -> None:
    net = make_fourier_feature_mlp(2, num_features=8, hidden=16, depth=2, seed=4)
    x = _x(5, b=5)
    h = net.hessian(x).squeeze(-1)
    h_ad = jax.vmap(jax.hessian(_val(net)))(x)
    assert jnp.allclose(h, h_ad, atol=1e-9)
    assert jnp.allclose(h, jnp.transpose(h, (0, 2, 1)), atol=1e-11)


def test_fourier_third_order_matches_autograd() -> None:
    net = make_fourier_feature_mlp(2, num_features=6, hidden=12, depth=2, seed=5)
    x = _x(6, b=4)
    parts = net.partials(x, 3)
    jf = jax.jacfwd
    t3 = jax.vmap(jf(jf(jf(_val(net)))))(x)
    assert jnp.allclose(parts[(3, 0)].squeeze(-1), t3[:, 0, 0, 0], atol=1e-8)
    assert jnp.allclose(parts[(2, 1)].squeeze(-1), t3[:, 0, 0, 1], atol=1e-8)
    assert jnp.allclose(parts[(0, 3)].squeeze(-1), t3[:, 1, 1, 1], atol=1e-8)


@pytest.mark.parametrize("scales", [1.0, (0.5, 2.0), (0.25, 1.0, 4.0)])
def test_fourier_multiscale_shapes_and_grad(scales: float | tuple[float, ...]) -> None:
    net = make_fourier_feature_mlp(
        2, num_features=4, hidden=10, depth=2, frequency_scale=scales, seed=6
    )
    n_bands = 1 if isinstance(scales, float) else len(scales)
    assert net.feature_dim == 2 * 4 * n_bands
    x = _x(7, b=5)
    g = net.gradient(x).squeeze(-1)
    g_ad = jax.vmap(jax.jacfwd(_val(net)))(x)
    assert jnp.allclose(g, g_ad, atol=1e-10)


def test_fourier_depth0_pure_rff_matches_autograd() -> None:
    net = make_fourier_feature_mlp(1, num_features=16, depth=0, frequency_scale=3.0, seed=7)
    x = _x(8, b=6, d=1)
    g = net.gradient(x).reshape(-1)
    g_ad = jax.vmap(jax.jacfwd(_val(net)))(x).reshape(-1)
    assert jnp.allclose(g, g_ad, atol=1e-9)


def test_fourier_jit_compatible() -> None:
    net = make_fourier_feature_mlp(2, num_features=6, hidden=10, depth=2, seed=8)
    x = _x(9, b=4)
    g_jit = jax.jit(lambda n, xx: n.gradient(xx))(net, x)
    assert jnp.allclose(g_jit, net.gradient(x), atol=1e-12)


def test_fourier_pytree_grad_flows() -> None:
    net = make_fourier_feature_mlp(2, num_features=6, hidden=10, depth=2, seed=9)
    x = _x(10, b=4)

    def loss(n: FourierFeatureMLP) -> jax.Array:
        v, g, h = n.value_grad_hessian(x)
        return jnp.mean((g[:, 1, 0] - 0.1 * h[:, 0, 0, 0]) ** 2)

    grads = jax.grad(loss)(net)
    assert jnp.all(jnp.isfinite(grads.w_ff))
    for w in grads.weights:
        assert jnp.all(jnp.isfinite(w))


def test_fourier_base_without_fastpath_raises() -> None:
    import dataclasses

    nofp = dataclasses.replace(get_activation("tanh"), name="tanh_nofp", fastpath=None)
    net = make_fourier_feature_mlp(2, num_features=4, hidden=8, depth=2, base=nofp)
    with pytest.raises(ValueError):
        net.gradient(_x(0, b=3))


def test_make_fourier_feature_mlp_validation() -> None:
    with pytest.raises(ValueError):
        make_fourier_feature_mlp(0, num_features=4)
    with pytest.raises(ValueError):
        make_fourier_feature_mlp(2, num_features=0)
    with pytest.raises(ValueError):
        make_fourier_feature_mlp(2, num_features=4, depth=-1)
    with pytest.raises(ValueError):
        make_fourier_feature_mlp(2, num_features=4, frequency_scale=-1.0)


# --- SIREN: exact derivatives ---------------------------------------------


def test_siren_gradient_matches_autograd() -> None:
    net = make_siren(1, hidden=24, depth=3, omega_0=10.0, seed=2)
    x = _x(3, b=6, d=1)
    g = net.gradient(x).reshape(-1)
    g_ad = jax.vmap(jax.jacfwd(_val(net)))(x).reshape(-1)
    assert jnp.allclose(g, g_ad, atol=1e-10)


def test_siren_high_order_matches_autograd() -> None:
    net = make_siren(2, hidden=16, depth=2, omega_0=8.0, seed=3)
    x = _x(4, b=4)
    parts = net.partials(x, 4)
    jf = jax.jacfwd
    t4 = jax.vmap(jf(jf(jf(jf(_val(net))))))(x)
    assert jnp.allclose(parts[(4, 0)].squeeze(-1), t4[:, 0, 0, 0, 0], atol=1e-7)
    assert jnp.allclose(parts[(2, 2)].squeeze(-1), t4[:, 0, 0, 1, 1], atol=1e-7)


def test_make_siren_validation() -> None:
    with pytest.raises(ValueError):
        make_siren(1, hidden=8, depth=0)
    with pytest.raises(ValueError):
        make_siren(1, hidden=8, depth=2, omega_0=0.0)


# --- Spectral capacity: measured (optimizer-free) evidence ----------------


def _target_high_freq(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(2 * math.pi * 5.0 * x) + 0.3 * jnp.sin(2 * math.pi * 1.0 * x)


def test_fourier_features_represent_high_frequency() -> None:
    """A linear fit on the random Fourier features reconstructs a 5-cycle target that a
    comparable random non-periodic (tanh) feature map of the same width cannot."""
    net = make_fourier_feature_mlp(1, num_features=64, depth=0, frequency_scale=5.0, seed=0)
    x_tr = jnp.linspace(0.0, 1.0, 256)[:, None]
    y_tr = _target_high_freq(x_tr)
    x_te = jnp.linspace(0.0, 1.0, 512)[:, None]
    y_te = _target_high_freq(x_te)

    feats = jnp.sin(x_tr @ net.w_ff.T + net.b_ff)
    w_ff, _, _, _ = jnp.linalg.lstsq(feats, y_tr, rcond=None)
    feats_te = jnp.sin(x_te @ net.w_ff.T + net.b_ff)
    mse_ff = float(jnp.mean((feats_te @ w_ff - y_te) ** 2))

    w_rand = jax.random.normal(jax.random.PRNGKey(1), (2 * 64, 1))
    b_rand = jax.random.normal(jax.random.PRNGKey(2), (2 * 64,))
    tf = jnp.tanh(x_tr @ w_rand.T + b_rand)
    w_t, _, _, _ = jnp.linalg.lstsq(tf, y_tr, rcond=None)
    tf_te = jnp.tanh(x_te @ w_rand.T + b_rand)
    mse_tanh = float(jnp.mean((tf_te @ w_t - y_te) ** 2))

    assert mse_ff < 1e-8
    assert mse_ff < 0.01 * mse_tanh


def test_value_matches_numpy_reference() -> None:
    net = make_fourier_feature_mlp(2, num_features=5, hidden=8, depth=2, base="tanh", seed=7)
    x = _x(8, b=4)
    h = np.asarray(x)
    h = np.sin(h @ np.asarray(net.w_ff).T + np.asarray(net.b_ff))
    ws = [np.asarray(w) for w in net.weights]
    bs = [np.asarray(b) for b in net.biases]
    for i in range(len(ws)):
        h = h @ ws[i].T + bs[i]
        if i < len(ws) - 1:
            h = np.tanh(h)
    assert np.allclose(np.asarray(net.value(x)), h, atol=1e-12)
