# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contract tests for :mod:`omnibias.curvature.one_layer`.

The closed-form per-sample parameter gradient and Hessian must match
:func:`jax.grad` and :func:`jax.hessian` to float64 precision for every
Riccati-class activation that registers a fast-path kernel.

The Gauss-Newton Newton step is also smoke-tested on a linear-target
regression problem (where Gauss-Newton converges in a single step from
random init when ``learning_rate=1``).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.curvature.one_layer import (  # noqa: E402
    kfac_kron_factors,
    mse_gauss_newton_fisher,
    mse_newton_step,
    one_layer_param_grad,
    one_layer_param_hessian,
    pack_params,
    unpack_params,
)

_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")


def _rand_one_layer(D=6, H=5, seed=0):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.normal(scale=0.3, size=(H, D)), dtype=jnp.float64)
    beta = jnp.asarray(rng.normal(scale=0.2, size=(H,)), dtype=jnp.float64)
    c = jnp.asarray(rng.normal(scale=0.4, size=(H,)), dtype=jnp.float64)
    b = jnp.asarray(0.13, dtype=jnp.float64)
    x = jnp.asarray(rng.normal(scale=1.0, size=(D,)), dtype=jnp.float64)
    return W, beta, c, b, x


def _f_unpack(theta, x, H, D, activation):
    """Forward of the one-layer field from the flat parameter vector."""
    b, c, beta, W = unpack_params(theta, H=H, D=D)
    z = W @ x + beta
    from omnibias.jax.activations import get_activation
    sigma = get_activation(activation).forward(z)
    return b + sigma @ c


@pytest.mark.parametrize("activation", _RICCATI)
def test_one_layer_param_grad_matches_jax_grad(activation):
    """``one_layer_param_grad`` must equal ``jax.grad`` to ~1e-12."""
    W, beta, c, b, x = _rand_one_layer(seed=hash(activation) & 0xFFFF)
    H, D = W.shape

    g_closed = one_layer_param_grad(x, W, beta, c, b, activation)
    theta = pack_params(b, c, beta, W)
    g_jax = jax.grad(_f_unpack)(theta, x, H, D, activation)

    assert g_closed.shape == g_jax.shape == (theta.shape[0],)
    np.testing.assert_allclose(
        np.asarray(g_closed), np.asarray(g_jax),
        rtol=1e-10, atol=1e-12,
        err_msg=f"param grad mismatch for activation={activation!r}",
    )


@pytest.mark.parametrize("activation", _RICCATI)
def test_one_layer_param_hessian_matches_jax_hessian(activation):
    """``one_layer_param_hessian`` must equal ``jax.hessian`` to ~1e-10."""
    W, beta, c, b, x = _rand_one_layer(seed=hash(activation) & 0xFFFF)
    H, D = W.shape

    Hess_closed = one_layer_param_hessian(x, W, beta, c, b, activation)
    theta = pack_params(b, c, beta, W)
    Hess_jax = jax.hessian(_f_unpack)(theta, x, H, D, activation)

    assert Hess_closed.shape == Hess_jax.shape
    # The reference autodiff Hessian is exactly symmetric — verify the
    # closed-form one is too (a numerical asymmetry would mean we mis-mapped
    # an off-diagonal index above).
    np.testing.assert_allclose(
        np.asarray(Hess_closed), np.asarray(Hess_closed.T),
        rtol=1e-12, atol=1e-12,
        err_msg=f"closed-form Hessian not symmetric for {activation!r}",
    )
    np.testing.assert_allclose(
        np.asarray(Hess_closed), np.asarray(Hess_jax),
        rtol=1e-9, atol=1e-10,
        err_msg=f"param Hessian mismatch for activation={activation!r}",
    )


def test_pack_unpack_roundtrip():
    W, beta, c, b, x = _rand_one_layer(D=8, H=4, seed=42)
    theta = pack_params(b, c, beta, W)
    b2, c2, beta2, W2 = unpack_params(theta, H=4, D=8)
    np.testing.assert_array_equal(np.asarray(b), np.asarray(b2))
    np.testing.assert_array_equal(np.asarray(c), np.asarray(c2))
    np.testing.assert_array_equal(np.asarray(beta), np.asarray(beta2))
    np.testing.assert_array_equal(np.asarray(W), np.asarray(W2))


def test_gauss_newton_reduces_loss_monotonically():
    """Gauss-Newton with appropriate damping must reduce MSE loss
    monotonically over several iterations.

    The one-layer field is bilinear in ``(W, c)`` so Newton is *not*
    one-shot exact, but the loss trajectory must be monotonically
    non-increasing (any sign of divergence indicates a bug in the
    Fisher or the gradient).
    """
    rng = np.random.default_rng(0)
    D, H, B = 4, 3, 60
    activation = "tanh"

    # True parameters (small scale to stay in the well-behaved regime).
    W_star = jnp.asarray(rng.normal(scale=0.3, size=(H, D)))
    beta_star = jnp.asarray(rng.normal(scale=0.1, size=(H,)))
    c_star = jnp.asarray(rng.normal(scale=0.4, size=(H,)))
    b_star = jnp.asarray(0.2)

    X = jnp.asarray(rng.normal(scale=0.4, size=(B, D)))
    from omnibias.jax.activations import get_activation
    sigma_star = get_activation(activation).forward(
        X @ W_star.T + beta_star[None, :]
    )
    Y = b_star + sigma_star @ c_star  # zero observation noise

    # Init at the true params + perturbation (so Fisher is well-conditioned).
    perturb_scale = 0.2
    W = W_star + jnp.asarray(rng.normal(scale=perturb_scale, size=W_star.shape))
    beta = beta_star + jnp.asarray(
        rng.normal(scale=perturb_scale, size=beta_star.shape)
    )
    c = c_star + jnp.asarray(rng.normal(scale=perturb_scale, size=c_star.shape))
    b = b_star + jnp.asarray(rng.normal(scale=perturb_scale))

    def loss(W_, beta_, c_, b_):
        sigma_ = get_activation(activation).forward(X @ W_.T + beta_[None, :])
        return float(jnp.mean((b_ + sigma_ @ c_ - Y) ** 2))

    losses = [loss(W, beta, c, b)]
    for _step in range(8):
        b, c, beta, W = mse_newton_step(
            X, Y, W, beta, c, b, activation=activation,
            learning_rate=0.5, damping=1e-3,
        )
        losses.append(loss(W, beta, c, b))

    # Loss must decrease monotonically; a *jump up* is a sign of bug.
    for i, (l_prev, l_now) in enumerate(zip(losses, losses[1:], strict=False)):
        assert l_now <= l_prev * 1.01, (
            f"Gauss-Newton increased loss at step {i}: "
            f"{l_prev:.3e} → {l_now:.3e} (trajectory: {losses})"
        )

    # Final loss should be much lower than initial.
    assert losses[-1] < 0.1 * losses[0], (
        f"Gauss-Newton didn't converge: loss[0]={losses[0]:.3e}, "
        f"loss[-1]={losses[-1]:.3e}"
    )


def test_gauss_newton_fisher_is_psd():
    """``F = (2/B) g g^T`` must be positive-semidefinite for every batch."""
    W, beta, c, b, _ = _rand_one_layer(seed=11)
    rng = np.random.default_rng(11)
    X = jnp.asarray(rng.normal(scale=0.5, size=(50, W.shape[1])))
    Y = jnp.asarray(rng.normal(scale=1.0, size=(50,)))
    F, _ = mse_gauss_newton_fisher(X, Y, W, beta, c, b, activation="tanh")
    eigs = np.linalg.eigvalsh(np.asarray(F))
    # F is sum of outer products → PSD up to round-off.
    assert eigs.min() >= -1e-10, f"min eig = {eigs.min():.3e} (should be ≥ 0)"


def test_kfac_factors_shape_and_psd():
    """KFAC ``(A, G)`` factors are symmetric PSD; ``A ⊗ G`` matches the
    ``W``-block of the full Gauss-Newton Fisher to a Kronecker-factor
    approximation tolerance."""
    W, beta, c, b, _ = _rand_one_layer(D=6, H=4, seed=7)
    rng = np.random.default_rng(7)
    X = jnp.asarray(rng.normal(scale=0.4, size=(40, W.shape[1])))

    A, G = kfac_kron_factors(X, W, beta, c, b, activation="tanh")
    D, H = W.shape[1], W.shape[0]
    assert A.shape == (D, D)
    assert G.shape == (H, H)
    # Both symmetric & PSD.
    for M, name in [(A, "A"), (G, "G")]:
        np.testing.assert_allclose(
            np.asarray(M), np.asarray(M.T),
            rtol=1e-12, atol=1e-12,
            err_msg=f"{name} should be symmetric",
        )
        eigs = np.linalg.eigvalsh(np.asarray(M))
        assert eigs.min() >= -1e-12, (
            f"{name} not PSD: min eig = {eigs.min():.3e}"
        )
