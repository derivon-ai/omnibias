# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for jax loss helpers (mirrors the torch test file)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax.losses import (  # noqa: E402
    causal_residual_loss,
    causal_weights_from_per_bin,
    entropy_consistent_residual,
    estimate_ntk_trace,
    mse_residual_loss,
    ntk_balanced_loss,
    sobolev_residual_loss,
    sobolev_weight,
)


def _rand_np(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float64)


# ---------------- Sobolev ------------------------------------------


def test_sobolev_p_zero_equals_mse():
    R = jnp.asarray(_rand_np((4, 16, 16), 0))
    loss = sobolev_residual_loss(R, L=2 * math.pi, sobolev_p=0.0)
    expected = jnp.mean(R * R)
    assert jnp.allclose(loss, expected, rtol=1e-12, atol=1e-12)


def test_sobolev_weight_p_zero_is_ones():
    R = jnp.asarray(_rand_np((4, 16, 16), 1))
    w = sobolev_weight(R, L=2 * math.pi, sobolev_p=0.0)
    assert jnp.allclose(w, jnp.ones_like(w))


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_sobolev_weight_max_one(p):
    R = jnp.asarray(_rand_np((1, 32, 32), 2))
    w = sobolev_weight(R, L=2 * math.pi, sobolev_p=p)
    assert jnp.all(w > 0)
    assert jnp.allclose(jnp.max(w), jnp.ones((), dtype=w.dtype))


def test_sobolev_1d_residual():
    R = jnp.asarray(_rand_np((4, 32), 3))
    loss = sobolev_residual_loss(R, L=2 * math.pi, sobolev_p=1.0)
    assert loss.ndim == 0
    assert float(loss) > 0


def test_sobolev_3d_residual():
    R = jnp.asarray(_rand_np((2, 8, 8, 8), 4))
    loss = sobolev_residual_loss(R, L=(2 * math.pi, 2 * math.pi, 1.0), sobolev_p=1.0)
    assert loss.ndim == 0
    assert float(loss) > 0


def test_sobolev_loss_smoother_than_mse():
    R = jnp.asarray(_rand_np((4, 32, 32), 5))
    mse = float(jnp.mean(R * R))
    sob = float(sobolev_residual_loss(R, L=2 * math.pi, sobolev_p=1.0))
    assert sob < mse


# ---------------- Causal weighting ---------------------------------


def test_causal_weights_non_increasing():
    L_per_bin = jnp.linspace(0.1, 1.0, 32, dtype=jnp.float64)
    w = causal_weights_from_per_bin(L_per_bin, epsilon=2.0)
    diffs = w[1:] - w[:-1]
    assert jnp.all(diffs <= 1e-15)
    assert jnp.allclose(w[0], jnp.ones((), dtype=w.dtype))


def test_causal_loss_matches_mse_when_eps_zero():
    R = jnp.asarray(_rand_np((8, 16, 16), 6))
    loss = causal_residual_loss(R, epsilon=0.0)
    expected = jnp.mean(R * R)
    assert jnp.allclose(loss, expected, rtol=1e-12, atol=1e-12)


def test_causal_loss_with_sobolev():
    R = jnp.asarray(_rand_np((8, 16, 16), 7))
    loss, w = causal_residual_loss(
        R, epsilon=1.0, L=2 * math.pi, sobolev_p=1.0,
        return_weights=True,
    )
    assert loss.ndim == 0
    assert w.shape == (8,)
    assert jnp.all(w[1:] - w[:-1] <= 1e-15)


def test_causal_loss_requires_L_when_sobolev_positive():
    R = jnp.asarray(_rand_np((4, 8), 8))
    with pytest.raises(ValueError, match="requires L"):
        causal_residual_loss(R, epsilon=1.0, sobolev_p=1.0)


def test_causal_weights_stop_grad():
    """The causal weights should be stop-gradient w.r.t. residual."""
    R = jnp.asarray(_rand_np((6, 8, 8), 9))

    def loss_fn(r):
        return causal_residual_loss(r, epsilon=1.0)

    grad = jax.grad(loss_fn)(R)
    assert grad.shape == R.shape
    assert not jnp.allclose(grad, 0.0)


# ---------------- Entropy-consistent ------------------------------


def test_entropy_default_is_mse():
    R = jnp.asarray(_rand_np((4, 8, 8), 10))
    loss = entropy_consistent_residual(R)
    expected = jnp.mean(R * R)
    assert jnp.allclose(loss, expected, rtol=1e-12, atol=1e-12)


def test_entropy_quadratic_matches_mse():
    R = jnp.asarray(_rand_np((4, 8, 8), 11))
    loss = entropy_consistent_residual(R, entropy_weight=lambda u: jnp.ones_like(u))
    expected = jnp.mean(R * R)
    assert jnp.allclose(loss, expected, rtol=1e-12, atol=1e-12)


# ---------------- NTK rebalance -----------------------------------


def test_ntk_balanced_equal_traces_gives_equal_weights():
    losses = {
        "pde": jnp.asarray(2.0, dtype=jnp.float64),
        "bc": jnp.asarray(3.0, dtype=jnp.float64),
        "ic": jnp.asarray(5.0, dtype=jnp.float64),
    }
    traces = {
        "pde": jnp.asarray(10.0, dtype=jnp.float64),
        "bc": jnp.asarray(10.0, dtype=jnp.float64),
        "ic": jnp.asarray(10.0, dtype=jnp.float64),
    }
    total, weights = ntk_balanced_loss(losses, ntk_traces=traces)
    for w in weights.values():
        assert math.isclose(w, 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_ntk_balanced_geometric_mean():
    losses = {"pde": jnp.asarray(1.0, dtype=jnp.float64),
              "bc": jnp.asarray(1.0, dtype=jnp.float64)}
    traces = {"pde": jnp.asarray(100.0, dtype=jnp.float64),
              "bc": jnp.asarray(1.0, dtype=jnp.float64)}
    _, w = ntk_balanced_loss(losses, ntk_traces=traces)
    assert math.isclose(w["pde"], 0.1, rel_tol=1e-9)
    assert math.isclose(w["bc"], 10.0, rel_tol=1e-9)


def test_estimate_ntk_trace_smoke():
    p = jnp.asarray(_rand_np((4, 3), 12))
    x = jnp.asarray(_rand_np((5, 3), 13))

    def loss_fn(params):
        return jnp.sum(x @ params.T)

    trace = estimate_ntk_trace(loss_fn, p)
    expected = float(jnp.sum(jnp.sum(x, axis=0) ** 2) * 4)
    assert math.isclose(float(trace), expected, rel_tol=1e-12, abs_tol=1e-12)


def test_ntk_balanced_validates_keys():
    losses = {"a": jnp.asarray(1.0, dtype=jnp.float64)}
    traces = {"b": jnp.asarray(1.0, dtype=jnp.float64)}
    with pytest.raises(ValueError, match="do not match"):
        ntk_balanced_loss(losses, ntk_traces=traces)


def test_mse_residual_loss():
    R = jnp.asarray(_rand_np((4, 8, 8), 14))
    expected = jnp.mean(R * R)
    assert jnp.allclose(mse_residual_loss(R), expected, rtol=1e-12, atol=1e-12)
