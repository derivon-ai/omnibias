# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contract tests for :mod:`omnibias.jax.laplacian` polylaplacian primitive.

The closed-form k-th iterated Laplacian

    Delta^k f(x) = sum_h c_h * sigma^{(2k)}(z_h) * ||W_h||^{2k}

must match the brute-force ``Delta^k = jax.jacfwd(jax.jacrev(...))^k``
trace computation to float64 precision on the one-layer omnibias
field, for every Riccati-class activation that registers a fast-path
kernel at orders >= 4.

This is the relativistic-VMC primitive: the mass-velocity
correction <p^4 psi> = <Delta^2 psi> needs k=2, and the third-order
Foldy-Wouthuysen expansion <p^6 psi> needs k=3.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.laplacian import (  # noqa: E402
    neural_field_laplacian,
    neural_field_polylaplacian,
    neural_field_value,
    neural_field_value_and_polylaplacian,
)

# Riccati-class activations support fastpath at arbitrary order.
_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")
_DS = (2, 4, 8)
_KS = (1, 2, 3)
_H = 8
_RTOL = 1e-7
_ATOL = 1e-9


def _rand_params(D: int, H: int, seed: int):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.3, size=(H, D))
    beta = rng.normal(scale=0.2, size=(H,))
    c = rng.normal(scale=0.4, size=(H,))
    b = 0.13
    x = rng.normal(scale=0.7, size=(D,))
    return (
        jnp.asarray(W, dtype=jnp.float64),
        jnp.asarray(beta, dtype=jnp.float64),
        jnp.asarray(c, dtype=jnp.float64),
        b,
        jnp.asarray(x, dtype=jnp.float64),
    )


def _brute_polylaplacian(f, x, k):
    """``Delta^k f`` by repeated ``trace(Hessian)``.

    Delta f = trace(jax.hessian(f)(x)).  We iterate this k times by
    defining the intermediate scalar field ``Delta^{i} f`` and
    computing its Hessian trace.  This is the conceptual reference
    against which the closed-form fastpath is benchmarked.
    """
    g = f
    for _ in range(k):
        prev_g = g

        def g(xx, _prev_g=prev_g):  # noqa: B023  # default-arg captures prev_g per iter
            return jnp.trace(jax.hessian(_prev_g)(xx))

    return g(x)


@pytest.mark.parametrize("activation", _RICCATI)
@pytest.mark.parametrize("D", _DS)
@pytest.mark.parametrize("k", _KS)
def test_neural_field_polylaplacian_matches_brute(activation: str, D: int, k: int) -> None:
    W, beta, c, b, x = _rand_params(D, _H, seed=hash((activation, D, k)) & 0xFFFF)

    def f(xx: jnp.ndarray) -> jnp.ndarray:
        return neural_field_value(xx, W, beta, c, b, activation)

    Delta_k_closed = neural_field_polylaplacian(x, W, beta, c, activation, k=k)
    Delta_k_ref = _brute_polylaplacian(f, x, k=k)

    np.testing.assert_allclose(
        np.asarray(Delta_k_closed),
        np.asarray(Delta_k_ref),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"Delta^{k} mismatch for {activation!r} at D={D}",
    )


@pytest.mark.parametrize("activation", _RICCATI)
def test_polylaplacian_k_eq_1_matches_laplacian(activation: str) -> None:
    """For k=1, polylaplacian must equal ``neural_field_laplacian``."""
    W, beta, c, _, x = _rand_params(6, _H, seed=hash(activation) & 0xFFFF)
    lap_k1 = neural_field_polylaplacian(x, W, beta, c, activation, k=1)
    lap_ref = neural_field_laplacian(x, W, beta, c, activation)
    np.testing.assert_allclose(
        np.asarray(lap_k1),
        np.asarray(lap_ref),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("activation", _RICCATI)
def test_value_and_polylaplacian_matches_separate(activation: str) -> None:
    """Fused (value, Delta^k) call equals separate calls."""
    W, beta, c, b, x = _rand_params(6, _H, seed=hash(activation) & 0xFFFF)
    for k in (1, 2, 3):
        f_fused, pk_fused = neural_field_value_and_polylaplacian(x, W, beta, c, b, activation, k=k)
        f_sep = neural_field_value(x, W, beta, c, b, activation)
        pk_sep = neural_field_polylaplacian(x, W, beta, c, activation, k=k)
        np.testing.assert_allclose(np.asarray(f_fused), np.asarray(f_sep), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(np.asarray(pk_fused), np.asarray(pk_sep), rtol=1e-14, atol=1e-14)


def test_batched_polylaplacian() -> None:
    """Batched input ``x`` of shape (B, D) yields output of shape (B,)."""
    D, H, B = 5, 8, 17
    rng = np.random.default_rng(20260520)
    W = jnp.asarray(rng.normal(scale=0.3, size=(H, D)))
    beta = jnp.asarray(rng.normal(scale=0.2, size=(H,)))
    c = jnp.asarray(rng.normal(scale=0.4, size=(H,)))
    x = jnp.asarray(rng.normal(scale=0.7, size=(B, D)))

    delta2 = neural_field_polylaplacian(x, W, beta, c, "tanh", k=2)
    assert delta2.shape == (B,)
    # First batch element must match the unbatched call.
    delta2_single = neural_field_polylaplacian(x[0], W, beta, c, "tanh", k=2)
    np.testing.assert_allclose(
        np.asarray(delta2[0]),
        np.asarray(delta2_single),
        rtol=1e-12,
        atol=1e-12,
    )
