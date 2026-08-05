# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression and contract tests for the JAX backend audit (T1.c).

Locks down three contracts:

1. **Negative-order validation** for every JAX activation fastpath
   kernel: ``ValueError`` on ``n < 0`` rather than a misleading
   ``NotImplementedError`` with a "got -1" message.

2. **JIT compatibility**: every closed-form Laplacian / Hessian /
   polylaplacian helper in :mod:`omnibias.jax.laplacian` must be
   ``jit``-able with ``order`` / ``k`` as a static argument and must
   not host-convert traced values.

3. **vmap compatibility**: every helper must be ``vmap``-able over a
   batch of evaluation points without throwing.

All tests run on CPU in float64.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import (  # noqa: E402
    JaxActivationSpec,
    get_activation,
    list_activations,
)
from omnibias.jax.laplacian import (  # noqa: E402
    neural_field_hessian,
    neural_field_laplacian,
    neural_field_polylaplacian,
    neural_field_value,
    neural_field_value_and_laplacian,
    neural_field_value_and_polylaplacian,
    neural_field_value_grad_hessian,
    neural_field_value_grad_laplacian,
)

_FASTPATH_ACTIVATIONS = [
    name for name in list_activations() if get_activation(name).fastpath is not None
]

_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")


# ---------------------------------------------------------------------------
# 1. Negative-order validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _FASTPATH_ACTIVATIONS)
def test_jax_fastpath_rejects_negative_order(name: str) -> None:
    """Every JAX spec.fastpath must raise ValueError on n < 0."""
    spec = get_activation(name)
    fp = spec.fastpath
    assert fp is not None
    with pytest.raises(ValueError, match="order n must be"):
        fp(jnp.zeros(1), -1)
    with pytest.raises(ValueError, match="order n must be"):
        fp(jnp.zeros(1), -7)


# ---------------------------------------------------------------------------
# 2. JIT compatibility.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _RICCATI)
def test_neural_field_laplacian_jit(name: str) -> None:
    rng = np.random.default_rng(0)
    D, H = 3, 5
    W = jnp.asarray(rng.normal(size=(H, D)))
    beta = jnp.asarray(rng.normal(size=(H,)))
    c = jnp.asarray(rng.normal(size=(H,)))
    x = jnp.asarray(rng.normal(size=(D,)))

    f_jit = jax.jit(neural_field_laplacian, static_argnums=(4,))
    val_jit = f_jit(x, W, beta, c, name)
    val_eager = neural_field_laplacian(x, W, beta, c, name)
    np.testing.assert_allclose(val_jit, val_eager, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("name", _RICCATI)
def test_neural_field_value_grad_hessian_jit(name: str) -> None:
    rng = np.random.default_rng(1)
    D, H = 4, 6
    W = jnp.asarray(rng.normal(size=(H, D)))
    beta = jnp.asarray(rng.normal(size=(H,)))
    c = jnp.asarray(rng.normal(size=(H,)))
    x = jnp.asarray(rng.normal(size=(D,)))
    b = 0.1

    f_jit = jax.jit(neural_field_value_grad_hessian, static_argnums=(5,))
    f, g, hess = f_jit(x, W, beta, c, b, name)

    f0, g0, h0 = neural_field_value_grad_hessian(x, W, beta, c, b, name)
    np.testing.assert_allclose(f, f0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(g, g0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hess, h0, rtol=0.0, atol=1e-12)

    assert g.shape == (D,)
    assert hess.shape == (D, D)
    np.testing.assert_allclose(hess, hess.T, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("name", _RICCATI)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_neural_field_polylaplacian_jit(name: str, k: int) -> None:
    rng = np.random.default_rng(2)
    D, H = 3, 5
    W = jnp.asarray(rng.normal(size=(H, D)))
    beta = jnp.asarray(rng.normal(size=(H,)))
    c = jnp.asarray(rng.normal(size=(H,)))
    x = jnp.asarray(rng.normal(size=(D,)))

    f_jit = jax.jit(neural_field_polylaplacian, static_argnums=(4, 5))
    val_jit = f_jit(x, W, beta, c, name, k)
    val_eager = neural_field_polylaplacian(x, W, beta, c, name, k)
    # XLA can reorder reductions, so the tolerance has to be a few ULP
    # of the magnitude of the result. 1e-12 absolute / 1e-10 relative is
    # plenty for 3-d test fields.
    np.testing.assert_allclose(val_jit, val_eager, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. vmap compatibility.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _RICCATI)
def test_neural_field_value_grad_hessian_vmap(name: str) -> None:
    rng = np.random.default_rng(3)
    B, D, H = 7, 3, 5
    W = jnp.asarray(rng.normal(size=(H, D)))
    beta = jnp.asarray(rng.normal(size=(H,)))
    c = jnp.asarray(rng.normal(size=(H,)))
    xs = jnp.asarray(rng.normal(size=(B, D)))
    b = 0.0

    f_v = jax.vmap(
        lambda x: neural_field_value_grad_hessian(x, W, beta, c, b, name),
        in_axes=0,
    )
    fs, gs, hs = f_v(xs)
    assert fs.shape == (B,)
    assert gs.shape == (B, D)
    assert hs.shape == (B, D, D)


@pytest.mark.parametrize("name", _RICCATI)
def test_neural_field_polylaplacian_vmap(name: str) -> None:
    rng = np.random.default_rng(4)
    B, D, H = 6, 3, 5
    W = jnp.asarray(rng.normal(size=(H, D)))
    beta = jnp.asarray(rng.normal(size=(H,)))
    c = jnp.asarray(rng.normal(size=(H,)))
    xs = jnp.asarray(rng.normal(size=(B, D)))

    f_v = jax.vmap(
        lambda x: neural_field_polylaplacian(x, W, beta, c, name, 2),
        in_axes=0,
    )
    out = f_v(xs)
    assert out.shape == (B,)


# ---------------------------------------------------------------------------
# 4. Closed-form Laplacian == jax.hessian trace cross-check
#    (cheap analytic-vs-autograd reference test).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _RICCATI)
def test_neural_field_laplacian_matches_jax_hessian(name: str) -> None:
    rng = np.random.default_rng(5)
    D, H = 4, 6
    W = jnp.asarray(rng.normal(scale=0.3, size=(H, D)))
    beta = jnp.asarray(rng.normal(scale=0.2, size=(H,)))
    c = jnp.asarray(rng.normal(scale=0.4, size=(H,)))
    x = jnp.asarray(rng.normal(scale=0.7, size=(D,)))

    def f(x):
        return neural_field_value(x, W, beta, c, 0.0, name)

    lap_closed = neural_field_laplacian(x, W, beta, c, name)
    lap_autograd = jnp.trace(jax.hessian(f)(x))
    np.testing.assert_allclose(
        lap_closed, lap_autograd, rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("name", _RICCATI)
def test_neural_field_hessian_matches_jax_hessian(name: str) -> None:
    rng = np.random.default_rng(6)
    D, H = 4, 6
    W = jnp.asarray(rng.normal(scale=0.3, size=(H, D)))
    beta = jnp.asarray(rng.normal(scale=0.2, size=(H,)))
    c = jnp.asarray(rng.normal(scale=0.4, size=(H,)))
    x = jnp.asarray(rng.normal(scale=0.7, size=(D,)))

    def f(x):
        return neural_field_value(x, W, beta, c, 0.0, name)

    hess_closed = neural_field_hessian(x, W, beta, c, name)
    hess_autograd = jax.hessian(f)(x)
    np.testing.assert_allclose(
        hess_closed, hess_autograd, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        hess_closed, hess_closed.T, rtol=0.0, atol=1e-13
    )


@pytest.mark.parametrize("name", _RICCATI)
def test_neural_field_grad_matches_jax_grad(name: str) -> None:
    rng = np.random.default_rng(7)
    D, H = 4, 6
    W = jnp.asarray(rng.normal(scale=0.3, size=(H, D)))
    beta = jnp.asarray(rng.normal(scale=0.2, size=(H,)))
    c = jnp.asarray(rng.normal(scale=0.4, size=(H,)))
    x = jnp.asarray(rng.normal(scale=0.7, size=(D,)))

    def f(x):
        return neural_field_value(x, W, beta, c, 0.0, name)

    _, grad_closed, _ = neural_field_value_grad_laplacian(x, W, beta, c, 0.0, name)
    grad_autograd = jax.grad(f)(x)
    np.testing.assert_allclose(
        grad_closed, grad_autograd, rtol=1e-10, atol=1e-13
    )


# ---------------------------------------------------------------------------
# 5. Riccati identity at order 1 (sanity check on the polynomial spec).
# ---------------------------------------------------------------------------


def test_sigmoid_riccati_identity() -> None:
    """``sigma'(z) - sigma * (1 - sigma) == 0`` (P_1(s) = s - s^2)."""
    spec = get_activation("sigmoid")
    z = jnp.linspace(-3.0, 3.0, 21)
    sigma_p = spec.fastpath(z, 1)
    s = spec.forward(z)
    riccati = s * (1.0 - s)
    np.testing.assert_allclose(sigma_p, riccati, rtol=0.0, atol=1e-13)


def test_tanh_riccati_identity() -> None:
    """``tanh'(z) - (1 - tanh^2(z)) == 0`` (T_1(t) = 1 - t^2)."""
    spec = get_activation("tanh")
    z = jnp.linspace(-3.0, 3.0, 21)
    t_p = spec.fastpath(z, 1)
    t = spec.forward(z)
    riccati = 1.0 - t * t
    np.testing.assert_allclose(t_p, riccati, rtol=0.0, atol=1e-13)


def test_exp_riccati_identity() -> None:
    """``exp'(z) - exp(z) == 0`` (P(E) = E)."""
    spec = get_activation("exp")
    z = jnp.linspace(-3.0, 3.0, 21)
    e_p = spec.fastpath(z, 1)
    e = spec.forward(z)
    np.testing.assert_allclose(e_p, e, rtol=0.0, atol=1e-13)
