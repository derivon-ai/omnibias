# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Beta-tempered smooth-surrogate tower tests (JAX).

Bit-identical agreement with torch is covered by ``tests/test_jax_parity.py``;
here we pin the JAX-specific surface: the ``tempered_activation`` functional
helper with a *traced* (learnable) ``beta``, ``jit`` safety, beta-scaling
identities, and the non-registered factories (``swish`` / ``soft_leaky_relu``).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import (  # noqa: E402
    get_activation,
    make_soft_leaky_relu_spec,
    make_soft_relu_spec,
    make_swish_spec,
    tempered_activation,
)


def _nth_grad(fn, n: int):
    g = fn
    for _ in range(n):
        g = jax.grad(g)
    return g


def test_soft_sign_soft_abs_aliases() -> None:
    assert get_activation("soft_sign").name == "smooth_sign"
    assert get_activation("soft_abs").name == "softabs"


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 4.0])
def test_soft_relu_beta_scaling_identity(beta: float) -> None:
    spec = make_soft_relu_spec(beta)
    softplus = get_activation("softplus")
    z = jnp.linspace(-3.0, 3.0, 41)
    for n in range(0, 6):
        lhs = np.asarray(spec.fastpath(z, n))
        rhs = np.asarray((beta ** (n - 1)) * softplus.fastpath(beta * z, n))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-6, err_msg=f"n={n}")


def test_tempered_activation_helper_matches_registered() -> None:
    z = jnp.linspace(-3.0, 3.0, 41)
    helper = tempered_activation("softplus", beta=1.0, scale="one_over_beta")
    reg = get_activation("soft_relu")
    for n in range(0, 6):
        np.testing.assert_allclose(
            np.asarray(helper.fastpath(z, n)), np.asarray(reg.fastpath(z, n)),
            rtol=1e-6, atol=1e-6,
        )


def test_tempered_activation_traced_beta_gradient() -> None:
    z = jnp.linspace(-3.0, 3.0, 41)

    def loss(beta):
        return tempered_activation("softplus", beta=beta, scale="one_over_beta").forward(z).sum()

    g = float(jax.grad(loss)(2.0))
    eps = 1e-6
    fd = (float(loss(2.0 + eps)) - float(loss(2.0 - eps))) / (2 * eps)
    assert abs(g - fd) < 1e-4, f"grad {g} vs finite-diff {fd}"


def test_tempered_activation_jit_traced_beta() -> None:
    z = jnp.linspace(-3.0, 3.0, 41)

    def bump(beta):
        return tempered_activation("softplus", beta=beta).fastpath(z, 2)

    jitted = jax.jit(bump)(3.0)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(bump(3.0)), rtol=1e-6, atol=1e-6)


def test_swish_beta1_is_silu() -> None:
    swish = make_swish_spec(1.0)
    silu = get_activation("silu")
    z = jnp.linspace(-4.0, 4.0, 41)
    for n in range(0, 6):
        np.testing.assert_allclose(
            np.asarray(swish.fastpath(z, n)), np.asarray(silu.fastpath(z, n)),
            rtol=1e-6, atol=1e-6, err_msg=f"n={n}",
        )


def test_swish_matches_autograd() -> None:
    spec = make_swish_spec(1.5)
    z = jnp.linspace(-4.0, 4.0, 41)
    fwd = lambda x: spec.forward(x)  # noqa: E731
    for n in range(1, 5):
        closed = np.asarray(spec.fastpath(z, n))
        auto = np.asarray(jax.vmap(_nth_grad(fwd, n))(z))
        np.testing.assert_allclose(closed, auto, rtol=1e-6, atol=1e-6, err_msg=f"n={n}")


def test_soft_relu_converges_to_relu() -> None:
    relu = get_activation("relu")
    z = jnp.asarray([-3.0, -1.0, -0.3, 0.3, 1.0, 3.0])
    prev = None
    for beta in (2.0, 8.0, 32.0, 128.0):
        err = float(jnp.max(jnp.abs(make_soft_relu_spec(beta).forward(z) - relu.forward(z))))
        if prev is not None:
            assert err < prev
        prev = err
    assert prev < 1e-2


def test_soft_leaky_relu_converges() -> None:
    alpha = 0.1
    z = jnp.asarray([-3.0, -1.0, -0.3, 0.3, 1.0, 3.0])
    hard = jnp.where(z > 0, z, alpha * z)
    err = float(jnp.max(jnp.abs(make_soft_leaky_relu_spec(alpha, 128.0).forward(z) - hard)))
    assert err < 1e-2
