# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form activation antiderivative tests for the JAX backend."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation  # noqa: E402

SUPPORTED_INTEGRALS = [
    "sigmoid",
    "tanh",
    "gaussian",
    "exp",
    "huber",
    "arctan",
    "log1pu2",
    "relu",
    "gelu",
    "sin",
    "cos",
    "sinh",
    "cosh",
    "tan",
    "cot",
    "sech",
    "coth",
    "softabs",
    "smooth_sign",
]


def _safe_points(name: str):
    if name in {"cot", "coth"}:
        return jnp.linspace(0.35, 1.35, 17)
    if name in {"huber", "relu"}:
        return jnp.asarray([-1.4, -0.7, -0.2, 0.2, 0.7, 1.4])
    return jnp.linspace(-0.8, 0.8, 17)


@pytest.mark.parametrize("name", SUPPORTED_INTEGRALS)
def test_integral_derivative_recovers_activation(name: str) -> None:
    spec = get_activation(name)
    assert spec.integral is not None
    z = _safe_points(name)
    grad = jax.grad(lambda x: jnp.sum(spec.integral(x)))(z)
    ref = spec.forward(z)
    assert jnp.allclose(grad, ref, atol=1e-9, rtol=1e-9), name
