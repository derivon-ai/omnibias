# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Piecewise (almost-everywhere) activation tower tests (JAX).

Bit-identical agreement with the torch tower is covered by
``tests/test_jax_parity.py``; here we pin the JAX execution model:
``jit`` / ``vmap`` safety, integer-order reduction, the linear-piece
``n >= 2 -> 0`` convention, and a self-contained check of the a.e. tower
against ``jax.grad`` on the open pieces.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation, list_activations  # noqa: E402

_PIECEWISE_NAMES = [
    "leaky_relu", "prelu", "relu6", "hardtanh", "hardsigmoid", "hardswish",
    "elu", "selu", "celu", "softshrink", "hardshrink", "threshold",
    "abs", "sign", "step", "softsign",
]
_LINEAR_PIECE_NAMES = [
    "leaky_relu", "prelu", "relu6", "hardtanh", "hardsigmoid",
    "softshrink", "hardshrink", "threshold", "abs",
]
_AUTOGRAD_SPEC = {
    "relu": (3, [0.0]),
    "leaky_relu": (3, [0.0]),
    "relu6": (3, [0.0, 6.0]),
    "hardtanh": (3, [-1.0, 1.0]),
    "elu": (4, [0.0]),
    "celu": (4, [0.0]),
    "softsign": (4, [0.0]),
    "silu": (5, []),
    "gelu": (5, []),
}


def _safe_samples(breakpoints: list[float], *, pad: float = 0.08) -> jnp.ndarray:
    grid = np.linspace(-5.0, 5.0, 221)
    if breakpoints:
        keep = np.ones_like(grid, dtype=bool)
        for bp in breakpoints:
            keep &= np.abs(grid - bp) > pad
        grid = grid[keep]
    return jnp.asarray(grid)


def _nth_grad(fn, n: int):
    g = fn
    for _ in range(n):
        g = jax.grad(g)
    return g


@pytest.mark.parametrize("name", _PIECEWISE_NAMES)
def test_jit_matches_eager(name: str) -> None:
    spec = get_activation(name)
    z = jnp.linspace(-4.0, 4.0, 33)
    for n in range(0, 4):
        eager = spec.fastpath(z, n)
        jitted = jax.jit(lambda zz, n=n: spec.fastpath(zz, n))(z)
        np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", _PIECEWISE_NAMES)
def test_vmap_matches_eager(name: str) -> None:
    spec = get_activation(name)
    z = jnp.linspace(-4.0, 4.0, 17)
    mapped = jax.vmap(lambda zi: spec.fastpath(zi, 1))(z)
    np.testing.assert_allclose(np.asarray(mapped), np.asarray(spec.fastpath(z, 1)), atol=1e-6)


@pytest.mark.parametrize("name", _PIECEWISE_NAMES)
def test_order_reduction(name: str) -> None:
    spec = get_activation(name)
    z = jnp.linspace(-4.0, 4.0, 41)
    np.testing.assert_allclose(np.asarray(spec.fastpath(z, 0)), np.asarray(spec.forward(z)))
    assert spec.derivative is not None
    np.testing.assert_allclose(np.asarray(spec.fastpath(z, 1)), np.asarray(spec.derivative(z)))


@pytest.mark.parametrize("name", _LINEAR_PIECE_NAMES)
def test_linear_pieces_zero_from_order_two(name: str) -> None:
    spec = get_activation(name)
    z = jnp.linspace(-4.0, 4.0, 41)
    for n in (2, 3, 4):
        out = np.asarray(spec.fastpath(z, n))
        assert np.all(out == 0.0), f"{name!r} order {n} not zero"


@pytest.mark.parametrize("name", sorted(_AUTOGRAD_SPEC))
def test_tower_matches_autograd(name: str) -> None:
    max_order, breakpoints = _AUTOGRAD_SPEC[name]
    spec = get_activation(name)
    z = _safe_samples(breakpoints)
    fwd_scalar = lambda x: spec.forward(x)  # noqa: E731
    for n in range(1, max_order + 1):
        closed = np.asarray(spec.fastpath(z, n))
        auto = np.asarray(jax.vmap(_nth_grad(fwd_scalar, n))(z))
        np.testing.assert_allclose(
            closed, auto, rtol=1e-6, atol=1e-6,
            err_msg=f"{name!r} order {n}: closed form disagrees with jax.grad",
        )


def test_all_registered() -> None:
    registered = set(list_activations())
    for name in _PIECEWISE_NAMES:
        assert name in registered
