# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity: the deep-network Laplacian fast lane, torch vs JAX.

Both :mod:`omnibias.jax.laplacian` and :mod:`omnibias.torch.laplacian` drive
the same pure-Python combinatorics in :mod:`omnibias.core.contraction` /
:mod:`omnibias.core.multi_index`, so identical ``(W, b, spec)`` layers (fed
identical numpy weight arrays) must agree to numerical precision:

* Tier A (``deep_field_laplacian``) and Tier B (``deep_field_polylaplacian``,
  ``mode="support"``) are exact closed-form contractions, so they are
  required to agree at ``rtol=atol=1e-6`` (float64, per
  :mod:`tests.test_jax_parity`'s convention -- comfortably tighter than that
  in practice, but this file keeps the same declared bound).
* Tier C (``mode="estimator"``) draws directions from each backend's own PRNG
  (``jax.random`` vs ``torch.Generator``), so a matching integer seed does
  **not** give bit-identical direction samples across backends. Its parity
  check is therefore statistical: both backends' sample means must be close
  to the shared Tier B exact value, not to each other bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.laplacian import (  # noqa: E402
    deep_field_laplacian as jax_deep_field_laplacian,
)
from omnibias.jax.laplacian import (  # noqa: E402
    deep_field_polylaplacian as jax_deep_field_polylaplacian,
)
from omnibias.torch.laplacian import (  # noqa: E402
    deep_field_laplacian as torch_deep_field_laplacian,
)
from omnibias.torch.laplacian import (  # noqa: E402
    deep_field_polylaplacian as torch_deep_field_polylaplacian,
)

_RTOL = 1e-6
_ATOL = 1e-6
_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp", "sin")


def _numpy_layers(dim: int, hidden: int, depth: int, activation: str, seed: int):
    """Plain numpy ``(W, b, activation_name)`` layer list, backend-neutral."""
    rng = np.random.default_rng(seed)
    dims = [dim] + [hidden] * depth + [1]
    layers = []
    for i in range(len(dims) - 1):
        scale = 1.0 / np.sqrt(dims[i])
        W = rng.normal(scale=scale, size=(dims[i + 1], dims[i]))
        b = rng.normal(scale=0.1, size=(dims[i + 1],))
        spec = None if i == len(dims) - 2 else activation
        layers.append((W, b, spec))
    return layers


def _to_jax(layers):
    return [
        (jnp.asarray(W, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64), spec)
        for W, b, spec in layers
    ]


def _to_torch(layers):
    return [
        (torch.as_tensor(W, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64), spec)
        for W, b, spec in layers
    ]


@pytest.mark.parametrize("dim", (2, 4, 8, 16))
@pytest.mark.parametrize("depth", (1, 2, 3))
@pytest.mark.parametrize("activation", _RICCATI)
def test_laplacian_parity(dim, depth, activation) -> None:
    layers_np = _numpy_layers(dim, hidden=6, depth=depth, activation=activation, seed=0)
    x_np = np.random.default_rng(1).normal(scale=0.3, size=(dim,))

    jax_lap = jax_deep_field_laplacian(jnp.asarray(x_np, dtype=jnp.float64), _to_jax(layers_np))
    torch_lap = torch_deep_field_laplacian(
        torch.as_tensor(x_np, dtype=torch.float64), _to_torch(layers_np)
    )
    assert float(jnp.abs(jax_lap[0] - float(torch_lap[0]))) <= _ATOL + _RTOL * abs(
        float(torch_lap[0])
    )


@pytest.mark.parametrize("k", (2, 3))
@pytest.mark.parametrize("activation", ("tanh", "sigmoid", "gaussian"))
def test_polylaplacian_support_parity(k, activation) -> None:
    dim = 4
    layers_np = _numpy_layers(dim, hidden=5, depth=2, activation=activation, seed=2)
    x_np = np.random.default_rng(3).normal(scale=0.2, size=(dim,))

    jax_out = jax_deep_field_polylaplacian(
        jnp.asarray(x_np, dtype=jnp.float64), _to_jax(layers_np), k, mode="support"
    )
    torch_out = torch_deep_field_polylaplacian(
        torch.as_tensor(x_np, dtype=torch.float64), _to_torch(layers_np), k, mode="support"
    )
    assert float(jnp.abs(jax_out[0] - float(torch_out[0]))) <= _ATOL + _RTOL * abs(
        float(torch_out[0])
    )


def test_polylaplacian_estimator_parity_is_statistical() -> None:
    """Different PRNGs, same expectation: both backends' sample means must
    track the shared Tier B exact value, not each other bit-for-bit."""
    dim, k = 4, 2
    layers_np = _numpy_layers(dim, hidden=5, depth=2, activation="tanh", seed=4)
    x_np = np.random.default_rng(5).normal(scale=0.2, size=(dim,))

    exact = float(
        jax_deep_field_polylaplacian(
            jnp.asarray(x_np, dtype=jnp.float64), _to_jax(layers_np), k, mode="support"
        )[0]
    )

    n_directions = 8000
    jax_means = [
        float(
            jax_deep_field_polylaplacian(
                jnp.asarray(x_np, dtype=jnp.float64),
                _to_jax(layers_np),
                k,
                mode="estimator",
                n_directions=n_directions,
                seed=seed,
            )[0]
        )
        for seed in range(5)
    ]
    torch_means = [
        float(
            torch_deep_field_polylaplacian(
                torch.as_tensor(x_np, dtype=torch.float64),
                _to_torch(layers_np),
                k,
                mode="estimator",
                n_directions=n_directions,
                seed=seed,
            )[0]
        )
        for seed in range(5)
    ]
    jax_grand_mean = sum(jax_means) / len(jax_means)
    torch_grand_mean = sum(torch_means) / len(torch_means)
    tol = 0.2 * (abs(exact) + 1.0)
    assert abs(jax_grand_mean - exact) < tol
    assert abs(torch_grand_mean - exact) < tol


def test_tier_a_ulp_divergence_is_in_tensordot_reduction() -> None:
    """Tier A polynomials and matmul agree; the gap is in ``tensordot``/``sum``.

    At ``(D=1, H=1)`` the backends can agree exactly under float64 x64, but
    wider hidden layers pick up a 1-2 ULP gap from differing reduction order
    even when every coefficient matches.
    """
    from omnibias.jax.laplacian import deep_field_laplacian as jax_lap
    from omnibias.torch.laplacian import deep_field_laplacian as torch_lap

    layers_np = _numpy_layers(dim=1, hidden=2, depth=1, activation="tanh", seed=0)
    x_np = np.random.default_rng(0).normal(size=(1,)) * 0.3
    t = float(
        torch_lap(
            torch.as_tensor(x_np, dtype=torch.float64),
            _to_torch(layers_np),
        )[0]
    )
    j = float(
        jax_lap(jnp.asarray(x_np, dtype=jnp.float64), _to_jax(layers_np))[0]
    )
    gap = abs(t - j)
    assert 0 < gap < 1e-15
