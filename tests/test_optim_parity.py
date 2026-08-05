# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity of the Gauss-Newton / energy-natural-gradient direction.

The torch and jax :func:`gauss_newton_direction` solve the same (regularised) normal
equations, so for a shared ``(J, r)`` they agree to float64 round-off in both the primal
(``P <= N``) and dual (``P > N``) regimes.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.optim import gauss_newton_direction as jax_gnd  # noqa: E402
from omnibias.jax.optim import gauss_newton_fisher as jax_gnf  # noqa: E402
from omnibias.jax.optim import natural_gradient_direction as jax_ngd  # noqa: E402
from omnibias.torch.optim import gauss_newton_direction as torch_gnd  # noqa: E402
from omnibias.torch.optim import gauss_newton_fisher as torch_gnf  # noqa: E402
from omnibias.torch.optim import natural_gradient_direction as torch_ngd  # noqa: E402

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize(
    ("n", "p"),
    [(20, 5), (8, 8), (6, 12)],  # over-determined (primal), square, over-parameterised (dual)
)
@pytest.mark.parametrize("damping", [1e-1, 1e-3])
def test_gauss_newton_direction_bit_parity(n: int, p: int, damping: float) -> None:
    rng = np.random.RandomState(n * 100 + p)
    jac = rng.randn(n, p)
    res = rng.randn(n)
    d_t = torch_gnd(torch.tensor(jac), torch.tensor(res), damping).numpy()
    d_j = np.asarray(jax_gnd(jnp.asarray(jac), jnp.asarray(res), damping))
    assert np.allclose(d_t, d_j, rtol=0, atol=1e-10)


def _spd(p: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    m = rng.randn(p, p)
    return m @ m.T + p * np.eye(p)


@pytest.mark.parametrize("p", [4, 9])
@pytest.mark.parametrize("damping", [0.0, 1e-2, 1.0])
def test_natural_gradient_direction_bit_parity(p: int, damping: float) -> None:
    """The dense natural-gradient solve ``(M + mu I)^{-1} g`` agrees torch <-> jax to round-off."""
    metric = _spd(p, p * 13 + 1)
    g = np.random.RandomState(p * 13 + 2).randn(p)
    d_t = torch_ngd(torch.tensor(metric), torch.tensor(g), damping=damping).numpy()
    d_j = np.asarray(jax_ngd(jnp.asarray(metric), jnp.asarray(g), damping=damping))
    assert np.allclose(d_t, d_j, rtol=0, atol=1e-10)


@pytest.mark.parametrize(("n", "p"), [(20, 5), (12, 8)])
def test_gauss_newton_fisher_bit_parity(n: int, p: int) -> None:
    """The closed-form Gauss-Newton Fisher ``(F, g)`` agrees torch <-> jax on a shared residual."""
    rng = np.random.RandomState(n * 7 + p)
    x = rng.randn(n, p)
    y = rng.randn(n)
    theta0 = rng.randn(p)

    xt, yt = torch.tensor(x), torch.tensor(y)
    xj, yj = jnp.asarray(x), jnp.asarray(y)
    f_t, g_t = torch_gnf(lambda th: xt @ th - yt, torch.tensor(theta0))
    f_j, g_j = jax_gnf(lambda th: xj @ th - yj, jnp.asarray(theta0))
    assert np.allclose(f_t.numpy(), np.asarray(f_j), rtol=0, atol=1e-10)
    assert np.allclose(g_t.numpy(), np.asarray(g_j), rtol=0, atol=1e-10)
