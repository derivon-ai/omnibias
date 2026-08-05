# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The shared anneal_descent core: torch <-> jax parity, unit box, anneal-to-vertex."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.discrete import AnnealSchedule


def _quadratic(seed: int, n: int = 6, lin_scale: float = 5.0) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    a = 0.3 * (m + m.T)
    b = lin_scale * rng.standard_normal(n)
    scale = 2.0 * float(np.sqrt(np.sum(a * a))) + float(np.max(np.abs(b)))
    return a, b, scale


def test_torch_jax_bit_identical() -> None:
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")
    from omnibias.discrete.jax import anneal_descent as ja
    from omnibias.discrete.torch import anneal_descent as ta

    a, b, scale = _quadratic(0)  # well-determined (strong linear term) -> commits cleanly
    n = a.shape[0]
    aj, bj = jnp.asarray(a), jnp.asarray(b)
    xj = np.asarray(ja(lambda x: 2.0 * (aj @ x) + bj, scale, n))
    at, bt = torch.as_tensor(a), torch.as_tensor(b)
    xt = ta(lambda x: 2.0 * (at @ x) + bt, scale, n).detach().numpy()
    assert np.max(np.abs(xj - xt)) < 1e-9


def test_output_is_in_the_unit_box() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from omnibias.discrete.jax import anneal_descent as ja

    a, b, scale = _quadratic(2)
    aj, bj = jnp.asarray(a), jnp.asarray(b)
    x = np.asarray(ja(lambda z: 2.0 * (aj @ z) + bj, scale, a.shape[0]))
    assert np.all(x >= 0.0) and np.all(x <= 1.0) and np.all(np.isfinite(x))


def test_annealing_collapses_toward_a_vertex() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from omnibias.discrete.jax import anneal_descent as ja

    a, b, scale = _quadratic(0)
    aj, bj = jnp.asarray(a), jnp.asarray(b)
    x = np.asarray(ja(lambda z: 2.0 * (aj @ z) + bj, scale, a.shape[0], AnnealSchedule(stages=16)))
    assert float(np.mean(np.minimum(x, 1.0 - x))) < 0.1


def test_descent_is_differentiable_in_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.discrete.torch import anneal_descent as ta

    a, b, scale = _quadratic(4, n=5)
    at = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    out = ta(lambda x: 2.0 * (at @ x) + bt, scale, 5, AnnealSchedule.fast())
    out.sum().backward()
    assert at.grad is not None and torch.all(torch.isfinite(at.grad))
    assert bt.grad is not None and torch.all(torch.isfinite(bt.grad))
