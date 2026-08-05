# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Differentiable QP/LP layer (torch): gradient parity with JAX + finite-diff sanity.

The strongest correctness check is torch<->jax *gradient* parity: both layers solve
the (bit-identical) interior-point problem and apply the same KKT adjoint, so their
gradients must agree to solver accuracy with no finite-difference noise. A single
directional finite-difference adds an absolute sanity check.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.convex.torch import lp_layer, qp_layer  # noqa: E402
from omnibias.convex.torch.solver import solve_qp  # noqa: E402


def _random_qp(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 3
    M = rng.standard_normal((n, n))
    Q = M @ M.T + 0.5 * np.eye(n)
    x_target = np.array([3.0, -3.0, 0.2])  # outside [-1, 1] -> active box constraints
    c = -Q @ x_target
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.ones(2 * n)
    return Q, c, A, b


def _t(x: object, grad: bool = False) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, requires_grad=grad)


def test_qp_layer_grad_matches_jax() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.convex.jax import qp_layer as qp_layer_jax

    Q, c, A, b = _random_qp(2)
    rng = np.random.default_rng(20)
    w = rng.standard_normal(c.shape[0])
    wt = torch.as_tensor(w, dtype=torch.float64)

    # ---- torch grads (one backward per parameter) ----
    def torch_grad(which: str) -> np.ndarray:
        Qt, ct, At, bt = (_t(Q, which == "Q"), _t(c, which == "c"),
                          _t(A, which == "A"), _t(b, which == "b"))
        leaf = {"Q": Qt, "c": ct, "A": At, "b": bt}[which]
        torch.dot(wt, qp_layer(Qt, ct, At, bt)).backward()
        return leaf.grad.numpy()

    # ---- jax grads ----
    def jax_grad(argnum: int) -> np.ndarray:
        f = lambda Qv, cv, Av, bv: jnp.dot(  # noqa: E731
            jnp.asarray(w), qp_layer_jax(Qv, cv, Av, bv)
        )
        args = (jnp.asarray(Q), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b))
        return np.asarray(jax.grad(f, argnums=argnum)(*args))

    for which, argnum in [("c", 1), ("b", 3), ("A", 2), ("Q", 0)]:
        np.testing.assert_allclose(torch_grad(which), jax_grad(argnum), rtol=1e-7, atol=1e-8)


def test_qp_layer_grad_c_finite_diff() -> None:
    Q, c, A, b = _random_qp(1)
    rng = np.random.default_rng(21)
    w = rng.standard_normal(c.shape[0])
    wt = torch.as_tensor(w, dtype=torch.float64)

    ct = _t(c, grad=True)
    torch.dot(wt, qp_layer(_t(Q), ct, _t(A), _t(b))).backward()
    grad = ct.grad.numpy()

    def eager(cc: np.ndarray) -> float:
        return float(np.dot(w, solve_qp(_t(Q), _t(cc), _t(A), _t(b)).x.numpy()))

    eps = 1e-3
    for _ in range(3):
        d = rng.standard_normal(c.shape[0])
        ad = float(np.dot(grad, d))
        fd = (eager(c + eps * d) - eager(c - eps * d)) / (2.0 * eps)
        np.testing.assert_allclose(ad, fd, atol=3e-4, rtol=3e-3)


def test_lp_layer_runs_and_is_flat_in_c() -> None:
    rng = np.random.default_rng(4)
    n = 2
    c = rng.standard_normal(n)
    A = np.vstack([rng.standard_normal((2, n)), np.eye(n), -np.eye(n)])
    b = np.concatenate([rng.uniform(0.5, 1.5, size=2), 3.0 * np.ones(2 * n)])
    ct = _t(c, grad=True)
    lp_layer(ct, _t(A), _t(b)).sum().backward()
    assert np.max(np.abs(ct.grad.numpy())) < 1e-3  # LP optimum is a vertex, flat in c
