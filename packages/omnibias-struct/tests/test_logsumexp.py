# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form lse / softmax primitives and the tower-built pairwise jet.

The Jacobian is ``softmax`` and the Hessian is ``beta (diag(p) - p p^T)`` -- checked
against autodiff -- and ``pairwise_lse_jet`` (``compose_jet`` on the softplus tower) is
checked against finite differences and across backends.
"""

from __future__ import annotations

import numpy as np
import pytest

BETA = 2.5
A_1D = np.array([-2.0, -0.5, 0.0, 0.3, 1.7], dtype=np.float64)
RTOL, ATOL = 1e-9, 1e-11


def _ref_softmax(a: np.ndarray, beta: float) -> np.ndarray:
    s = beta * a
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def _ref_lse(a: np.ndarray, beta: float) -> np.ndarray:
    s = beta * a
    m = s.max(axis=-1)
    return (m + np.log(np.exp(s - m[..., None]).sum(axis=-1))) / beta


def test_torch_primitives_vs_reference_and_autodiff() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import (
        logsumexp_beta,
        logsumexp_beta_hessian,
        logsumexp_beta_jacobian,
        softmax_beta,
    )

    a = torch.tensor(A_1D, requires_grad=True)
    val = logsumexp_beta(a, BETA)
    assert np.isclose(val.item(), _ref_lse(A_1D, BETA), rtol=RTOL, atol=ATOL)
    assert np.allclose(softmax_beta(a.detach(), BETA).numpy(), _ref_softmax(A_1D, BETA), rtol=RTOL, atol=ATOL)
    val.backward()
    jac = logsumexp_beta_jacobian(a.detach(), BETA).numpy()
    assert np.allclose(a.grad.numpy(), jac, rtol=RTOL, atol=ATOL)  # grad lse == softmax
    hess = logsumexp_beta_hessian(a.detach(), BETA).numpy()
    p = _ref_softmax(A_1D, BETA)
    assert np.allclose(hess, BETA * (np.diag(p) - np.outer(p, p)), rtol=RTOL, atol=ATOL)


def test_jax_primitives_vs_autodiff() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import (
        logsumexp_beta,
        logsumexp_beta_hessian,
        logsumexp_beta_jacobian,
    )

    a = jnp.asarray(A_1D)
    grad = np.asarray(jax.grad(lambda x: logsumexp_beta(x, BETA))(a))
    assert np.allclose(grad, np.asarray(logsumexp_beta_jacobian(a, BETA)), rtol=RTOL, atol=ATOL)
    hess = np.asarray(jax.hessian(lambda x: logsumexp_beta(x, BETA))(a))
    assert np.allclose(hess, np.asarray(logsumexp_beta_hessian(a, BETA)), rtol=RTOL, atol=ATOL)


def test_pairwise_lse_matches_lse_of_pair() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import logsumexp_beta, pairwise_lse

    a = jnp.asarray(np.array([0.3, -1.0, 2.0]))
    b = jnp.asarray(np.array([-0.4, 0.5, 1.5]))
    pair = np.asarray(pairwise_lse(a, b, BETA))
    ref = np.asarray(logsumexp_beta(jnp.stack([a, b], axis=-1), BETA, axis=-1))
    assert np.allclose(pair, ref, rtol=RTOL, atol=ATOL)


def test_pairwise_lse_jet_from_tower_matches_finite_difference() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import pairwise_lse, pairwise_lse_jet

    a0, b0, db = jnp.asarray(0.3), jnp.asarray(-0.4), jnp.asarray(1.0)
    jet = np.asarray(pairwise_lse_jet(a0, b0, db, BETA, order=2))

    def f(t: float) -> float:
        return float(pairwise_lse(a0, b0 + t * db, BETA))

    h = 1e-5
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(h) - 2 * f(0.0) + f(-h)) / (h * h)
    assert np.isclose(jet[0], f(0.0), rtol=RTOL, atol=ATOL)  # c0 = value
    assert np.isclose(jet[1], d1, atol=1e-6)  # c1 = f'(0)
    assert np.isclose(jet[2], d2 / 2.0, atol=1e-4)  # c2 = f''(0) / 2


def test_primitives_and_jets_torch_jax_parity() -> None:
    pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from omnibias.struct.jax import logsumexp_beta as lj
    from omnibias.struct.jax import pairwise_lse_jet as jj
    from omnibias.struct.torch import logsumexp_beta as lt
    from omnibias.struct.torch import pairwise_lse_jet as jt

    vt = lt(torch.tensor(A_1D), BETA).item()
    vj = float(lj(jnp.asarray(A_1D), BETA))
    assert abs(vt - vj) < ATOL
    f64 = torch.float64
    tj = np.asarray(
        jt(torch.tensor(0.3, dtype=f64), torch.tensor(-0.4, dtype=f64), torch.tensor(1.0, dtype=f64), BETA, order=3).detach()
    )
    jjj = np.asarray(jj(jnp.asarray(0.3), jnp.asarray(-0.4), jnp.asarray(1.0), BETA, order=3))
    assert np.allclose(tj, jjj, rtol=RTOL, atol=ATOL)
