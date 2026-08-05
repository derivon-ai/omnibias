# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Modern Hopfield / attention operators with closed-form lse derivatives.

Cross-backend parity (torch vs jax) on float64 inputs. Closed-form Jacobian
and Hessian are checked against autodiff references.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.hopfield.jax import ops as jhop  # noqa: E402
from omnibias.hopfield.torch import ops as thop  # noqa: E402

BETA = 2.5
A_1D = np.array([-2.0, -0.5, 0.0, 0.3, 1.7], dtype=np.float64)
RTOL = 1e-9
ATOL = 1e-11


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _ref_softmax(a: np.ndarray, beta: float = 1.0) -> np.ndarray:
    scaled = beta * a
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp_a = np.exp(shifted)
    return exp_a / exp_a.sum(axis=-1, keepdims=True)


def _ref_logsumexp(a: np.ndarray, beta: float = 1.0) -> np.ndarray:
    scaled = beta * a
    max_a = scaled.max(axis=-1)
    log_sum = max_a + np.log(np.exp(scaled - max_a[..., None]).sum(axis=-1))
    return log_sum / beta


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


def test_softmax_properties() -> None:
    a_t = torch.as_tensor(A_1D, dtype=torch.float64)
    a_j = jnp.asarray(A_1D, dtype=jnp.float64)
    for hop in (thop, jhop):
        p = _np(hop.softmax(a_t if hop is thop else a_j, beta=BETA))
        assert np.allclose(p.sum(axis=-1), 1.0, rtol=RTOL, atol=ATOL)
        ref = _ref_softmax(A_1D, beta=BETA)
        assert np.allclose(p, ref, rtol=RTOL, atol=ATOL)
    # shift invariance
    shifted = A_1D + 100.0
    p0 = _np(thop.softmax(torch.as_tensor(A_1D, dtype=torch.float64), beta=BETA))
    p1 = _np(thop.softmax(torch.as_tensor(shifted, dtype=torch.float64), beta=BETA))
    assert np.allclose(p0, p1, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Closed-form lse Jacobian / Hessian vs autodiff
# ---------------------------------------------------------------------------


def test_logsumexp_jacobian_matches_autodiff() -> None:
    a_np = A_1D
    a_t = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    lse_t = thop.logsumexp_value(a_t, beta=BETA)
    grad_t, = torch.autograd.grad(lse_t, a_t)
    jac_closed = _np(thop.logsumexp_jacobian(a_t.detach(), beta=BETA))
    assert np.allclose(jac_closed, _np(grad_t), rtol=1e-9, atol=ATOL)

    a_j = jnp.asarray(a_np, dtype=jnp.float64)
    grad_j = jax.grad(lambda x: jhop.logsumexp_value(x, beta=BETA))(a_j)
    jac_j = _np(jhop.logsumexp_jacobian(a_j, beta=BETA))
    assert np.allclose(jac_j, _np(grad_j), rtol=1e-9, atol=ATOL)


def test_logsumexp_hessian_matches_autodiff() -> None:
    a_np = A_1D

    def _lse_torch(x: torch.Tensor) -> torch.Tensor:
        return thop.logsumexp_value(x, beta=BETA)

    a_t = torch.as_tensor(a_np, dtype=torch.float64)
    hess_ad = torch.autograd.functional.hessian(_lse_torch, a_t)
    hess_closed = _np(thop.logsumexp_hessian(a_t, beta=BETA))
    assert np.allclose(hess_closed, _np(hess_ad), rtol=1e-7, atol=1e-9)
    assert np.allclose(hess_closed.sum(axis=-1), 0.0, atol=1e-9)
    eigvals = np.linalg.eigvalsh(hess_closed)
    assert np.all(eigvals >= -1e-9)

    a_j = jnp.asarray(a_np, dtype=jnp.float64)
    hess_j_ad = jax.hessian(lambda x: jhop.logsumexp_value(x, beta=BETA))(a_j)
    hess_j_closed = _np(jhop.logsumexp_hessian(a_j, beta=BETA))
    assert np.allclose(hess_j_closed, _np(hess_j_ad), rtol=1e-7, atol=1e-9)
    assert np.allclose(hess_j_closed.sum(axis=-1), 0.0, atol=1e-9)
    eigvals_j = np.linalg.eigvalsh(hess_j_closed)
    assert np.all(eigvals_j >= -1e-9)


def test_logsumexp_derivatives_random_sample() -> None:
    """Random-sample soundness: the closed-form lse Jacobian/Hessian match autodiff
    across a random sample of inputs and inverse temperatures, and the torch/jax
    closed forms agree bit-closely on each. Extends the single-vector checks above.
    """
    rng = np.random.default_rng(20)
    for _ in range(25):
        dim = int(rng.integers(2, 7))
        a_np = rng.standard_normal(dim)
        beta = float(rng.uniform(0.3, 5.0))

        a_t = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        lse_t = thop.logsumexp_value(a_t, beta=beta)
        (grad_t,) = torch.autograd.grad(lse_t, a_t)
        jac_closed = _np(thop.logsumexp_jacobian(a_t.detach(), beta=beta))
        assert np.allclose(jac_closed, _np(grad_t), rtol=1e-8, atol=1e-10)

        hess_ad = torch.autograd.functional.hessian(
            partial(thop.logsumexp_value, beta=beta), a_t.detach()
        )
        hess_closed = _np(thop.logsumexp_hessian(a_t.detach(), beta=beta))
        assert np.allclose(hess_closed, _np(hess_ad), rtol=1e-6, atol=1e-9)
        # The softmax Hessian is a symmetric PSD generator with zero row sums.
        assert np.allclose(hess_closed.sum(axis=-1), 0.0, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(hess_closed) >= -1e-8)

        a_j = jnp.asarray(a_np, dtype=jnp.float64)
        assert np.allclose(jac_closed, _np(jhop.logsumexp_jacobian(a_j, beta=beta)), rtol=RTOL, atol=ATOL)
        assert np.allclose(hess_closed, _np(jhop.logsumexp_hessian(a_j, beta=beta)), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Hopfield retrieval and energy
# ---------------------------------------------------------------------------


def test_dominant_pattern_retrieval() -> None:
    rng = np.random.default_rng(0)
    d, n = 8, 6
    X_np = 0.1 * rng.standard_normal((n, d))
    dominant = rng.standard_normal(d)
    X_np[0] = dominant
    xi_np = dominant + 0.02 * rng.standard_normal(d)
    beta = 50.0

    X_t = torch.as_tensor(X_np, dtype=torch.float64)
    xi_t = torch.as_tensor(xi_np, dtype=torch.float64)
    out_t = _np(thop.modern_hopfield_retrieve(xi_t, X_t, beta=beta))
    assert np.allclose(out_t, dominant, rtol=0.05, atol=0.05)

    X_j = jnp.asarray(X_np, dtype=jnp.float64)
    xi_j = jnp.asarray(xi_np, dtype=jnp.float64)
    out_j = _np(jhop.modern_hopfield_retrieve(xi_j, X_j, beta=beta))
    assert np.allclose(out_j, dominant, rtol=0.05, atol=0.05)


def test_retrieval_descends_energy() -> None:
    rng = np.random.default_rng(1)
    d, n = 5, 10
    for _ in range(5):
        X_np = rng.standard_normal((n, d))
        xi_np = rng.standard_normal(d)
        beta = 3.0

        X_t = torch.as_tensor(X_np, dtype=torch.float64)
        xi_t = torch.as_tensor(xi_np, dtype=torch.float64)
        e0 = _np(thop.hopfield_energy(xi_t, X_t, beta=beta))
        xi_new = thop.modern_hopfield_retrieve(xi_t, X_t, beta=beta)
        e1 = _np(thop.hopfield_energy(xi_new, X_t, beta=beta))
        assert e1 <= e0 + 1e-10

        X_j = jnp.asarray(X_np, dtype=jnp.float64)
        xi_j = jnp.asarray(xi_np, dtype=jnp.float64)
        e0_j = _np(jhop.hopfield_energy(xi_j, X_j, beta=beta))
        xi_new_j = jhop.modern_hopfield_retrieve(xi_j, X_j, beta=beta)
        e1_j = _np(jhop.hopfield_energy(xi_new_j, X_j, beta=beta))
        assert float(e1_j) <= float(e0_j) + 1e-10


def test_attention_equals_retrieve_single_query() -> None:
    rng = np.random.default_rng(2)
    d, n = 4, 7
    X_np = rng.standard_normal((n, d))
    xi_np = rng.standard_normal(d)
    beta = 2.0

    X_t = torch.as_tensor(X_np, dtype=torch.float64)
    xi_t = torch.as_tensor(xi_np, dtype=torch.float64)
    retrieve = _np(thop.modern_hopfield_retrieve(xi_t, X_t, beta=beta))
    attn = _np(thop.attention(xi_t.unsqueeze(0), X_t, X_t, beta=beta)).squeeze(0)
    assert np.allclose(retrieve, attn, rtol=RTOL, atol=ATOL)

    X_j = jnp.asarray(X_np, dtype=jnp.float64)
    xi_j = jnp.asarray(xi_np, dtype=jnp.float64)
    retrieve_j = _np(jhop.modern_hopfield_retrieve(xi_j, X_j, beta=beta))
    attn_j = _np(jhop.attention(xi_j[None, :], X_j, X_j, beta=beta)).squeeze(0)
    assert np.allclose(retrieve_j, attn_j, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------


def test_hopfield_lse_matches_struct_tower() -> None:
    """Hopfield wrappers must stay bit-identical to the shared struct lse_beta path."""
    from omnibias.struct.jax._logsumexp import (
        logsumexp_beta as j_lse,
    )
    from omnibias.struct.jax._logsumexp import (
        logsumexp_beta_hessian as j_hess,
    )
    from omnibias.struct.jax._logsumexp import (
        logsumexp_beta_jacobian as j_jac,
    )
    from omnibias.struct.torch._logsumexp import (
        logsumexp_beta as t_lse,
    )
    from omnibias.struct.torch._logsumexp import (
        logsumexp_beta_hessian as t_hess,
    )
    from omnibias.struct.torch._logsumexp import (
        logsumexp_beta_jacobian as t_jac,
    )

    a_t = torch.as_tensor(A_1D, dtype=torch.float64)
    a_j = jnp.asarray(A_1D, dtype=jnp.float64)
    assert np.allclose(
        _np(thop.logsumexp_value(a_t, beta=BETA)),
        _np(t_lse(a_t, BETA)),
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        _np(thop.logsumexp_jacobian(a_t, beta=BETA)),
        _np(t_jac(a_t, BETA)),
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        _np(thop.logsumexp_hessian(a_t, beta=BETA)),
        _np(t_hess(a_t, BETA)),
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        _np(jhop.logsumexp_value(a_j, beta=BETA)),
        _np(j_lse(a_j, BETA)),
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        _np(jhop.logsumexp_jacobian(a_j, beta=BETA)),
        _np(j_jac(a_j, BETA)),
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        _np(jhop.logsumexp_hessian(a_j, beta=BETA)),
        _np(j_hess(a_j, BETA)),
        rtol=0.0,
        atol=0.0,
    )


def test_cross_backend_parity() -> None:
    rng = np.random.default_rng(3)
    a_np = rng.standard_normal((2, 5))
    xi_np = rng.standard_normal((2, 4))
    X_np = rng.standard_normal((2, 6, 4))
    Q_np = rng.standard_normal((2, 3, 4))
    beta = BETA

    a_t = torch.as_tensor(a_np, dtype=torch.float64)
    a_j = jnp.asarray(a_np, dtype=jnp.float64)
    for name in (
        "softmax",
        "logsumexp_value",
        "logsumexp_jacobian",
        "logsumexp_hessian",
    ):
        ft = getattr(thop, name)(a_t, beta=beta)
        fj = getattr(jhop, name)(a_j, beta=beta)
        assert np.allclose(_np(ft), _np(fj), rtol=RTOL, atol=ATOL), name

    xi_t = torch.as_tensor(xi_np, dtype=torch.float64)
    X_t = torch.as_tensor(X_np, dtype=torch.float64)
    Q_t = torch.as_tensor(Q_np, dtype=torch.float64)
    xi_j = jnp.asarray(xi_np, dtype=jnp.float64)
    X_j = jnp.asarray(X_np, dtype=jnp.float64)
    Q_j = jnp.asarray(Q_np, dtype=jnp.float64)

    for name in ("modern_hopfield_retrieve", "hopfield_energy", "attention"):
        if name == "attention":
            ft = getattr(thop, name)(Q_t, X_t, X_t, beta=beta)
            fj = getattr(jhop, name)(Q_j, X_j, X_j, beta=beta)
        else:
            ft = getattr(thop, name)(xi_t, X_t, beta=beta)
            fj = getattr(jhop, name)(xi_j, X_j, beta=beta)
        assert np.allclose(_np(ft), _np(fj), rtol=RTOL, atol=ATOL), name
