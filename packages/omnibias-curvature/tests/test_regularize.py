# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The eps -> 0 rank / regularization collapse (JAX differentiable register).

* :func:`regularized_solve` solves ``(A + eps I) x = b`` (and ``damped_solve`` /
  ``mse_newton_step`` delegate to it, so those tests pin the bit-identical refactor).
* :func:`min_norm_solve` equals ``numpy.linalg.pinv @ b`` on full-rank *and*
  rank-deficient matrices -- the collapse limit taken stably, no blow-up.
* :func:`rank_collapse` picks a certified damping that provably meets a target
  condition number (numpy-checked) and attaches a sealed conditioning certificate.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.proof.certificate import verify_certificate_digest  # noqa: E402
from omnibias.curvature.natural_gradient import damped_solve  # noqa: E402
from omnibias.curvature.regularize import (  # noqa: E402
    CollapseResult,
    min_norm_solve,
    numerical_rank,
    rank_collapse,
    regularization_path,
    regularized_solve,
)


def _spd(rng: np.random.Generator, p: int, cond: float | None = None) -> np.ndarray:
    a = rng.normal(size=(p, p))
    m = a @ a.T + 0.1 * np.eye(p)
    if cond is not None:  # rescale spectrum to a target condition number
        w, v = np.linalg.eigh(m)
        w = np.linspace(1.0, cond, p)
        m = (v * w) @ v.T
    return 0.5 * (m + m.T)


def _low_rank(rng: np.random.Generator, p: int, r: int) -> np.ndarray:
    a = rng.normal(size=(p, r))
    m = a @ a.T
    return 0.5 * (m + m.T)


# --------------------------------------------------------------------------- #
# regularized_solve: normal equations + delegation.
# --------------------------------------------------------------------------- #
def test_regularized_solve_normal_equations() -> None:
    rng = np.random.default_rng(0)
    p = 6
    A = jnp.asarray(_spd(rng, p))
    b = jnp.asarray(rng.normal(size=p))
    eps = 1e-2
    x = regularized_solve(A, b, eps=eps)
    resid = (A + eps * jnp.eye(p)) @ x - b
    assert float(jnp.max(jnp.abs(resid))) < 1e-10


def test_regularized_solve_matches_damped_solve() -> None:
    rng = np.random.default_rng(1)
    p = 5
    A = jnp.asarray(_spd(rng, p))
    b = jnp.asarray(rng.normal(size=p))
    for eps in (0.0, 1e-6, 1e-2):
        assert jnp.allclose(
            regularized_solve(A, b, eps=eps), damped_solve(A, b, damping=eps),
            rtol=0, atol=0,  # delegation must be *bit-identical*
        )


def test_regularized_solve_guards() -> None:
    with pytest.raises(ValueError, match="square"):
        regularized_solve(jnp.zeros((2, 3)), jnp.zeros(2))
    with pytest.raises(ValueError, match="rhs must be"):
        regularized_solve(jnp.eye(3), jnp.zeros(2))
    with pytest.raises(ValueError, match="eps must be"):
        regularized_solve(jnp.eye(3), jnp.zeros(3), eps=-1.0)


# --------------------------------------------------------------------------- #
# min_norm_solve: the collapse limit == Moore-Penrose.
# --------------------------------------------------------------------------- #
def test_min_norm_solve_matches_pinv_full_rank() -> None:
    rng = np.random.default_rng(2)
    for _ in range(10):
        p = rng.integers(2, 6)
        A = _spd(rng, p)
        b = rng.normal(size=p)
        x = min_norm_solve(jnp.asarray(A), jnp.asarray(b), rcond=1e-12)
        x0 = np.linalg.pinv(A, rcond=1e-12) @ b
        assert np.allclose(np.asarray(x), x0, rtol=1e-8, atol=1e-8)


def test_min_norm_solve_matches_pinv_rank_deficient() -> None:
    rng = np.random.default_rng(3)
    for _ in range(10):
        p = int(rng.integers(3, 6))
        r = p - 1
        A = _low_rank(rng, p, r)
        y = rng.normal(size=p)
        b = A @ y  # range-consistent
        x = min_norm_solve(jnp.asarray(A), jnp.asarray(b), rcond=1e-9)
        x0 = np.linalg.pinv(A, rcond=1e-9) @ b
        assert np.allclose(np.asarray(x), x0, rtol=1e-7, atol=1e-7)
        assert numerical_rank(jnp.asarray(A), rcond=1e-9) == r


def test_min_norm_is_the_eps_to_zero_limit() -> None:
    """(A + eps I)^{-1} b -> A^+ b as eps -> 0 on a range-consistent system."""
    rng = np.random.default_rng(4)
    p, r = 5, 3
    A = jnp.asarray(_low_rank(rng, p, r))
    b = A @ jnp.asarray(rng.normal(size=p))  # consistent
    limit = min_norm_solve(A, b, rcond=1e-9)
    prev = None
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        x = regularized_solve(A, b, eps=eps)
        err = float(jnp.linalg.norm(x - limit))
        if prev is not None:
            assert err <= prev + 1e-12  # monotone approach to the collapse limit
        prev = err
    assert prev is not None and prev < 1e-5


# --------------------------------------------------------------------------- #
# regularization_path.
# --------------------------------------------------------------------------- #
def test_regularization_path_rows_match_regularized_solve() -> None:
    rng = np.random.default_rng(5)
    p = 4
    A = jnp.asarray(_spd(rng, p))
    b = jnp.asarray(rng.normal(size=p))
    grid = jnp.asarray([1e-1, 1e-2, 1e-3, 1e-4])
    path = regularization_path(A, b, grid)
    assert path.shape == (4, p)
    for k, eps in enumerate([1e-1, 1e-2, 1e-3, 1e-4]):
        assert jnp.allclose(path[k], regularized_solve(A, b, eps=eps), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# rank_collapse: certified-damped + min-norm modes.
# --------------------------------------------------------------------------- #
def test_rank_collapse_target_condition_meets_target() -> None:
    rng = np.random.default_rng(6)
    A_np = _spd(rng, 5, cond=1e6)  # deliberately ill-conditioned
    A = jnp.asarray(A_np)
    b = jnp.asarray(rng.normal(size=5))
    res = rank_collapse(A, b, target_condition=100.0)
    assert isinstance(res, CollapseResult)
    assert res.eps > 0.0
    # the certified damping provably meets the target (numpy oracle re-check)
    assert np.linalg.cond(A_np + res.eps * np.eye(5)) <= 100.0 + 1e-6
    assert res.certificate is not None and verify_certificate_digest(res.certificate)
    assert res.certificate["payload"]["type"] == "conditioning"


def test_rank_collapse_min_norm_mode() -> None:
    rng = np.random.default_rng(7)
    p, r = 5, 3
    A_np = _low_rank(rng, p, r)
    A = jnp.asarray(A_np)
    b = A @ jnp.asarray(rng.normal(size=p))
    res = rank_collapse(A, b, rcond=1e-9)  # target_condition None -> min-norm
    assert res.eps == 0.0
    assert res.effective_rank == r
    x0 = np.linalg.pinv(A_np, rcond=1e-9) @ np.asarray(b)
    assert np.allclose(np.asarray(res.solution), x0, rtol=1e-7, atol=1e-7)
    # rank-deficient: certificate is honest (kappa hi = inf, not positive definite)
    assert res.certificate is not None
    assert res.certificate["payload"]["positive_definite"] is False


def test_rank_collapse_certify_false_has_no_certificate() -> None:
    rng = np.random.default_rng(8)
    A = jnp.asarray(_spd(rng, 4))
    b = jnp.asarray(rng.normal(size=4))
    res = rank_collapse(A, b, target_condition=50.0, certify=False)
    assert res.certificate is None
