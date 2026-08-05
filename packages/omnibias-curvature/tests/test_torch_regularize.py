# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The eps -> 0 collapse -- PyTorch twin + torch<->jax parity.

The torch register matches ``numpy.linalg.pinv`` / normal equations on its own,
and agrees with the JAX twin: the numerical solves to a calibrated x64 tolerance,
and the certified damping + sealed conditioning certificate **bit-identically**
(both are produced by the shared pure-Python core verifier).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.core.proof.certificate import verify_certificate_digest  # noqa: E402
from omnibias.curvature.torch.regularize import (  # noqa: E402
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
    if cond is not None:
        _, v = np.linalg.eigh(m)
        w = np.linspace(1.0, cond, p)
        m = (v * w) @ v.T
    return 0.5 * (m + m.T)


def _low_rank(rng: np.random.Generator, p: int, r: int) -> np.ndarray:
    a = rng.normal(size=(p, r))
    m = a @ a.T
    return 0.5 * (m + m.T)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# torch twin correctness.
# --------------------------------------------------------------------------- #
def test_regularized_solve_normal_equations() -> None:
    rng = np.random.default_rng(0)
    p = 6
    A = _t(_spd(rng, p))
    b = _t(rng.normal(size=p))
    eps = 1e-2
    x = regularized_solve(A, b, eps=eps)
    resid = (A + eps * torch.eye(p, dtype=torch.float64)) @ x - b
    assert float(torch.max(torch.abs(resid))) < 1e-10


def test_min_norm_solve_matches_pinv() -> None:
    rng = np.random.default_rng(1)
    for _ in range(8):
        p = int(rng.integers(3, 6))
        A = _low_rank(rng, p, p - 1)
        y = rng.normal(size=p)
        b = A @ y  # range-consistent
        x = min_norm_solve(_t(A), _t(b), rcond=1e-9)
        x0 = np.linalg.pinv(A, rcond=1e-9) @ b
        assert np.allclose(x.numpy(), x0, rtol=1e-7, atol=1e-7)
        assert numerical_rank(_t(A), rcond=1e-9) == p - 1


def test_regularization_path_rows() -> None:
    rng = np.random.default_rng(2)
    p = 4
    A = _t(_spd(rng, p))
    b = _t(rng.normal(size=p))
    grid = torch.tensor([1e-1, 1e-2, 1e-3], dtype=torch.float64)
    path = regularization_path(A, b, grid)
    assert path.shape == (3, p)
    for k, eps in enumerate([1e-1, 1e-2, 1e-3]):
        assert torch.allclose(path[k], regularized_solve(A, b, eps=eps))


def test_rank_collapse_target_condition() -> None:
    rng = np.random.default_rng(3)
    A_np = _spd(rng, 5, cond=1e6)
    res = rank_collapse(_t(A_np), _t(rng.normal(size=5)), target_condition=100.0)
    assert isinstance(res, CollapseResult)
    assert res.eps > 0.0
    assert np.linalg.cond(A_np + res.eps * np.eye(5)) <= 100.0 + 1e-6
    assert res.certificate is not None and verify_certificate_digest(res.certificate)


def test_regularized_solve_guards() -> None:
    with pytest.raises(ValueError, match="square"):
        regularized_solve(torch.zeros((2, 3), dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="eps must be"):
        regularized_solve(torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), eps=-1.0)


# --------------------------------------------------------------------------- #
# torch <-> jax parity.
# --------------------------------------------------------------------------- #
def test_torch_jax_parity() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.curvature import regularize as jreg

    rng = np.random.default_rng(20260729)
    # well-conditioned instance for the tight solve-parity check (the qubo lesson)
    A_np = _spd(rng, 5, cond=50.0)
    b_np = rng.normal(size=5)
    A_j, b_j = jnp.asarray(A_np), jnp.asarray(b_np)
    A_t, b_t = _t(A_np), _t(b_np)

    for eps in (1e-1, 1e-3, 1e-6):
        xj = np.asarray(jreg.regularized_solve(A_j, b_j, eps=eps))
        xt = regularized_solve(A_t, b_t, eps=eps).numpy()
        assert np.allclose(xj, xt, rtol=1e-9, atol=1e-9)

    # min-norm agrees (A^+ b is unique; eigh conventions differ, so looser tol)
    xj = np.asarray(jreg.min_norm_solve(A_j, b_j, rcond=1e-10))
    xt = min_norm_solve(A_t, b_t, rcond=1e-10).numpy()
    assert np.allclose(xj, xt, rtol=1e-8, atol=1e-8)

    # the certified damping + sealed certificate are bit-identical across backends
    rj = jreg.rank_collapse(A_j, b_j, target_condition=10.0)
    rt = rank_collapse(A_t, b_t, target_condition=10.0)
    assert rj.eps == rt.eps  # exact: same pure-Python certified_damping
    assert rj.certificate is not None and rt.certificate is not None
    assert rj.certificate["digest"] == rt.certificate["digest"]
    assert rj.effective_rank == rt.effective_rank
