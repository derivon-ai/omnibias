# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Sparse l_p relaxation: torch <-> jax parity, unit box, closed-form gradient, validation."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.discrete import AnnealSchedule
from omnibias.discrete.sparse import sparse_least_squares


def _toy(seed: int = 0, m: int = 16, n: int = 6, lam: float = 0.3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    xstar = np.zeros(n)
    xstar[rng.choice(n, 2, replace=False)] = rng.standard_normal(2)
    b = A @ xstar + 0.02 * rng.standard_normal(m)
    return sparse_least_squares(A, b, lam)


@pytest.mark.parametrize("p", [1.0, 0.5, 0.1])
def test_torch_jax_bit_identical(p: float) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from omnibias.discrete.sparse.jax import sparse_relaxation as jax_relax
    from omnibias.discrete.sparse.torch import sparse_relaxation as torch_relax

    worst = 0.0
    for seed in range(4):
        prob = _toy(seed)
        xj = np.asarray(jax_relax(prob, p=p))
        xt = torch_relax(prob, p=p).detach().numpy()
        worst = max(worst, float(np.max(np.abs(xj - xt))))
    assert worst < 1e-8


def test_output_is_in_the_unit_box() -> None:
    pytest.importorskip("jax")
    from omnibias.discrete.sparse.jax import sparse_relaxation as jax_relax

    x = np.asarray(jax_relax(_toy(), p=0.5))
    assert np.all(x >= 0.0) and np.all(x <= 1.0) and np.all(np.isfinite(x))


def test_relaxation_is_finite_and_shaped_in_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.discrete.sparse.torch import sparse_relaxation as torch_relax

    out = torch_relax(_toy(), p=0.3, schedule=AnnealSchedule.fast())
    assert out.shape[0] == 6 and torch.all(torch.isfinite(out))


@pytest.mark.parametrize("p", [1.0, 0.5, 0.2])
def test_closed_form_lp_gradient_matches_finite_difference(p: float) -> None:
    # The twins descend grad_x = A^T A x - A^T b + lambda p (x + eps)^{p-1}; verify it is
    # the gradient of the stated relaxed energy 1/2||A x - b||^2 + lambda sum (x+eps)^p.
    prob = _toy(11)
    gram = prob.gram_matrix
    corr = prob.correlation
    lam = prob.lam
    eps = 1e-3

    def relaxed_energy(x: np.ndarray) -> float:
        fit = 0.5 * float(x @ gram @ x) - float(corr @ x) + 0.5 * float(prob.b @ prob.b)
        return fit + lam * float(np.sum((x + eps) ** p))

    def closed_grad(x: np.ndarray) -> np.ndarray:
        return gram @ x - corr + lam * p * (x + eps) ** (p - 1.0)

    rng = np.random.default_rng(1)
    x = 0.2 + 0.6 * rng.random(prob.n)  # interior of the box
    analytic = closed_grad(x)
    h = 1e-6
    numeric = np.array(
        [(relaxed_energy(x + h * np.eye(prob.n)[i]) - relaxed_energy(x - h * np.eye(prob.n)[i]))
         / (2 * h) for i in range(prob.n)]
    )
    assert np.max(np.abs(analytic - numeric)) < 1e-4


def test_parameter_validation() -> None:
    pytest.importorskip("jax")
    from omnibias.discrete.sparse.jax import sparse_relaxation as jax_relax

    prob = _toy()
    for bad_p in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="0 < p <= 1"):
            jax_relax(prob, p=bad_p)
    with pytest.raises(ValueError, match="eps"):
        jax_relax(prob, p=0.5, eps=0.0)
