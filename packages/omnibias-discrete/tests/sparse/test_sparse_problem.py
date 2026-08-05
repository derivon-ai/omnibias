# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Sparse problem seam: energy <-> polynomial agreement, flip fast path, refit, frontends."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.discrete.sparse import (
    BestSubsetProblem,
    SupportSelectionProblem,
    cardinality_constrained,
    sparse_least_squares,
)


def _toy(seed: int = 0, m: int = 8, n: int = 4, lam: float = 0.5):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    xstar = np.zeros(n)
    xstar[rng.choice(n, 2, replace=False)] = rng.standard_normal(2)
    b = A @ xstar + 0.02 * rng.standard_normal(m)
    return A, b, lam


def test_energy_is_the_penalised_least_squares() -> None:
    A, b, lam = _toy()
    prob = SupportSelectionProblem(A=A, b=b, lam=lam)
    x = np.array([0.3, 1.0, 0.0, 0.7])  # a continuous point
    expected = 0.5 * float((A @ x - b) @ (A @ x - b)) + lam * float(np.sum(x))
    assert float(prob.energy(x)) == pytest.approx(expected)


def test_polynomial_reproduces_energy_on_every_vertex() -> None:
    pytest.importorskip("omnibias.sos")
    A, b, lam = _toy(1)
    prob = SupportSelectionProblem(A=A, b=b, lam=lam)
    poly = prob.to_polynomial()
    for bits in itertools.product([0, 1], repeat=prob.n):
        x = np.array(bits, dtype=float)
        assert float(prob.energy(x)) == pytest.approx(float(poly.evaluate(list(bits))))


def test_flip_deltas_matches_finite_difference() -> None:
    A, b, lam = _toy(2)
    prob = SupportSelectionProblem(A=A, b=b, lam=lam)
    rng = np.random.default_rng(7)
    for _ in range(20):
        z = rng.integers(0, 2, size=prob.n).astype(float)
        fast = prob.flip_deltas(z)
        manual = np.array(
            [float(prob.energy(np.where(np.arange(prob.n) == i, 1.0 - z, z)))
             - float(prob.energy(z)) for i in range(prob.n)]
        )
        assert np.max(np.abs(fast - manual)) < 1e-9


def test_batch_energy_matches_pointwise() -> None:
    A, b, lam = _toy(3)
    prob = SupportSelectionProblem(A=A, b=b, lam=lam)
    rows = np.array(list(itertools.product([0.0, 1.0], repeat=prob.n)))
    batch = np.asarray(prob.energy(rows))
    pointwise = np.array([float(prob.energy(r)) for r in rows])
    assert np.allclose(batch, pointwise)


def test_grad_scale_is_positive() -> None:
    A, b, lam = _toy(4)
    assert SupportSelectionProblem(A=A, b=b, lam=lam).grad_scale() > 0.0


def test_shape_and_value_validation() -> None:
    with pytest.raises(ValueError, match="2-D"):
        SupportSelectionProblem(A=np.zeros(3), b=np.zeros(3))
    with pytest.raises(ValueError, match="length"):
        SupportSelectionProblem(A=np.zeros((3, 2)), b=np.zeros(4))
    with pytest.raises(ValueError, match="nonnegative"):
        SupportSelectionProblem(A=np.zeros((3, 2)), b=np.zeros(3), lam=-1.0)


def test_best_subset_energy_and_refit() -> None:
    A, b, lam = _toy(5)
    prob = BestSubsetProblem(A=A, b=b, lam=lam)
    z = np.array([1.0, 0.0, 1.0, 0.0])
    support = z >= 0.5
    a_sel = A[:, support]
    w_sel, *_ = np.linalg.lstsq(a_sel, b, rcond=None)
    resid = 0.5 * float((a_sel @ w_sel - b) @ (a_sel @ w_sel - b))
    assert float(prob.energy(z)) == pytest.approx(resid + lam * 2.0)

    w, refit_resid = prob.refit(z)
    assert w.shape == (prob.n,)
    assert np.allclose(w[~support], 0.0)  # zero off the support
    assert refit_resid == pytest.approx(resid)


def test_best_subset_empty_support_is_full_target_norm() -> None:
    A, b, lam = _toy(6)
    prob = BestSubsetProblem(A=A, b=b, lam=lam)
    z = np.zeros(prob.n)
    assert float(prob.energy(z)) == pytest.approx(0.5 * float(b @ b))


def test_frontends_dispatch_fork_and_types() -> None:
    A, b, lam = _toy(8)
    assert isinstance(sparse_least_squares(A, b, lam), SupportSelectionProblem)
    assert isinstance(sparse_least_squares(A, b, lam, continuous=True), BestSubsetProblem)
    prob = cardinality_constrained(A, b, k=2)
    assert isinstance(prob, SupportSelectionProblem)
    assert prob.lam > 0.0  # calibrated a positive penalty from the target cardinality
    with pytest.raises(ValueError, match="1 <= k <= n"):
        cardinality_constrained(A, b, k=0)
