# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Sparse certificates: the A (SOS) and B (convex) sandwiches, degrade paths, and Fork C."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from omnibias.discrete import brute_force_min, certify_gap, decode, flip_deltas
from omnibias.discrete.sparse import (
    BestSubsetProblem,
    SparseFitResult,
    SupportSelectionProblem,
    certified_sparse_fit,
    certify_best_subset_gap,
    sparse_least_squares,
)


def _data(seed: int = 0, m: int = 10, n: int = 4, lam: float = 0.4):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    xstar = np.zeros(n)
    xstar[rng.choice(n, 2, replace=False)] = rng.standard_normal(2)
    b = A @ xstar + 0.02 * rng.standard_normal(m)
    return A, b, lam


def test_fork_a_sos_sandwiches_the_optimum() -> None:
    pytest.importorskip("omnibias.sos")
    A, b, lam = _data(0)
    prob = SupportSelectionProblem(A=A, b=b, lam=lam)
    _bx, e_min = brute_force_min(prob)
    assignment, _ = decode(prob, n_starts=16)
    cert = certify_gap(prob, np.array(assignment, float), level=1,
                       claim_label="sparse support-selection energy")
    assert cert.lower_bound <= e_min + 1e-6  # rigorous lower bound below the true optimum
    assert cert.energy >= e_min - 1e-9  # decoded upper bound
    assert cert.is_sound and cert.certified and cert.method == "sos"


def test_fork_a_flip_fast_path_matches_energy_only_fallback() -> None:
    A, b, lam = _data(1)
    prob = SupportSelectionProblem(A=A, b=b, lam=lam)
    rng = np.random.default_rng(3)
    for _ in range(10):
        z = rng.integers(0, 2, size=prob.n).astype(float)
        fast = flip_deltas(prob, z)  # uses SupportSelectionProblem.flip_deltas
        base = float(prob.energy(z))
        neigh = np.tile(z, (prob.n, 1))
        idx = np.arange(prob.n)
        neigh[idx, idx] = 1.0 - neigh[idx, idx]
        fallback = np.asarray(prob.energy(neigh)) - base
        assert np.max(np.abs(fast - fallback)) < 1e-9


def test_fork_b_convex_sandwiches_the_optimum() -> None:
    pytest.importorskip("omnibias.convex")
    A, b, lam = _data(2, m=12, n=4)  # overdetermined so A^T A is positive definite
    prob = BestSubsetProblem(A=A, b=b, lam=lam)
    _bx, e_min = brute_force_min(prob)
    assignment, _ = decode(prob, n_starts=16)
    cert = certify_best_subset_gap(prob, np.array(assignment, float))
    assert cert.method == "convex" and cert.certified
    assert cert.lower_bound <= e_min + 1e-6
    assert cert.is_sound


def test_fork_b_degrades_to_ols_floor_without_convex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "omnibias.convex", None)  # force the import to fail
    A, b, lam = _data(2, m=12, n=4)
    prob = BestSubsetProblem(A=A, b=b, lam=lam)
    _bx, e_min = brute_force_min(prob)
    assignment, _ = decode(prob, n_starts=16)
    cert = certify_best_subset_gap(prob, np.array(assignment, float))
    assert cert.method == "ols_floor" and not cert.certified
    assert cert.lower_bound <= e_min + 1e-9 and cert.is_sound


def test_fork_b_ols_floor_is_below_every_subset_residual() -> None:
    # The unconstrained OLS residual (all features) is a valid lower bound on E_min.
    A, b, lam = _data(4, m=12, n=5)
    prob = BestSubsetProblem(A=A, b=b, lam=lam)
    _bx, e_min = brute_force_min(prob)
    w_ols, *_ = np.linalg.lstsq(A, b, rcond=None)
    floor = 0.5 * float((A @ w_ols - b) @ (A @ w_ols - b))
    assert floor <= e_min + 1e-9


def test_certificates_reject_non_binary_points() -> None:
    A, b, lam = _data(5)
    with pytest.raises(ValueError, match="binary"):
        certify_best_subset_gap(BestSubsetProblem(A=A, b=b, lam=lam), np.full(4, 0.5))


def test_fork_c_seals_surrogate_and_refits_coefficients() -> None:
    pytest.importorskip("omnibias.sos")
    A, b, lam = _data(6, m=12, n=4)
    res = certified_sparse_fit(A, b, lam, level=1, n_starts=16)
    assert isinstance(res, SparseFitResult)
    assert res.coefficients.shape == (4,)
    # coefficients are supported exactly on the decoded support
    nz = tuple(int(i) for i in np.nonzero(res.coefficients)[0])
    assert nz == res.support
    assert res.n_selected == len(res.support)
    assert res.certificate.is_sound
    assert "surrogate" in res.note.lower()


def test_fork_c_refit_residual_matches_best_subset_energy() -> None:
    pytest.importorskip("omnibias.sos")
    A, b, lam = _data(7, m=12, n=4)
    res = certified_sparse_fit(A, b, lam, level=1, n_starts=16)
    # The refit residual is the fit term of BestSubsetProblem on the same support.
    z = np.zeros(4)
    for i in res.support:
        z[i] = 1.0
    bss = BestSubsetProblem(A=A, b=b, lam=lam)
    _w, resid = bss.refit(z)
    assert res.refit_residual == pytest.approx(resid)
