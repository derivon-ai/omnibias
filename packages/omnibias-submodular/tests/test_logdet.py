# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Log-determinant / DPP: monotone-submodular, Schur marginals, greedy-path + certificate."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    LogDeterminant,
    UniformMatroid,
    brute_force_max,
    certify_submodular_gap,
    greedy_maximize,
    is_monotone_submodular,
    lazy_greedy,
    log_det_dpp,
    stochastic_greedy,
    verify_guarantee,
)


def _kernel(seed: int, n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a @ a.T / n + np.eye(n) * 0.1  # symmetric, well-conditioned PD


def test_log_determinant_is_monotone_submodular() -> None:
    for seed in range(6):
        fn = LogDeterminant(_kernel(seed))
        monotone, submodular = is_monotone_submodular(fn, samples=128, seed=seed)
        assert monotone, f"seed {seed}: not monotone"
        assert submodular, f"seed {seed}: not submodular"


def test_value_matches_direct_slogdet_and_empty_is_zero() -> None:
    fn = LogDeterminant(_kernel(1))
    assert fn.value(np.zeros(fn.n)) == 0.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = (rng.random(fn.n) < 0.5).astype(float)
        sel = np.where(x > 0.5)[0]
        if sel.size == 0:
            continue
        ks = fn.kernel[np.ix_(sel, sel)]
        _, direct = np.linalg.slogdet(np.eye(sel.size) + ks)
        assert abs(float(fn.value(x)) - float(direct)) < 1e-10


def test_value_batch_matches_per_row() -> None:
    fn = LogDeterminant(_kernel(2))
    rng = np.random.default_rng(3)
    batch = (rng.random((16, fn.n)) < 0.5).astype(float)
    vals = np.asarray(fn.value(batch), dtype=float)
    for row, v in zip(batch, vals, strict=True):
        assert abs(float(fn.value(row)) - float(v)) < 1e-12


def test_schur_marginals_match_value_differences() -> None:
    for seed in range(6):
        fn = LogDeterminant(_kernel(seed))
        rng = np.random.default_rng(100 + seed)
        for _ in range(12):
            x = (rng.random(fn.n) < 0.45).astype(float)
            gains = fn.marginal_gains(x)
            base = float(fn.value(x))
            for i in range(fn.n):
                if x[i] > 0.5:
                    assert abs(gains[i]) < 1e-10  # already selected -> zero marginal
                    continue
                xi = x.copy()
                xi[i] = 1.0
                assert abs(gains[i] - (float(fn.value(xi)) - base)) < 1e-9


def test_marginals_are_nonnegative_monotone() -> None:
    for seed in range(6):
        fn = LogDeterminant(_kernel(seed))
        rng = np.random.default_rng(200 + seed)
        for _ in range(12):
            x = (rng.random(fn.n) < 0.5).astype(float)
            assert np.all(fn.marginal_gains(x) >= -1e-12)


def test_multilinear_raises_greedy_path() -> None:
    fn = LogDeterminant(_kernel(0))
    with pytest.raises(NotImplementedError, match="greedy-path"):
        fn.multilinear(np.full(fn.n, 0.5))


def test_lazy_greedy_matches_greedy_and_certificate_is_sound() -> None:
    for seed in range(6):
        prob = log_det_dpp(_kernel(seed), k=4)
        fn, matroid = prob.function, prob.matroid
        lazy_sel, lazy_val = lazy_greedy(fn, matroid)
        greedy_sel, greedy_val = greedy_maximize(fn, matroid)
        assert abs(lazy_val - greedy_val) < 1e-9
        assert lazy_sel == greedy_sel  # distinct gains -> identical set
        cert = certify_submodular_gap(prob, lazy_sel)
        _, opt = brute_force_max(fn, matroid)
        assert cert.value <= opt + 1e-9
        assert opt <= cert.upper_bound + 1e-9
        assert cert.internal_consistent
        assert verify_guarantee(prob, lazy_sel)  # greedy on a cardinality matroid: (1-1/e)


def test_stochastic_greedy_is_feasible_and_sound() -> None:
    for seed in range(6):
        prob = log_det_dpp(_kernel(seed), k=4)
        sel, val = stochastic_greedy(prob.function, prob.matroid, epsilon=0.1, seed=seed)
        xv = np.asarray(sel, dtype=float)
        assert prob.matroid.is_independent(xv)
        _, opt = brute_force_max(prob.function, prob.matroid)
        assert val <= opt + 1e-9


def test_log_det_dpp_frontend_defaults_uniform_matroid() -> None:
    prob = log_det_dpp(_kernel(0), k=3)
    assert isinstance(prob.matroid, UniformMatroid)
    assert prob.matroid.k == 3
    assert prob.function.n == 8


def test_rejects_non_psd_and_non_square_kernel() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        LogDeterminant(np.array([[1.0, 2.0], [2.0, 1.0]]))  # eigenvalues 3, -1
    with pytest.raises(ValueError, match="square"):
        LogDeterminant(np.ones((2, 3)))
