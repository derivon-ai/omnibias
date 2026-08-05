# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Accelerated greedy: lazy (CELF) reproduces greedy exactly; stochastic hits its ratio."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    ONE_MINUS_INV_E,
    Coverage,
    FacilityLocation,
    PartitionMatroid,
    UniformMatroid,
    brute_force_max,
    greedy_maximize,
    lazy_greedy,
    stochastic_greedy,
)
from omnibias.submodular.matroid import Matroid


def _coverage(seed: int) -> Coverage:
    rng = np.random.default_rng(seed)
    return Coverage((rng.random((10, 8)) < 0.35).astype(float), rng.random(10) + 0.3)


def _facility(seed: int) -> FacilityLocation:
    rng = np.random.default_rng(seed)
    return FacilityLocation(rng.random((9, 8)), rng.random(9) + 0.3)


def _matroids(n: int) -> list[Matroid]:
    return [
        UniformMatroid(n, 3),
        UniformMatroid(n, 5),
        PartitionMatroid([[0, 1, 2, 3], [4, 5, 6, 7]], [2, 2]),
        PartitionMatroid([[0, 1, 2, 3, 4], [5, 6, 7]], [3, 1]),
    ]


def test_lazy_greedy_matches_greedy_value() -> None:
    # CELF exploits submodularity to skip re-evaluations but must return the same value.
    for seed in range(10):
        fn = _coverage(seed)
        for matroid in _matroids(fn.n):
            _, lazy_val = lazy_greedy(fn, matroid)
            _, greedy_val = greedy_maximize(fn, matroid)
            assert abs(lazy_val - greedy_val) < 1e-9, f"seed {seed}: {lazy_val} != {greedy_val}"


def test_lazy_greedy_matches_greedy_set_on_distinct_gains() -> None:
    # With continuous facility gains, marginal-gain ties are measure-zero, so the
    # lowest-index tie-break of the heap reproduces the exact naive-greedy set.
    for seed in range(8):
        fn = _facility(seed)
        for matroid in _matroids(fn.n):
            lazy_sel, _ = lazy_greedy(fn, matroid)
            greedy_sel, _ = greedy_maximize(fn, matroid)
            assert lazy_sel == greedy_sel, f"seed {seed}: {lazy_sel} != {greedy_sel}"


def test_lazy_greedy_is_feasible() -> None:
    for seed in range(6):
        fn = _coverage(seed)
        for matroid in _matroids(fn.n):
            sel, _ = lazy_greedy(fn, matroid)
            assert matroid.is_independent(np.asarray(sel, dtype=float))


def test_stochastic_greedy_is_feasible() -> None:
    for seed in range(6):
        fn = _coverage(seed)
        for matroid in _matroids(fn.n):
            sel, _ = stochastic_greedy(fn, matroid, epsilon=0.1, seed=seed)
            xv = np.asarray(sel, dtype=float)
            assert matroid.is_independent(xv)
            assert int(xv.sum()) <= matroid.rank()


def test_stochastic_greedy_meets_ratio_in_expectation() -> None:
    # The (1 - 1/e - eps) guarantee is in expectation over the sampling; average the
    # achieved ratio across many seeds (anti-overfit) rather than trusting one draw.
    epsilon = 0.1
    target = ONE_MINUS_INV_E - epsilon
    for inst in range(4):
        fn = _coverage(inst)
        matroid = UniformMatroid(fn.n, 4)
        _, opt = brute_force_max(fn, matroid)
        ratios = []
        for seed in range(24):
            _, val = stochastic_greedy(fn, matroid, epsilon=epsilon, seed=seed)
            ratios.append(val / opt)
        assert float(np.mean(ratios)) >= target, f"inst {inst}: mean {np.mean(ratios)} < {target}"


def test_stochastic_greedy_is_deterministic_per_seed() -> None:
    fn = _coverage(3)
    matroid = UniformMatroid(fn.n, 4)
    a = stochastic_greedy(fn, matroid, epsilon=0.1, seed=7)
    b = stochastic_greedy(fn, matroid, epsilon=0.1, seed=7)
    assert a == b


def test_stochastic_greedy_rejects_bad_epsilon() -> None:
    fn = _coverage(0)
    matroid = UniformMatroid(fn.n, 3)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="epsilon"):
            stochastic_greedy(fn, matroid, epsilon=bad)
