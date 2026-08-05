# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Non-monotone maximization: GraphCut, double greedy (1/2), measured CG (1/e), sound UB."""

from __future__ import annotations

from math import exp

import numpy as np
import pytest
from omnibias.submodular import (
    GraphCut,
    UniformMatroid,
    brute_force_max,
    certify_nonmonotone_gap,
    double_greedy,
    is_monotone_submodular,
    measured_continuous_greedy,
    nonmonotone_upper_bound,
)
from omnibias.submodular.problem import ContinuousGreedySchedule

_INV_E = exp(-1.0)


def _graph(seed: int, n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    w = np.triu(a, 1) * (rng.random((n, n)) < 0.6)
    return w + w.T  # symmetric, zero diagonal


def _path_graph(n: int = 6) -> np.ndarray:
    w = np.zeros((n, n))
    for i in range(n - 1):
        w[i, i + 1] = w[i + 1, i] = 1.0
    return w


def _unconstrained_opt(fn: GraphCut) -> float:
    # UniformMatroid(n, n) is the no-op constraint, so brute force is the global max cut.
    _, opt = brute_force_max(fn, UniformMatroid(fn.n, fn.n))
    return opt


def test_graphcut_is_submodular_and_non_monotone() -> None:
    fn = GraphCut(_path_graph())
    _, submodular = is_monotone_submodular(fn, samples=128)
    assert submodular
    # Direct non-monotone witness: the full set has an empty cut, a singleton does not.
    assert fn.value(np.ones(fn.n)) == 0.0
    assert fn.value(np.eye(fn.n)[0]) > 0.0


def test_graphcut_multilinear_matches_value_on_cube_numpy() -> None:
    for seed in range(4):
        fn = GraphCut(_graph(seed))
        rng = np.random.default_rng(seed)
        for _ in range(10):
            x = (rng.random(fn.n) < 0.5).astype(float)
            assert abs(float(fn.multilinear(x)) - float(fn.value(x))) < 1e-12


def test_graphcut_grad_matches_generic_difference() -> None:
    for seed in range(4):
        fn = GraphCut(_graph(seed))
        rng = np.random.default_rng(10 + seed)
        p = rng.random(fn.n)
        closed = fn.multilinear_grad(p)
        # dF/dp_i = F(p|p_i=1) - F(p|p_i=0), exact since F is multilinear.
        for i in range(fn.n):
            hi = p.copy()
            hi[i] = 1.0
            lo = p.copy()
            lo[i] = 0.0
            fd = float(fn.multilinear(hi)) - float(fn.multilinear(lo))
            assert abs(closed[i] - fd) < 1e-9


def test_double_greedy_deterministic_meets_one_third() -> None:
    for seed in range(8):
        fn = GraphCut(_graph(seed))
        _, val = double_greedy(fn, randomized=False)
        opt = _unconstrained_opt(fn)
        assert val <= opt + 1e-9
        if opt > 1e-9:
            assert val >= opt / 3.0 - 1e-9, f"seed {seed}: {val} < OPT/3 = {opt / 3.0}"


def test_double_greedy_randomized_meets_one_half_in_expectation() -> None:
    # The 1/2 guarantee is in expectation over the internal coin flips; average across seeds.
    for inst in range(4):
        fn = GraphCut(_graph(inst))
        opt = _unconstrained_opt(fn)
        if opt <= 1e-9:
            continue
        ratios = [double_greedy(fn, randomized=True, seed=s)[1] / opt for s in range(40)]
        assert float(np.mean(ratios)) >= 0.5 - 0.02, f"inst {inst}: mean {np.mean(ratios)}"
        assert max(ratios) <= 1.0 + 1e-9


def test_double_greedy_is_deterministic_per_seed() -> None:
    fn = GraphCut(_graph(1))
    assert double_greedy(fn, seed=5) == double_greedy(fn, seed=5)


def test_measured_continuous_greedy_matroid_meets_one_over_e() -> None:
    schedule = ContinuousGreedySchedule(steps=50, beta=50.0)
    ratios = []
    for seed in range(6):
        fn = GraphCut(_graph(seed))
        matroid = UniformMatroid(fn.n, 4)
        sel, val = measured_continuous_greedy(fn, matroid, schedule=schedule)
        assert matroid.is_independent(np.asarray(sel, dtype=float))
        _, opt = brute_force_max(fn, matroid)
        assert val <= opt + 1e-9
        if opt > 1e-9:
            ratios.append(val / opt)
    assert float(np.mean(ratios)) >= _INV_E - 0.02, f"mean {np.mean(ratios)} < 1/e"


def test_nonmonotone_upper_bound_is_sound() -> None:
    for seed in range(6):
        fn = GraphCut(_graph(seed))
        ub = nonmonotone_upper_bound(fn)
        assert _unconstrained_opt(fn) <= ub + 1e-9  # unconstrained OPT <= UB
        # any matroid-constrained optimum is <= the unconstrained one, so also <= UB
        _, opt_k = brute_force_max(fn, UniformMatroid(fn.n, 3))
        assert opt_k <= ub + 1e-9


def test_certify_nonmonotone_gap_records_ratio_and_is_sound() -> None:
    fn = GraphCut(_graph(2))
    # Unconstrained: double greedy, a-priori 1/2.
    sel, val = double_greedy(fn, seed=0)
    cert = certify_nonmonotone_gap(fn, sel)
    assert cert.method == "nonmonotone-singleton"
    assert abs(cert.approx_ratio - 0.5) < 1e-12
    assert abs(cert.value - val) < 1e-9
    assert cert.value <= _unconstrained_opt(fn) + 1e-9 <= cert.upper_bound + 1e-9
    assert cert.internal_consistent
    # Matroid: measured continuous greedy, a-priori 1/e.
    matroid = UniformMatroid(fn.n, 4)
    sel_k, _ = measured_continuous_greedy(fn, matroid)
    cert_k = certify_nonmonotone_gap(fn, sel_k, matroid=matroid)
    assert abs(cert_k.approx_ratio - _INV_E) < 1e-12
    assert cert_k.internal_consistent


def test_certify_nonmonotone_gap_rejects_infeasible_and_nonbinary() -> None:
    fn = GraphCut(_graph(0))
    matroid = UniformMatroid(fn.n, 2)
    with pytest.raises(ValueError, match="feasible"):
        certify_nonmonotone_gap(fn, [1, 1, 1, 0, 0, 0, 0, 0], matroid=matroid)
    with pytest.raises(ValueError, match="0/1"):
        certify_nonmonotone_gap(fn, np.full(fn.n, 0.5))
