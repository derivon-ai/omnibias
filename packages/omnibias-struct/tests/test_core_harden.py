# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Regressions for the hardened pure-numpy core: CTC traceback, log-space count, bounds.

Each test pins a flaw the refinement fixed: CTC had no hard traceback; ``log(N)`` needed a
bignum and blew up for infeasible / astronomical instances; the certificate crashed on a
zero-path problem; and the per-step bound needed an honest relationship to ``log(N)/beta``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _struct_helpers import dag_weight_matrix, random_chain, random_dag, sample_ctc
from omnibias.struct import (
    ChainTrellis,
    CTCLattice,
    brute_force_ctc,
    certify_soft_dp,
    count_paths,
    ctc_best,
    ctc_best_alignment,
    log_num_paths,
    logsumexp_gap_bound,
    shortest_path,
    stepwise_gap_bound,
    viterbi,
)

# --- CTC hard traceback --------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ctc_best_alignment_value_and_collapse(seed: int) -> None:
    lattice, lp = sample_ctc(seed)
    value, alignment = ctc_best_alignment(lattice, lp)
    assert abs(value - ctc_best(lattice, lp)) < 1e-12
    assert abs(value - brute_force_ctc(lattice, lp)) < 1e-9
    assert len(alignment) == lp.shape[0]
    assert lattice.collapse(alignment) == tuple(int(y) for y in lattice.targets)
    assert abs(sum(lp[t, alignment[t]] for t in range(lp.shape[0])) - value) < 1e-9


# --- log-space path count ------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_log_num_paths_matches_exact_count(seed: int) -> None:
    trellis = random_chain(seed)
    assert abs(log_num_paths(trellis) - math.log(count_paths(trellis))) < 1e-9
    dag = random_dag(seed)
    assert abs(log_num_paths(dag) - math.log(count_paths(dag))) < 1e-9
    lattice, lp = sample_ctc(seed)
    t = lp.shape[0]
    assert abs(log_num_paths(lattice, t) - math.log(count_paths(lattice, t))) < 1e-9


def test_log_num_paths_is_overflow_free() -> None:
    # 10 ** 500 paths: exact integer count is a 501-digit bignum; log stays a finite float.
    big = ChainTrellis(np.zeros((500, 10)), np.zeros((10, 10)))
    log_n = log_num_paths(big)
    assert math.isfinite(log_n)
    assert abs(log_n - 500 * math.log(10)) < 1e-9


def test_log_num_paths_infeasible_ctc_is_neg_inf() -> None:
    lattice = CTCLattice(np.array([1, 2, 1]), num_classes=3)  # L = 3 labels
    assert log_num_paths(lattice, 2) == -math.inf  # T = 2 < L: cannot emit every label


def test_certify_rejects_infeasible_zero_path_problem() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        certify_soft_dp(0.0, 0.0, 0, 4.0)


# --- per-step bound: honest relationship to the global bound -------------


def test_stepwise_bound_equals_global_for_a_chain() -> None:
    trellis = random_chain(0)
    beta = 4.0
    assert abs(stepwise_gap_bound(trellis, beta) - logsumexp_gap_bound(count_paths(trellis), beta)) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stepwise_bound_dominates_global_and_stays_sound(seed: int) -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_shortest_path

    dag = random_dag(seed, n=6)
    beta = 4.0
    hard, _ = shortest_path(dag)
    soft = float(soft_shortest_path(torch.tensor(dag_weight_matrix(dag)), dag, beta))
    realized = hard - soft  # min-sense gap >= 0
    global_bound = logsumexp_gap_bound(count_paths(dag), beta)
    stepwise = stepwise_gap_bound(dag, beta)
    assert realized <= stepwise + 1e-9  # sound
    assert global_bound <= stepwise + 1e-12  # provably dominated (never tighter)


def test_certify_with_stepwise_is_never_worse_than_global() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_viterbi

    trellis = random_chain(1)
    beta = 8.0
    hard, _ = viterbi(trellis)
    soft = float(
        soft_viterbi(
            torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), beta,
            start=torch.tensor(trellis.start),
        )
    )
    n = count_paths(trellis)
    c_global = certify_soft_dp(hard, soft, n, beta)
    c_step = certify_soft_dp(hard, soft, n, beta, stepwise_bound=stepwise_gap_bound(trellis, beta))
    assert c_step.gap_bound <= c_global.gap_bound + 1e-12
    assert c_step.is_sound
    assert "stepwise" in c_step.method
