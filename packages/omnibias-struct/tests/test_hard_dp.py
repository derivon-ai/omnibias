# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact hard DP must match the brute-force oracle on small instances (all three fronts)."""

from __future__ import annotations

import numpy as np
import pytest
from _struct_helpers import random_chain, random_dag, sample_ctc
from omnibias.struct import (
    brute_force_ctc,
    brute_force_shortest_path,
    brute_force_viterbi,
    ctc_best,
    shortest_path,
    viterbi,
)

SEEDS = range(8)


@pytest.mark.parametrize("seed", SEEDS)
def test_viterbi_matches_bruteforce(seed: int) -> None:
    trellis = random_chain(seed)
    value, path = viterbi(trellis)
    brute_value, brute_path = brute_force_viterbi(trellis)
    assert abs(value - brute_value) < 1e-9
    # the returned path is a genuine argmax path (its own score equals the optimum)
    assert abs(trellis.path_score(path) - value) < 1e-9
    assert abs(trellis.path_score(brute_path) - brute_value) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_shortest_path_matches_bruteforce(seed: int) -> None:
    dag = random_dag(seed)
    cost, path = shortest_path(dag)
    brute_cost, _ = brute_force_shortest_path(dag)
    assert abs(cost - brute_cost) < 1e-9
    assert path[0] == dag.source and path[-1] == dag.sink
    assert abs(dag.path_cost(path) - cost) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_ctc_best_matches_bruteforce(seed: int) -> None:
    lattice, log_probs = sample_ctc(seed)
    assert abs(ctc_best(lattice, log_probs) - brute_force_ctc(lattice, log_probs)) < 1e-9


def test_count_paths_and_alignments() -> None:
    trellis = random_chain(0, n_steps=4, n_states=3)
    assert trellis.count_paths() == 3**4
    dag = random_dag(0)
    # brute-force path count agrees with the closed DP count
    assert dag.count_paths() == sum(1 for _ in dag.enumerate_paths())
    lattice, _ = sample_ctc(0)
    n_align = lattice.count_alignments(4)
    brute = sum(
        1
        for idx in range(lattice.num_classes**4)
        for seq in [[(idx // lattice.num_classes**k) % lattice.num_classes for k in range(4)]]
        if lattice.collapse(seq) == tuple(int(y) for y in lattice.targets)
    )
    assert n_align == brute


def test_shortest_path_topological_order_enforced() -> None:
    from omnibias.struct import DAG

    with pytest.raises(ValueError, match="topological order"):
        DAG(3, {(2, 1): 1.0})
