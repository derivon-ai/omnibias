# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The ``lse_beta >= max`` sandwich certificate: sound across seeds, shrinking with beta.

Uses the exact brute-force log-partition as the soft value so the gate is a rigorous
statement about the *bound*, independent of any backend.
"""

from __future__ import annotations

import math

import pytest
from _struct_helpers import random_chain, random_dag
from omnibias.struct import (
    brute_force_partition,
    brute_force_shortest_path,
    brute_force_viterbi,
    certify_soft_dp,
    count_paths,
    logsumexp_gap_bound,
    shortest_path,
    viterbi,
)

SEEDS = range(8)
BETAS = (1.0, 4.0, 16.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_max_sense_sandwich_holds(seed: int) -> None:
    trellis = random_chain(seed)
    hard, _ = viterbi(trellis)
    brute_hard, _ = brute_force_viterbi(trellis)
    num_paths = count_paths(trellis)
    for beta in BETAS:
        soft = brute_force_partition(trellis, beta)
        cert = certify_soft_dp(hard, soft, num_paths, beta, brute_force_value=brute_hard)
        assert cert.lse_ge_max_holds  # soft >= max
        assert cert.gap_bound_holds  # soft <= max + log(N)/beta
        assert cert.is_sound
        assert cert.agrees_with_bruteforce
        assert soft >= hard - 1e-9
        assert soft <= hard + cert.gap_bound + 1e-9
        assert cert.absolute_gap <= cert.gap_bound + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_min_sense_sandwich_holds(seed: int) -> None:
    dag = random_dag(seed)
    hard_cost, _ = shortest_path(dag)
    brute_cost, _ = brute_force_shortest_path(dag)
    num_paths = count_paths(dag)
    for beta in BETAS:
        soft_cost = -brute_force_partition(dag, beta)  # max-convention value negated
        cert = certify_soft_dp(
            hard_cost, soft_cost, num_paths, beta, sense="min", brute_force_value=brute_cost
        )
        assert cert.lse_ge_max_holds  # soft_cost <= min cost
        assert cert.gap_bound_holds  # soft_cost >= min cost - log(N)/beta
        assert cert.is_sound
        assert soft_cost <= hard_cost + 1e-9
        assert soft_cost >= hard_cost - cert.gap_bound - 1e-9


def test_gap_shrinks_with_beta() -> None:
    trellis = random_chain(3)
    hard, _ = viterbi(trellis)
    num_paths = count_paths(trellis)
    gaps = [
        certify_soft_dp(hard, brute_force_partition(trellis, b), num_paths, b).absolute_gap
        for b in (1.0, 4.0, 16.0, 64.0)
    ]
    assert all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))
    assert gaps[-1] < 1e-2


def test_logsumexp_gap_bound_math() -> None:
    assert logsumexp_gap_bound(1, 2.0) == 0.0  # a single path has no gap
    assert math.isclose(logsumexp_gap_bound(8, 2.0), math.log(8) / 2.0)
    with pytest.raises(ValueError):
        logsumexp_gap_bound(0, 1.0)
    with pytest.raises(ValueError):
        logsumexp_gap_bound(4, 0.0)


def test_certificate_rejects_bad_sense() -> None:
    with pytest.raises(ValueError, match="sense"):
        certify_soft_dp(0.0, 0.0, 1, 1.0, sense="argmax")
