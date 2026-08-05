# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Each backend's soft DP equals the exact log-partition and anneals to the hard optimum."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from _struct_helpers import dag_weight_matrix, random_chain, random_dag, sample_ctc
from omnibias.struct import (
    brute_force_partition,
    certify_soft_dp,
    count_paths,
    ctc_best,
    shortest_path,
    viterbi,
)


@pytest.fixture(params=["torch", "jax"])
def backend(request: Any) -> tuple[Any, Callable[[Any], Any]]:
    """Return ``(soft_dp_module, to_tensor)`` for the requested backend (skips if absent)."""
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        import omnibias.struct.torch as mod

        return mod, (lambda a: torch.tensor(np.asarray(a, dtype=float)))
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.struct.jax as mod

    return mod, (lambda a: jnp.asarray(np.asarray(a, dtype=float)))


def test_soft_viterbi_equals_log_partition(backend: tuple[Any, Callable[[Any], Any]]) -> None:
    mod, to_t = backend
    for seed in range(4):
        trellis = random_chain(seed)
        emissions = to_t(trellis.emissions)
        transitions = to_t(trellis.transitions)
        start = to_t(trellis.start)
        for beta in (1.0, 4.0, 16.0):
            soft = float(mod.soft_viterbi(emissions, transitions, beta, start=start))
            assert abs(soft - brute_force_partition(trellis, beta)) < 1e-9


def test_soft_viterbi_anneals_and_certifies(backend: tuple[Any, Callable[[Any], Any]]) -> None:
    mod, to_t = backend
    trellis = random_chain(2)
    emissions, transitions, start = (to_t(trellis.emissions), to_t(trellis.transitions), to_t(trellis.start))
    hard, _ = viterbi(trellis)
    num_paths = count_paths(trellis)
    prev_gap = np.inf
    for beta in (1.0, 4.0, 16.0, 64.0):
        soft = float(mod.soft_viterbi(emissions, transitions, beta, start=start))
        cert = certify_soft_dp(hard, soft, num_paths, beta)
        assert cert.is_sound  # hard <= soft <= hard + log(N)/beta
        assert cert.absolute_gap <= prev_gap + 1e-12
        prev_gap = cert.absolute_gap
    assert prev_gap < 1e-3  # essentially recovered the hard optimum


def test_soft_shortest_path_equals_softmin_and_anneals(backend: tuple[Any, Callable[[Any], Any]]) -> None:
    mod, to_t = backend
    for seed in range(4):
        dag = random_dag(seed)
        weights = to_t(dag_weight_matrix(dag))
        for beta in (1.0, 8.0):
            soft = float(mod.soft_shortest_path(weights, dag, beta))
            assert abs(soft - (-brute_force_partition(dag, beta))) < 1e-9
    dag = random_dag(0)
    weights = to_t(dag_weight_matrix(dag))
    hard, _ = shortest_path(dag)
    soft_hi = float(mod.soft_shortest_path(weights, dag, 128.0))
    assert abs(soft_hi - hard) < 1e-3
    cert = certify_soft_dp(hard, soft_hi, count_paths(dag), 128.0, sense="min")
    assert cert.is_sound


def test_soft_ctc_equals_log_partition_and_anneals(backend: tuple[Any, Callable[[Any], Any]]) -> None:
    mod, to_t = backend
    for seed in range(4):
        lattice, log_probs = sample_ctc(seed)
        lp = to_t(log_probs)
        for beta in (1.0, 8.0):
            soft = float(mod.soft_ctc(lp, lattice, beta))
            assert abs(soft - brute_force_partition(lattice, beta, log_probs)) < 1e-9
    lattice, log_probs = sample_ctc(1)
    soft_hi = float(mod.soft_ctc(to_t(log_probs), lattice, 256.0))
    assert abs(soft_hi - ctc_best(lattice, log_probs)) < 1e-3
