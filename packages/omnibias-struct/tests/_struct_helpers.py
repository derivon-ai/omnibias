# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deterministic tiny problem builders shared by the omnibias-struct tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omnibias.struct import DAG, ChainTrellis, CTCLattice


def random_chain(seed: int, n_steps: int = 4, n_states: int = 3) -> ChainTrellis:
    """A random linear-chain trellis (emissions + transitions + start)."""
    rng = np.random.default_rng(seed)
    emissions = rng.standard_normal((n_steps, n_states))
    transitions = rng.standard_normal((n_states, n_states))
    start = rng.standard_normal(n_states)
    return ChainTrellis(emissions, transitions, start)


def random_dag(seed: int, n: int = 5, p: float = 0.5) -> DAG:
    """A random topological DAG; the ``u -> u+1`` chain guarantees a source->sink path."""
    rng = np.random.default_rng(seed)
    edges: dict[tuple[int, int], float] = {}
    for u in range(n):
        for v in range(u + 1, n):
            if v == u + 1 or rng.random() < p:
                edges[(u, v)] = float(0.1 + 2.0 * rng.random())
    return DAG(n, edges, source=0, sink=n - 1)


def dag_weight_matrix(dag: DAG) -> NDArray[np.float64]:
    """Dense finite-edge weight matrix (0 off the edge set) for the backend layers."""
    w = np.zeros((dag.num_nodes, dag.num_nodes))
    for (u, v), weight in dag.edges.items():
        w[u, v] = weight
    return w


def sample_ctc(seed: int, n_steps: int = 4, num_classes: int = 3) -> tuple[CTCLattice, NDArray[np.float64]]:
    """A random log-softmax emission table with a fixed 2-label target (blank = 0)."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n_steps, num_classes))
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    lattice = CTCLattice(np.array([1, 2]), num_classes=num_classes, blank=0)
    return lattice, log_probs
