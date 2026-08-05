# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the causal parent-ranking demo."""

from __future__ import annotations

from examples.symbolic_discovery.causal_term_discovery.benchmark import (
    VARIABLE_NAMES,
    evaluate_benchmark,
    make_dataset,
)


def test_dataset_shape_matches_named_variables() -> None:
    data = make_dataset(n_samples=256, seed=1)
    assert data.shape == (256, len(VARIABLE_NAMES))


def test_benchmark_recovers_directed_chain_and_ranks_spectator_last() -> None:
    results = evaluate_benchmark(n_samples=4000, seed=0)
    assert results["structure_exact"] is True
    assert results["recovered_edges"] == [("x0", "x1"), ("x1", "x2")]
    assert results["acyclicity_residual"] < 1e-6
    assert results["spectator_ranked_last"] is True
    assert "not a certified DAG" in results["honesty_note"]
