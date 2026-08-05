# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the Buckingham-Pi dimensionless-group demo."""

from __future__ import annotations

from examples.symbolic_discovery.dimensional_groups.benchmark import evaluate_benchmark


def test_benchmark_recovers_reynolds_and_pendulum_groups() -> None:
    results = evaluate_benchmark()

    reynolds = results["reynolds"]
    assert reynolds["n_groups"] == 1
    assert reynolds["matches_reynolds_number"] is True
    assert reynolds["is_dimensionless"] is True

    pendulum = results["pendulum"]
    assert pendulum["n_groups"] == 1
    assert pendulum["matches_period_law"] is True
    assert pendulum["mass_exponent_is_zero"] is True
    assert pendulum["is_dimensionless"] is True


def test_library_filter_keeps_only_dimensionless_monomials() -> None:
    results = evaluate_benchmark()
    kept = results["library_filter"]["dimensionless_kept"]
    assert {"t": 2, "g": 1, "L": -1} in kept
    assert {"t": 1} not in kept
    assert {"t": 1, "g": 1} not in kept
