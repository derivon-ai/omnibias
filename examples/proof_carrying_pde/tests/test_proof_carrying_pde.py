# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke tests for the proof-carrying PDE example."""

from __future__ import annotations

from examples.proof_carrying_pde.benchmark import evaluate_benchmark


def test_proof_carrying_pde_benchmark() -> None:
    result = evaluate_benchmark()
    assert result["verdict"] == "PROVED"
    assert result["digest_ok"] is True
    assert result["schema_ok"] is True
    assert result["replay_ok"] is True
    assert result["honesty_ok"] is True
    assert result["unproven_claim"] is False
    assert result["error_bound"] < 1e-9
