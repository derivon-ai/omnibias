# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""06-01 G4/G5: every committed benchmark JSON is classified and checked."""

from __future__ import annotations

from _schema import (
    classify_artifact,
    exemption_reason,
    iter_benchmark_artifacts,
    validate_artifact,
    validate_committed_artifacts,
)


def test_every_committed_artifact_validates() -> None:
    errors = validate_committed_artifacts()
    assert errors == [], "\n".join(errors)


def test_unnamed_baseline_is_rejected() -> None:
    payload = {
        "schema": "example-v1",
        "generated_utc": "2026-08-25T00:00:00Z",
        "gates": {"all_passed": True, "entries": []},
        "baseline": {"mae": 0.1},
    }
    errors = validate_artifact(payload, name="example.json")
    assert any("missing a name" in e for e in errors)


def test_named_baseline_is_accepted() -> None:
    payload = {
        "schema": "example-v1",
        "generated_utc": "2026-08-25T00:00:00Z",
        "gates": {"all_passed": True, "entries": []},
        "baseline": {"name": "zero predictor", "mae": 1.0},
        "seeds": [0, 1, 2, 3, 4],
        "per_seed": [{"seed": i} for i in range(5)],
    }
    assert validate_artifact(payload, name="example.json") == []
    assert exemption_reason(payload) is None


def test_campaign_envelope_is_classified() -> None:
    payload = {
        "benchmark": "ipm_scaffold_smoke",
        "absolute_gates": {"gates": {"all_passed": True, "entries": []}},
    }
    assert classify_artifact(payload) == "campaign_envelope"
    assert validate_artifact(payload, name="ipm.json") == []


def test_per_seed_may_repeat_seeds_across_conditions() -> None:
    payload = {
        "schema": "example-v1",
        "generated_utc": "2026-08-25T00:00:00Z",
        "gates": {"all_passed": True, "entries": []},
        "baseline": {"name": "Monte Carlo"},
        "seeds": [0, 1],
        "per_seed": [
            {"seed": 0, "delta": 1.0},
            {"seed": 1, "delta": 1.0},
            {"seed": 0, "delta": 0.1},
            {"seed": 1, "delta": 0.1},
        ],
    }
    assert validate_artifact(payload, name="ig.json") == []


def test_directory_is_nonempty() -> None:
    assert iter_benchmark_artifacts(), "docs/benchmarks/ must keep JSON artifacts"
